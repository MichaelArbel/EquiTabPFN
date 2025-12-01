import torch
from equitabpfn.utils import Activation, squared_dist, gaussian_kernel, exp_kernel
from torch.nn.modules.transformer import Dropout, LayerNorm, Linear, _get_activation_fn
from torch import nn
import numpy as np




class MLPDecoder(nn.Module):
    def __init__(self, device=None, dtype=None, **kwarg):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.kwarg = kwarg
        emsize = kwarg["emsize"]
        if "pointwise_mlp" in self.kwarg:
            pointwise_mlp = self.kwarg["pointwise_mlp"]
            if "use_mlp" not in self.kwarg["pointwise_mlp"]:
                self.kwarg["pointwise_mlp"]["use_mlp"] = True
        else:
            pointwise_mlp = {"use_mlp": False}

        if pointwise_mlp["use_mlp"]:
            dim_feedforward = pointwise_mlp["dim_feedforward"]
            if pointwise_mlp["with_layer_norm"]:
                LN = LayerNorm(
                    dim_feedforward,
                    eps=pointwise_mlp["layer_norm_eps"],
                    **factory_kwargs
                )
            else:
                LN = torch.nn.Identity()
            self.head = nn.Sequential(
                Linear(emsize, dim_feedforward, **factory_kwargs),
                LN,
                Activation(_get_activation_fn(pointwise_mlp["activation"])),
                Dropout(pointwise_mlp["dropout"]),
                Linear(dim_feedforward, 1, **factory_kwargs),
                Dropout(pointwise_mlp["dropout"]),
            )
        else:
            self.head = Linear(emsize, 1, **factory_kwargs)

    def forward(self, X, X_tr, Y_tr, bw_token):
        m, b, f, d = X.shape
        return self.head(X).squeeze(-1)







class KDEDecoder(nn.Module):
    def __init__(self, device=None, dtype=None, **kwarg):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.kwarg = kwarg
        if "bw" not in self.kwarg:
            self.kwarg["bw"] = 1.0
        if "kernel_type" in kwarg:
            kwarg["kernel"] = kwarg["kernel_type"]
        if "kernel" not in self.kwarg:
            kwarg["kernel"] = "gaussian"
        if kwarg["kernel"] == "gaussian":
            self.kernel = gaussian_kernel
        elif kwarg["kernel"] == "exp":
            self.kernel = exp_kernel
        if "pointwise_mlp" in self.kwarg:
            pointwise_mlp = self.kwarg["pointwise_mlp"]
            dim_feedforward = pointwise_mlp["dim_feedforward"]
            if pointwise_mlp["with_layer_norm"]:
                LN = LayerNorm(
                    dim_feedforward,
                    eps=pointwise_mlp["layer_norm_eps"],
                    **factory_kwargs
                )
            else:
                LN = torch.nn.Identity()
            self.pointwise_mlp = nn.Sequential(
                Linear(1, dim_feedforward, **factory_kwargs),
                LN,
                Activation(_get_activation_fn(pointwise_mlp["activation"])),
                Dropout(pointwise_mlp["dropout"]),
                Linear(dim_feedforward, 1, **factory_kwargs),
                Dropout(pointwise_mlp["dropout"]),
            )

        else:
            self.pointwise_mlp = None

    def forward(self, X, X_tr, Y_tr, bw_token):
        #kernel = self.kernel(X_tr, X, self.kwarg["bw"] * bw_token)
        #output = torch.einsum("nbd,nmb->mbd", Y_tr, kernel)  # Maybe use original y
        X = X.permute(1,0,2).unsqueeze(1)
        X_tr = X_tr.permute(1,0,2).unsqueeze(1)
        Y_tr = Y_tr.permute(1,0,2).unsqueeze(1)
        output = torch.nn.functional.scaled_dot_product_attention(X, X_tr, Y_tr)

        if self.pointwise_mlp is not None:
            output = output + (self.pointwise_mlp(output.unsqueeze(-1))).squeeze(-1)
        output = output.squeeze(1).permute(1,0,2)
        return output


