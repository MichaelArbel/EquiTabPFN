import torch
import torch.nn as nn
from typing import Any
from equitabpfn.utils import OneHot
from torch import float64

from tabpfn.model.encoders import SeqEncStep


def handle_nan(x, num_classes):
    y = x.squeeze().long()
    mask = y == -100
    y[mask] = num_classes
    return y

def compute_lookup(x, w, b, num_classes):

    """
    x: LongTensor of shape (T, B), with values in {0..num_classes-1}
    returns: Tensor of shape (T, B, num_classes, embed_dim)
    """
    
    T,B,_ = x.shape
    V = num_classes +1
    E = w.shape[0]


    # 1) pull out weight & bias, collapse the Linear(1→E) into an E-vector
    #    weight: (E,1) → (E,)
    #w = self.layer.weight.squeeze(1)   # shape (E,)
    #b = self.layer.bias                # shape (E,)

    # 2) Build the “base” output full of bias: (T, B, V, E)
        #    by broadcasting b
    out = b.view(1, 1, 1, E).expand(T, B, V, E).clone()

    # 3) Prepare the source weight to add: shape (T, B, 1, E)
    src = w.view(1, 1, 1, E).expand(T, B, 1, E) # → (T, B, 1, E)
    # 4) Build index tensor for scatter_add
    x = handle_nan(x, num_classes)

    idx = x.view(T, B, 1, 1).expand(T, B, 1, E)

    # 5) Add w into the correct “channel” c = x[t,b]

    out.scatter_add_(2, idx, src)

    return out[:,:,:num_classes,:]




class EquivarientInputEncoderStep(SeqEncStep):
    def __init__(self,
                *,
                num_features: int,
                emsize: int,
                num_classes: int,
                replace_nan_by_zero: bool = False,
                bias: bool = True,
                in_keys: tuple[str, ...] = ("main",),
                out_keys: tuple[str, ...] = ("output",),
        ):
        
        super().__init__(in_keys, out_keys)
        self.layer = nn.Linear(num_features, emsize, bias=bias)
        self.replace_nan_by_zero = replace_nan_by_zero
        self.num_classes = num_classes
        self.one_hot = OneHot(num_classes)
        self.target_tokens= nn.Parameter(torch.randn(1, 1, 1, emsize))
        self.val_one_hot = None


    def _fit(self, *x: torch.Tensor, **kwargs: Any):
        """Fit the encoder step. Does nothing for LinearInputEncoderStep."""

    def _transform(self, *x: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor]:
        """Apply the linear transformation to the input.

        Args:
            *x: The input tensors to concatenate and transform.
            **kwargs: Unused keyword arguments.

        Returns:
            A tuple containing the transformed tensor.
        """
        if isinstance(x,tuple):
            x = x[0] # 
        single_eval_pos = kwargs['single_eval_pos']
        seq_len = x.shape[0]
        x = x[:single_eval_pos]
        x = self.one_hot(x)
        shape_x = x.shape
        out = self.layer(x.unsqueeze(-1))


        # This should add mask tokens

        target_token = self.target_tokens.repeat(
            seq_len - single_eval_pos, shape_x[1], shape_x[2], 1
        )
        out = torch.cat([out, target_token], 0)


        return (out,)



    def _transform(self, *x: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor]:
        """Apply the linear transformation to the input.

        Args:
            *x: The input tensors to concatenate and transform.
            **kwargs: Unused keyword arguments.

        Returns:
            A tuple containing the transformed tensor.
        """
        if isinstance(x,tuple):
            x = x[0] # 
        single_eval_pos = kwargs['single_eval_pos']
        seq_len = x.shape[0]

        with torch.no_grad():
            self.val_one_hot = self.one_hot(x[:single_eval_pos])
        out = self.layer(self.val_one_hot.unsqueeze(-1))

        # out = compute_lookup(x[:single_eval_pos].long(), 
        #                 self.layer.weight.squeeze(1),
        #                 self.layer.bias, 
        #                 self.num_classes)

        shape_x = out.shape

        t = self.target_tokens.view(1, 1, 1, -1)  # → (1, 1, 1, 150)


        # 2) expand it to (pad_len, B, C, 150) without copying
        pad_view = t.expand(seq_len - single_eval_pos, shape_x[1], shape_x[2], -1)


        
        # 3) allocate your output buffer once per forward
        new_out = out.new_empty(seq_len, shape_x[1], shape_x[2], shape_x[3])

        # 4) fill it via slicing
        new_out[:single_eval_pos] = out
        new_out[single_eval_pos:]  = pad_view



        #out = self.layer(x.unsqueeze(-1))
        #out = out.repeat(1,1,10,1)


        return (new_out,)












class EquiOneHotAndLinear(nn.Linear):
    def __init__(self, num_classes, emsize):
        super().__init__(1, emsize)
        self.num_classes = num_classes
        self.emsize = emsize
        self.one_hot = OneHot(num_classes)

    def forward(self, x, with_one_hot=False):
        out = self.one_hot(x)
        if x.dtype == float64:
            out = out.double()
        if with_one_hot:
            return super().forward(out.unsqueeze(-1)), out
        else:
            return super().forward(out.unsqueeze(-1))


class OneHotAndLinear(nn.Linear):
    def __init__(self, num_classes, emsize):
        super().__init__(num_classes, emsize)
        self.num_classes = num_classes
        self.emsize = emsize
        self.one_hot = OneHot(num_classes)

    def forward(self, x, with_one_hot=False):
        out = self.one_hot(x)
        if with_one_hot:
            return super().forward(out), out
        else:
            return super().forward(out)

