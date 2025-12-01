from typing import Optional
import torch
import torch.nn as nn
from torch.nn.modules.transformer import Optional, Tensor
from equitabpfn.models.layer import TransformerEncoderLayer
from mothernet.utils import SeqBN
from mothernet.models.encoders import Linear, NanHandlingEncoder
from equitabpfn.utils import batched_pca, OneHot, get_init_method, instantiate
from equitabpfn.models.decoders import KDEDecoder
from torch.utils.checkpoint import checkpoint



class BiAttentionEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=False,
        pre_norm=False,
        device=None,
        dtype=None,
        recompute_attn=False,
        feature_mask_mode="Bq2Bk" # [none, ABq2Bk, Bq2Bk, Aq2Bk, Bq2Ak] discards corresponding edges between queries and keys of type A or B
    ):
        super().__init__()
        self.cross_feature_attention = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation
        )
        self.cross_sample_attention = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
        )
        self.feature_mask_mode= feature_mask_mode

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, input_mask:Optional[Tensor] = None) -> Tensor:
        # src_mask is in with eval position, applies only to samples
        # src comes in as samples x batch x feature x emsize
        # reshape to features x (samples * batch) x emsize for cross-feature attention
        reshaped = src.reshape(-1, *src.shape[2:]).transpose(0, 1)
        # post_feature_attention = checkpoint(self.cross_feature_attention,reshaped, None, use_reentrant = True)
        post_feature_attention = self.cross_feature_attention(reshaped, input_mask,mask_mode=self.feature_mask_mode)
        

        # from cross-feature attention, we get features x (samples * batch) x emsize
        # reshape back to original, then reshape to samples x (batch * feature) x emsize
        reshaped = post_feature_attention.transpose(0, 1).reshape(src.shape)
        reshaped = reshaped.reshape(src.shape[0], -1, src.shape[-1])
        res = self.cross_sample_attention(reshaped, src_mask, mask_mode="ABq2Bk")
        return res.reshape(src.shape)


class EquiTabPFN(nn.Module):
    def __init__(
        self,
        *,
        n_out: int,
        emsize: int,
        nhead: int,
        nhid_factor: int,
        nlayers: int,
        n_features: int,
        dropout: float = 0.0,
        y_encoder_layer,
        decoder_kwarg: dict = None,
        input_normalization: bool = False,
        init_method: str | None = None,
        pre_norm: bool = False,
        activation="gelu",
        recompute_attn: bool = False,
        all_layers_same_init: bool = False,
        efficient_eval_masking: bool = True,
        tabpfn_zero_weights: bool = False,
        output_features: str = "all_features",
        equivariant_encoder: bool = False,
        feature_mask_mode: str = "Bq2Bk",
        compile_model: bool= False
    ):
        super().__init__()
        self.equivariant_encoder = equivariant_encoder
        self.y_encoder = y_encoder_layer
        nhid = emsize * nhid_factor

        def encoder_layer_creator():
            return BiAttentionEncoderLayer(
                emsize,
                nhead,
                nhid,
                dropout,
                activation=activation,
                pre_norm=pre_norm,
                recompute_attn=recompute_attn,
                feature_mask_mode=feature_mask_mode
            )

        self.transformer = TransformerEncoderDiffInit(encoder_layer_creator, nlayers)
        if compile_model:
            self.transformer = torch.compile(self.transformer)
        #self.layers = nn.ModuleList([encoder_layer_creator() for _ in range(nlayers)])
        self.emsize = emsize
        if self.equivariant_encoder == True:
            self.encoder = NanHandlingEncoder(1, emsize)
        else:
            self.encoder = NanHandlingEncoder(n_features, emsize)

        self.decoder = (
            instantiate(decoder_kwarg["name"])(**decoder_kwarg["kwargs"])
            if decoder_kwarg is not None
            else KDEDecoder()
        )
        self.flatten = True
        self.output_features = output_features
        # if isinstance(self.decoder, MLPDecoder):
        #     self.output_features = "y_features"
        #     self.flatten = False

        self.target_token = nn.Parameter(torch.randn(1, 1, 1, emsize))
        self.input_ln = SeqBN(emsize) if input_normalization else None
        self.init_method = init_method
        self.tabpfn_zero_weights = tabpfn_zero_weights
        self.n_out = n_out
        self.nhid = nhid

        self.preprocessor = None

        self.init_weights()

    def init_weights(self):
        if self.init_method is not None:
            self.apply(get_init_method(self.init_method))
        if self.tabpfn_zero_weights:
            for bilayer in self.layers:
                for layer in [
                    bilayer.cross_feature_attention,
                    bilayer.cross_sample_attention,
                ]:
                    nn.init.zeros_(layer.linear2.weight)
                    nn.init.zeros_(layer.linear2.bias)
                    attns = (
                        layer.self_attn
                        if isinstance(layer.self_attn, nn.ModuleList)
                        else [layer.self_attn]
                    )
                    for attn in attns:
                        nn.init.zeros_(attn.out_proj.weight)
                        nn.init.zeros_(attn.out_proj.bias)

    def forward(self, src, src_mask=None, single_eval_pos=None,
            only_return_standard_out=True,
                    categorical_inds=[]):
        assert isinstance(
            src, tuple
        ), "inputs (src) have to be given as (x,y) or (style,x,y) tuple"
        if len(src) == 3:  # style is given
            style_src, x_src, y_src = src
        else:
            x_src, y_src = src
        if src_mask is None:
            src_mask = single_eval_pos

        ## This should be a one_hot_encoder
        y_src, y_src_one_hot = self.y_encoder(
            y_src.unsqueeze(-1) if len(y_src.shape) < len(x_src.shape) else y_src,
            with_one_hot=True,
        )
        
        #if self.preprocessor:
        #   x_src = self.preprocessor({"main":x_src}, single_eval_pos=src_mask)

        if x_src.shape[-1] <=100:
            zero_fill = torch.zeros(x_src.shape[:2]+(100-x_src.shape[-1],), device= x_src.device)
            x_src = torch.cat([x_src, zero_fill],2)
        else:
            x_src = batched_pca(x_src)          

        if self.equivariant_encoder:
            x_src = self.encoder(x_src.unsqueeze(-1))
        else:
            x_src = self.encoder(x_src).unsqueeze(-2)
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-2)
        shape_y = y_src.shape
        shape_x = x_src.shape
        target_token = self.target_token.repeat(
            x_src.shape[0] - src_mask, shape_y[1], shape_y[2], 1
        )
        y_src = torch.cat([y_src[:src_mask], target_token], 0)
        #src = torch.cat([y_src, x_src], 2)
        
        src = torch.cat([x_src,y_src], 2)

        if self.input_ln is not None:
            src = self.input_ln(src)
        output = src
        #with torch.backends.cuda.enable_mem_efficient_sdp():
        output = self.transformer(src, src_mask=src_mask, input_mask=shape_x[2])
        #for mod in self.layers:
        #    output = mod(output, src_mask=src_mask, input_mask=shape_x[2])

        if self.output_features == "y_features":
            output = output[:, :, shape_x[2] :, :]
        elif self.output_features == "x_features":
            output = output[:, :, : shape_x[2], :]
        if self.flatten:
            output = output.view(
                output.shape[0], output.shape[1], output.shape[2] * output.shape[3]
            )
        output = self.decoder(
            output[src_mask:],
            output[:src_mask],
            y_src_one_hot[:src_mask],
            bw_token=torch.ones(output.shape[1], device=output.device),
        )

        return output

class TransformerEncoderDiffInit(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer_creator: a function generating objects of TransformerEncoderLayer class without args (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer_creator, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer_creator() for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, input_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=src_mask, input_mask=input_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output