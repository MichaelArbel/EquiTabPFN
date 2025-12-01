from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


from tabpfn.model.mlp import MLP
from tabpfn.model.multi_head_attention import MultiHeadAttention
from tabpfn.model.layer import PerFeatureEncoderLayer

from tabpfn.base import initialize_tabpfn_model
from tabpfn.utils import infer_random_state
from equitabpfn.utils import instantiate

from equitabpfn.models.decoders import KDEDecoder
from tabpfn.model.encoders import SequentialEncoder, NanHandlingEncoderStep, LinearInputEncoderStep
from tabpfn.model.encoders import VariableNumFeaturesEncoderStep, InputNormalizationEncoderStep

from equitabpfn.models.encoders import EquivarientInputEncoderStep
 
from equitabpfn.utils import OneHot

from mothernet.models.encoders import Linear, NanHandlingEncoder


DEFAULT_EMSIZE = 128

class EquiTabPFNv2(nn.Module):
    def __init__(
        self,
        compile_model: bool= False,
        y_encoder = None,
        num_classes = 10,
        load_dict=False,
    ):
        super().__init__()
        static_seed, rng = infer_random_state(None)
        model, self.config_, _ = initialize_tabpfn_model(
            model_path="auto",
            which="classifier",
            fit_mode="low_memory",
            static_seed=static_seed,
            load_dict=load_dict,
        )


        features_per_group = model.features_per_group = 2
        print(f"feature per group: {features_per_group}")


        emsize = model.encoder[-1].layer.out_features
        in_keys_x = ("main", "nan_indicators")
        if y_encoder is None:
            y_encoder = SequentialEncoder(
                NanHandlingEncoderStep(),
                EquivarientInputEncoderStep(
                    num_features=1,
                    emsize=emsize,
                    num_classes=num_classes,
                    replace_nan_by_zero=False,
                    bias=True,
                    out_keys=("output",),
                    in_keys=("main",),
                ),
            )
        preprocessor = SequentialEncoder(
                        NanHandlingEncoderStep(
                                        keep_nans = False,
                                        in_keys=("main",),
                                        out_keys=("main","nan_indicators"),
                                        ),
                        InputNormalizationEncoderStep(normalize_on_train_only=True,
                                              normalize_to_ranking=False,
                                              normalize_x=True,
                                              remove_outliers=False,
                                              in_keys=("main",),
                                              out_keys=("output",)
                                              ),
                        )

        encoder = SequentialEncoder(
                NanHandlingEncoderStep(in_keys=("main",),
                                        out_keys=("main","nan_indicators",)
                                        ),
                LinearInputEncoderStep(
                    num_features=len(in_keys_x)*features_per_group,
                    emsize=emsize,
                    replace_nan_by_zero=False,
                    bias=True,
                    in_keys=in_keys_x,
                    out_keys=("output",),
                ),
            )

        self.one_hot = OneHot(num_classes)
        self.decoder = KDEDecoder(
            bw= 1.0,
            kernel= "gaussian",
            pointwise_mlp= {"use_mlp":True,
                            "dim_feedforward":emsize,
                            "with_layer_norm":True,
                            "layer_norm_eps":1.,
                            "activation": 'gelu',
                            "dropout": 0.0} )

        self.y_encoder = y_encoder
        self.encoder = encoder
        self.transformer_encoder = model
        self.transformer_encoder.y_encoder = y_encoder
        self.transformer_encoder.encoder = encoder
        #self.transformer_encoder.preprocessor = preprocessor
        self.output_features = "all_features"
        self.flatten = True
        


        if compile_model:
            self.transformer_encoder = torch.compile(self.transformer_encoder)
            self.decoder = torch.compile(self.decoder)

    def modules_to_shard(self):
        module_list = {"level_1": [], "level_2": [], "level_3": [self.transformer_encoder], "level_4":[self]}
        for m in self.transformer_encoder.modules():
            if isinstance(m, MultiHeadAttention) or isinstance(m, MLP): 
                module_list["level_1"].append(m)
            if isinstance(m, PerFeatureEncoderLayer):
                module_list["level_2"].append(m)
        return module_list


    def forward(self, *src, **kwargs):
        assert isinstance(
            src, tuple
        ), "inputs (src) have to be given as (x,y) or (style,x,y) tuple"
        if len(src)==1:
            src = src[0]

        if len(src)==3:
            _,x,y = src
        elif len(src)==2:
            x,y = src
        single_eval_pos = kwargs['single_eval_pos']
                
        kwargs["encode_only"] = True
        src_mask = single_eval_pos

        output = self.transformer_encoder(*src, **kwargs)
        output = output.transpose(0,1)

        if self.flatten:
            output = output.view(
                output.shape[0], output.shape[1], output.shape[2] * output.shape[3]
            )

        y_one_hot = self.y_encoder[1].val_one_hot.to(dtype=output.dtype)

        output = self.decoder(
            output[src_mask:],
            output[:src_mask],
            y_one_hot[:src_mask],
            bw_token=torch.ones(output.shape[1], device=output.device),
        )
        return output




