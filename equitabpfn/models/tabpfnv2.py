from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


from tabpfn.model.mlp import MLP
from tabpfn.model.multi_head_attention import MultiHeadAttention
from tabpfn.model.layer import PerFeatureEncoderLayer

from tabpfn.base import initialize_tabpfn_model

from tabpfn.utils import infer_random_state

from mothernet.prediction.tabpfn import load_model_workflow

import importlib
import inspect
import pathlib

def get_module_path(module_name):
    try:
        module = importlib.import_module(module_name)
        return inspect.getfile(module)
    except (ImportError, TypeError):
        return None
class TabPFNv2(nn.Module):
    def __init__(
        self,
        compile_model: bool= False,
        load_dict: bool = False,
        model_version: str = "V2",
    ):
        super().__init__()
        static_seed, rng = infer_random_state(None)
        if model_version=="V2":
            model, self.config_, _ = initialize_tabpfn_model(
                model_path="auto",
                which="classifier",
                fit_mode="low_memory",
                static_seed=static_seed,
                load_dict=load_dict
            )
            base_path = pathlib.Path(get_module_path("mothernet.prediction.tabpfn")).parent.parent.resolve()
            _, self.config_, _ = load_model_workflow(-1, add_name='download', base_path=base_path, device='cpu',
                                                         eval_addition='')
        else:
            base_path = pathlib.Path(get_module_path("mothernet.prediction.tabpfn")).parent.parent.resolve()
            model, self.config_, _ = load_model_workflow(-1, add_name='download', base_path=base_path, device='cpu',
                                                         eval_addition='')
        #model.features_per_group = 100


        self.transformer_encoder = model
        # assert hasattr(y_encoder.forward, "with_one_hot")
        if compile_model:
            self.transformer_encoder = torch.compile(self.transformer_encoder)

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
        print(lol)
        return self.transformer_encoder(*src, **kwargs)



