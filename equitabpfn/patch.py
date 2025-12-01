import numpy as np
import random




def setup_environment():

    import torch.utils.checkpoint
    original_checkpoint = torch.utils.checkpoint.checkpoint
    def patched_checkpoint(*args, use_reentrant=True, **kwargs):
        kwargs = {}
        return original_checkpoint(*args, use_reentrant=use_reentrant, **kwargs)
    torch.utils.checkpoint.checkpoint = patched_checkpoint

    import torch.cuda.amp
    def patched_autocast(*args,**kwargs):
        return torch.amp.autocast('cuda',*args, **kwargs)
    torch.cuda.amp.autocast = patched_autocast

    import sys
    import types
    dummy_mlflow = types.ModuleType("mlflow")
    dummy_mlflow.__doc__ = "This is a dummy mlflow module for monkey patching."
    sys.modules["mlflow"] = dummy_mlflow

    from mothernet.distributions import log_uniform_sampler_f

    import torch
    torch.set_float32_matmul_precision('high')
    from torch.utils.data import DataLoader
    
    from mothernet.priors.classification_adapter import BalancedBinarize, RegressionNormalized, MulticlassSteps, MulticlassRank
    from mothernet.utils import (nan_handling_missing_for_a_reason_value, nan_handling_missing_for_no_reason_value,
                             nan_handling_missing_for_unknown_reason_value, normalize_by_used_features_f, normalize_data,
                             remove_outliers)

    from mothernet.distributions import sample_distributions, uniform_int_sampler_f, parse_distributions
    from mothernet.priors.utils import CategoricalActivation, randomize_classes

    import mothernet.dataloader as dataloader
    import mothernet.priors.classification_adapter as classification_adapter












