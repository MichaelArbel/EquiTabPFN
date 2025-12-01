import os
import random
import numpy as np
import torch
from torch import nn
import importlib
from torch import Tensor
from collections import OrderedDict
from collections import UserDict



import os
import json
import pickle
import time
from datetime import datetime
import yaml

class SimpleFSLogger:
    """Filesystem-based logger fully compatible with your Trainer."""

    def __init__(self, root_dir="logs", run_name=None):
        self.root_dir = root_dir
        os.makedirs(root_dir, exist_ok=True)

        if run_name is None:
            run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.run_dir = os.path.join(root_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        self.metrics_dir = os.path.join(self.run_dir, "metrics")
        self.artifacts_dir = os.path.join(self.run_dir, "artifacts")
        self.metadata_dir = os.path.join(self.run_dir, "metadata")
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

        print(f"[Logger] Logging into: {self.run_dir}")

    def log_metadata(self, configs):
        """Append metrics to logs/metrics/<log_name>.jsonl"""
        config_path = os.path.join(self.metrics_dir, f"config.yaml")


        with open(config_path, "a") as f:
            yaml.dump(configs, f, default_flow_style=False)
    # -----------------------------------------------
    def log_metrics(self, metrics: dict, log_name: str = "metrics"):
        """Append metrics to logs/metrics/<log_name>.jsonl"""
        metrics_path = os.path.join(self.metrics_dir, f"{log_name}.json")

        entry = {
            "time": time.time(),
            **metrics
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # -----------------------------------------------
    def log_artifacts(self, obj, artifact_name: str, artifact_type: str = "pickle"):
        """
        artifact_name may include subfolders: "ckpt/last"
        """
        path = os.path.join(self.artifacts_dir, f"{artifact_name}.{artifact_type}")

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(obj, f)

        print(f"[Logger] Saved artifact → {path}")

    # -----------------------------------------------
    def load_artifacts(self, artifact_name: str, artifact_type: str = "pickle"):
        """Load saved artifact"""
        path = os.path.join(self.artifacts_dir, f"{artifact_name}.{artifact_type}")

        if not os.path.exists(path):
            print(f"[Logger] Artifact not found: {path}")
            return None

        with open(path, "rb") as f:
            print(f"[Logger] Loaded artifact ← {path}")
            return pickle.load(f)



# class ConfigDict(UserDict):
#     """A dict subclass that allows attribute-style access (dot-access)."""

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # recursively convert nested dicts
#         for key, value in self.data.items():
#             self.data[key] = self._convert(value)

#     def _convert(self, value):
#         if isinstance(value, dict):
#             return ConfigDict(value)
#         elif isinstance(value, list):
#             return [self._convert(v) for v in value]
#         return value

#     # Dot-access methods
#     def __getattr__(self, key):
#         try:
#             return self.data[key]
#         except KeyError:
#             raise AttributeError(f"'ConfigDict' object has no attribute '{key}'")

#     def __setattr__(self, key, value):
#         if key == "data":
#             super().__setattr__(key, value)
#         else:
#             self.data[key] = self._convert(value)

#     def __delattr__(self, key):
#         try:
#             del self.data[key]
#         except KeyError:
#             raise AttributeError(f"'ConfigDict' object has no attribute '{key}'")


class ConfigDict(dict):
    """A dict subclass that allows attribute-style access (dot-access)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # recursively convert nested dicts
        for key, value in list(self.items()):
            super().__setitem__(key, self._convert(value))

    def _convert(self, value):
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            return ConfigDict(value)
        elif isinstance(value, list):
            return [self._convert(v) for v in value]
        return value

    # dot-access methods
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = self._convert(value)

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{key}'")

    def __repr__(self):
        return f"{self.__class__.__name__}({dict(self)})"




def batched_pca(data, num_components=100):
    """
    Perform PCA independently on each of the B datasets in a tensor of shape (N, B, D).
    Projects each dataset to the top `num_components` principal components.
    
    Args:
        data (torch.Tensor): Input tensor of shape (N, B, D)
        num_components (int): Number of principal components to retain

    Returns:
        torch.Tensor: PCA-projected data of shape (N, B, num_components)
    """
    N, D = data.shape
    device = data.device




    X = data # Shape: (N, D)

    # Center the data
    mean = X.mean(dim=0, keepdim=True)
    X_centered = X - mean

    # Compute covariance matrix: (D x D)
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False) 

    # Sort eigenvalues and eigenvectors in descending order
    V_top = Vh[:num_components, :]  # (num_components, D)

    # Project onto principal components
    proj = X_centered @ V_top.T

    # Project data

    return proj, mean, V_top

def apply_pca(X, mean, V_top):
    X_centered = X-mean
    return X_centered @V_top.T



def squared_dist(X, X_2):
    norm_2 = torch.sum(X**2, axis=-1).unsqueeze(1) + torch.sum(
        X_2**2, axis=-1
    ).unsqueeze(0)
    norm_2 -= 2 * torch.einsum("ibd,jbd->ijb", X, X_2)
    return norm_2


def gaussian_kernel(X, X_2, sigma=1.0):
    sigma *= np.sqrt(X.shape[-1])  ### normalize by feature dimension
    norm_2 = squared_dist(X, X_2)
    if isinstance(sigma, Tensor):
        assert sigma.shape[0] == norm_2.shape[-1]
        exponant = -0.5 * torch.einsum("nmb,b->nmb", norm_2, 1.0 / sigma)
    else:
        exponant = -(0.5 / sigma) * squared_dist(X, X_2)
    K = torch.softmax(exponant, dim=0)
    return K


def exp_kernel(X, X_2, sigma):
    sigma *= np.sqrt(X.shape[-1])
    exponant = torch.einsum("ibd,jbd, b->ijb", X, X_2, 1.0 / sigma)
    K = torch.softmax(exponant, dim=0)
    return K


def var(X):
    sec_mom = torch.sum(X**2, axis=-1).mean(axis=0)
    mean = (torch.mean(X, axis=0) ** 2).sum(axis=-1)
    return 2 * (sec_mom - mean)


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Boolean value expected.")


def instantiate(module_name):
    module, attr = os.path.splitext(module_name)
    try:
        module = importlib.import_module(module)
        return getattr(module, attr[1:])
    except BaseException:
        try:
            module = instantiate(module)
            return getattr(module, attr[1:])
        except BaseException:
            return eval(module + attr[1:])


def assign_device(device):
    """
    Assigns a device for PyTorch based on the provided device identifier.

    Parameters:
    - device (int): Device identifier. If positive, it represents the GPU device
                   index; if -1, it sets the device to 'cuda'; if -2, it sets
                   the device to 'cpu'.

    Returns:
    - device (str): The assigned device, represented as a string.
                    'cuda:X' if device > -1 and CUDA is available, where X is
                    the provided device index. 'cuda' if device is -1.
                    'cpu' if device is -2.
    """
    if device > -1:
        device = (
            "cuda:" + str(device)
            if torch.cuda.is_available() and device > -1
            else "cpu"
        )
    elif device == -1:
        device = "cuda"
    elif device == -2:
        device = "cpu"
    return device


def get_dtype(dtype):
    """
    Returns the PyTorch data type based on the provided integer identifier.

    Parameters:
    - dtype (int): Integer identifier representing the desired data type.
                   64 corresponds to torch.double, and 32 corresponds to torch.float.

    Returns:
    - torch.dtype: PyTorch data type corresponding to the provided identifier.

    Raises:
    - NotImplementedError: If the provided identifier is not recognized (not 64 or 32).
    """
    if dtype == 64:
        return torch.double
    elif dtype == 32:
        return torch.float
    else:
        raise NotImplementedError("Unkown type")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_init_method(init_method):
    if init_method is None:
        return None
    if init_method == "kaiming-uniform":
        method = nn.init.kaiming_uniform_
    if init_method == "kaiming-normal":
        method = nn.init.kaiming_normal_
    if init_method == "xavier-uniform":
        method = nn.init.xavier_uniform_
    if init_method == "xavier-normal":
        method = nn.init.xavier_normal_

    def init_weights_inner(layer):
        if isinstance(layer, nn.Linear):
            method(layer.weight)
            nn.init.zeros_(layer.bias)

    return init_weights_inner


def get_original_state_dict(state_dict):
    prefix = "_orig_mod."
    return OrderedDict((k.replace(prefix,''), v) for k, v in state_dict.items())











class Activation(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def forward(self, x):
        return self.activation(x)


class OneHot(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        if (x == -100).any():
            pass
        y = x.squeeze().long()
        mask = y == -100
        y[mask] = 0
        out = torch.nn.functional.one_hot(y, self.num_classes).float()

        if out.ndim == 3:
            out[:, :, 0][mask] = 0
        else:
            out[:, 0][mask] = 0
            out = out.unsqueeze(1)
        return out


class Criterion:
    def __init__(self, max_num_classes, logits=True):
        self.logits = logits
        if max_num_classes == 2:
            self._criterion = nn.BCEWthLogitsLoss(reduction="none")

            def eval_func(output, targets):
                if not self.logits:
                    output = torch.log(output)
                return self._criterion(output.flatten(), targets.flatten())

            self.eval_func = eval_func
        elif max_num_classes > 2:
            self._criterion = nn.CrossEntropyLoss(reduction="none")
            self.n_out = max_num_classes
            if self.logits:

                def eval_func(output, targets):
                    return self._criterion(
                        output.reshape(-1, self.n_out), targets.long().flatten()
                    )

            else:
                self.one_hot = OneHot(self.n_out)

                def eval_func(output, targets):
                    one_hot_targets = (
                        self.one_hot(targets).reshape(-1, self.n_out).long()
                    )
                    return (
                        (output.reshape(-1, self.n_out) - one_hot_targets) ** 2
                    ).mean(dim=-1)

            self.eval_func = eval_func
        else:
            raise ValueError(f"Invalid number of classes: {max_num_classes}")

    def __call__(self, output, targets):
        return self.eval_func(output, targets)


def save_model(model, optimizer, scheduler, path, filename, config_sample):
    optimizer_dict = optimizer.state_dict() if optimizer is not None else None

    import cloudpickle

    torch.save(
        (model.state_dict(), optimizer_dict, scheduler, config_sample),
        os.path.join(path, filename),
        pickle_module=cloudpickle,
    )


def figure_path():
    import equitabpfn
    from pathlib import Path

    path = Path(equitabpfn.__path__[0]).parent / "figures"
    path.mkdir(exist_ok=True)
    return path
