from equitabpfn.utils import instantiate, get_original_state_dict
from huggingface_hub import snapshot_download

from pathlib import Path
import os 
import pickle
import torch
import yaml

def get_model_info(config):
    classification_cfg = config["classification"]
    is_NanHandlingEncoder = (
        (classification_cfg["nan_prob_no_reason"] > 0.0)
        or (classification_cfg["nan_prob_a_reason"] > 0.0)
        or (classification_cfg["nan_prob_unknown_reason"] > 0.0)
    )

    return {
        "num_features": config["num_features"],
        "max_num_classes": classification_cfg["max_num_classes"],
        "is_NanHandlingEncoder": is_NanHandlingEncoder,
    }


def get_model(config, model_info):
    num_features = model_info["num_features"]
    max_num_classes = model_info["max_num_classes"]

    if max_num_classes > 2:
        n_out = max_num_classes
    else:
        n_out = 1
    
    new_classes = ["equitabpfn.models.tabpfnv2.TabPFNv2",
              "equitabpfn.models.equitabpfnv2.EquiTabPFNv2",
              "equitabpfn.models.equitabpfnv2_pretrained.EquiTabPFNv2"]


    if config["bkbn"]["name"] == "mothernet.models.tabpfn.TabPFN":
        emsize = config["bkbn"]["kwargs"]["emsize"]
        y_encoder = instantiate(config["y_encoder"]["name"])(
            emsize=emsize, **config["y_encoder"]["kwargs"]
        )

        encoder = instantiate(config["encoder"]["name"])(
            num_features, emsize, **config["encoder"]["kwargs"]
        )
        model = instantiate(config["bkbn"]["name"])(
            encoder_layer=encoder,
            y_encoder_layer=y_encoder,
            n_out=n_out,
            **config["bkbn"]["kwargs"],
        )
    
    elif config["bkbn"]["name"] in new_classes: 
        model =  instantiate(config["bkbn"]["name"])(
                                **config["bkbn"]["kwargs"]
                                )
    else:
        emsize = config["bkbn"]["kwargs"]["emsize"]

        y_encoder = instantiate(config["y_encoder"]["name"])(
            emsize=emsize, **config["y_encoder"]["kwargs"]
        )
        model = instantiate(config["bkbn"]["name"])(
            y_encoder_layer=y_encoder,
            n_features=num_features,
            n_out=n_out,
            **config["bkbn"]["kwargs"],
        )

    return model


try:
    from functools import cache
except ImportError:
    from functools import lru_cache

    cache = lru_cache(maxsize=None)


# @cache
def load_model_from_states(states, device, verbose=False, uncompiled_model_keys=False):

    if uncompiled_model_keys:
        model_state = get_original_state_dict(states[0])
    else:
        model_state = states[0]
    config_sample = states[-1]
    model_info = get_model_info(config_sample["prior"])
    model = get_model(config_sample["model"], model_info)

    module_prefix = "module."
    model_state = {k.replace(module_prefix, ""): v for k, v in model_state.items()}
    model_state.pop("criterion.weight", None)

    decoder_summary_weights = [
        "query",
        "output_layer.q_proj_weight",
        "output_layer.in_proj_weight",
        "output_layer.k_proj_weight",
        "output_layer.v_proj_weight",
        "output_layer.in_proj_bias",
        "output_layer.out_proj.weight",
        "output_layer.out_proj.bias",
    ]

    for weights in decoder_summary_weights:
        full_name = "decoder." + weights
        if full_name in model_state:
            model_state["decoder.summary_layer." + weights] = model_state.pop(full_name)

    if (
        "encoder.weight" in model_state
        and "model_type" in config_sample
        and config_sample["model_type"] == "additive"
    ):
        model_state["encoder.1.weight"] = model_state.pop("encoder.weight")
        model_state["encoder.1.bias"] = model_state.pop("encoder.bias")

    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    return model, config_sample


def load_yaml_file(filename):
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
    return data

def load_model_from_name(
    root: str | Path,
    model_name: str,
    uncompiled_model_keys: bool = False,
    device: str = "cpu",
):
    name = f"{model_name}"
    model_path =  os.path.join(root, model_name)
    if not os.path.exists(model_path):
        snapshot_download(
            repo_id="equitabpfn/checkpoints",
            repo_type="dataset",
            allow_patterns=f"{model_name}*",
            local_dir=root,
            force_download=False,
        )
    
    file_name = os.path.join(model_path, "artifacts", "ckpt/last.pickle" )
    config = load_yaml_file(os.path.join(model_path, "metadata","config.yaml"))
    try: 
        
        state_dict, _ = torch.load(
                file_name, map_location=device, weights_only=False
            )
    
    except:

        with open(file_name, "rb") as f:
            ckpt = pickle.load(f)
        state_dict = ckpt["model"]

        
    states = state_dict, config

    model, config = load_model_from_states(
        states,
        device=device,
        verbose=False,
        uncompiled_model_keys=uncompiled_model_keys
    )
    return model, config



def load_model(model_path, device, verbose=False):
    import torch

    states = torch.load(model_path, map_location=device, weights_only=False)
    return load_model_from_states(states, device=device, verbose=False)
