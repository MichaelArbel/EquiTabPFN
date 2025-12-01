# EquiTabPFN

This repository provides code for the paper 
[EquiTabPFN: A Target-Permutation Equivariant Prior Fitted Network](https://neurips.cc/virtual/2025/poster/118521) 
including:
- code to launch training
- code to launch evaluations
- code to reproduce figures

## Installation

To install run the following:
```bash

# if you don't have `uv`, install it, for instance by doing 
curl -LsSf https://astral.sh/uv/install.sh | sh

# then install equitabpfn and its dependencies 
git clone git@github.com:MichaelArbel/EquiTabPFN-dev.git
cd EquiTabPFN-dev
git checkout setup

uv venv

source .venv/bin/activate

uv pip install -r requirements.txt
uv pip install -e .

# install Mothernet at commit used for this paper
git clone https://github.com/microsoft/ticl
pushd ticl
git checkout 4cb7ac0fc04aa15256995dd85f379562ea6e994c
uv pip install -e .
popd


```

Then run this to test that your environment works:

```bash
# To test that your setup work
uv run pytest tst
```



## Reproducing

### Training

To train Equitabpfn using the same configuration as in the paper, run


```bash
# Run the training script after 
python equitabpfn/main.py \
                --data_path "data"\
                --config_dir  "equitabpfn/configs/"\
                --output_dir "data/outputs/"\
                --run_name "test/"

```


### Loading pre-trained models


```python

checkpoint_path = 'data/models'
from equitabpfn.models.model_builder import load_model_from_name
model, config = load_model_from_name(root=checkpoint_path, model_name="equitabpfn")

```


### TabZilla Evaluations

To rerun TabZilla evaluations see evaluation/README.md



### Figure and analysis

Scripts to reproduce figures and analysis are available under evaluation directory. Please follow instructions in evaluation/README.md


## Citation

In case this work is useful for your research, please cite the following paper:

```bibtex
@inproceedings{
arbel2025equitabpfn,
title={EquiTab{PFN}: A Target-Permutation Equivariant Prior Fitted Network},
author={Michael Arbel and David Salinas and Frank Hutter},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=LrnZDU9g7N}
}
```