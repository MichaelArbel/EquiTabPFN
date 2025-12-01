
# Tabzilla evaluation


## Reproducing the figures of the paper from pre-computed scores

We provide pre-computed scores (accuracy, AUC, etc) on a huggingface repository to produce the main figures of the paper. The following python script will automatically download these scores and generate the figures. 

```bash

python generate_all.py

```



## Re-computing the scores

To re-compute the scores on tabzilla benchmark datasets, follow these instructions.


### Installing Tabzilla

To install Tabzilla do the following:

```bash
# clone and install dependencies
git clone https://github.com/yzeng58/tabzilla.git

source .venv/bin/activate


uv pip install -r tabzilla/TabZilla/pip_requirements.txt

# install tabpfn at commit used for this paper
# gets right commit and apply diff
pushd tabzilla
git checkout d245097b1a251a9ea7da2b3fece28f0bc8e5c377
#git apply ../notebooks/tabzilla_evaluation/diff
git apply ../evaluation/tabzilla_evaluation/changes.diff
popd

# download and preprocess all datasets
pushd tabzilla/TabZilla



python tabzilla_data_preprocessing.py --process_all
# for a single dataset, do this instead
# python tabzilla_data_preprocessing.py --dataset_name openml__Australian__146818
popd
```


### Installing TabPNF

To install TabPFN do the following:


```bash


# Installing TabPFN


git clone https://github.com/PriorLabs/TabPFN.git
pushd TabPFN
git checkout e8744e461dbd092d82389f6351f5f9cd5789d9d4
git apply ../evaluation/tabzilla_evaluation/changes_tabpfn.diff

uv pip install tabpfn_extensions==0.1.1
uv pip install -e .
popd
```




### Evaluating a pre-trained models:
Make sure your pre-trained models are saved in '../data/models/MODEL_NAME' then run: 



```bash
# eval.sh 
# the three arguments are: model, dataset and output diretory

./eval.sh EquiTabPFNV1_1_0  openml__balance-scale__11 data/eval
```



### Evaluating several pre-trained models on slurm


```bash
# eval.sh 
# the three arguments are: model, dataset and output diretory

OUTPUTDIR=my/output/dir

# Evaluation on datasets with more than 10 classes
python launch_slurm.py --output_dir $OUTPUTDIR --dataset_type more_10c

# Evaluation on datasets with less than 10 classes
python launch_slurm.py --output_dir $OUTPUTDIR --dataset_type less_10c


```




