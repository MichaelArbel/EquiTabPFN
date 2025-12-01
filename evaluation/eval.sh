#!/bin/bash


# dataset=$1
# model=$2
# output_dir=$3





model="EquiTabPFNV1_1_0"
dataset="openml__balance-scale__11"
output_dir="data/eval"
TABZILLAPATH="tabzilla/TabZilla/"


IFS=',' read -r -a models <<< "$model"


echo "Arg1: $arg1"
echo "List items:"
for m in "${models[@]}"; do
	echo "Running evaluation on $dataset for $m"
	python -m ipdb $TABZILLAPATH/tabzilla_experiment.py \
	--model_name $m \
	--dataset_dir $TABZILLAPATH/datasets/$dataset \
	--experiment_config $TABZILLAPATH/3000samples_experiment_config.yml \
	--output_dir $output_dir
done

