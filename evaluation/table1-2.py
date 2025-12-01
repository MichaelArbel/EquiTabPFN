#!/usr/bin/env python3
"""Boxplot figure script — standalone as possible."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.data_io import read_results_pickle, save, read_metadatasets
from utils.processing import process_df
import os
from equitabpfn.utils import figure_path
import seaborn as sns
import matplotlib.colors as mcolors

from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator
import matplotlib.ticker as mtick
from autorank import plot_stats, autorank

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='more_10c', type=str)
parser.add_argument("--ref", default="KNN", type=str)
parser.add_argument("--metric", default="Accuracy", type=str)
parser.add_argument("--raw_data_path", default="", type=str)



args = parser.parse_args()
project_root = Path(__file__).resolve().parents[1]
metric = args.metric
ref = args.ref
dataset = args.dataset

df = read_results_pickle(args.raw_data_path, args.dataset, args.metric)


if df is None:
    raise FileNotFoundError('results pickle not found. Please create data/results_<dataset>.pkl first.')

selected_datasets = df["dataset"].unique()
metafeatures_df = read_metadatasets("metafeatures_clean.csv.zip")
df_datasets_stats = metafeatures_df.drop_duplicates("dataset_name")[
    [
        "dataset_name",
        "f__pymfe.general.nr_inst",
        "f__pymfe.general.nr_class",
        "f__pymfe.general.nr_attr",
    ]
]
sel = df_datasets_stats[df_datasets_stats["dataset_name"].isin(selected_datasets)]
_rename_map = {"f__pymfe.general.nr_inst": "Samples",
              "f__pymfe.general.nr_class": "Classes",
              "f__pymfe.general.nr_attr": "Features",
              }
sel.rename(columns=_rename_map, inplace=True)
sel[['name', 'taskId']] = sel['dataset_name'].str.extract(r'openml__(.*?)__(\d+)')
sel = sel[['taskId', 'name', 'Classes', 'Features', 'Samples']]
sel['taskId'] = sel['taskId'].astype(int)
sel = sel.sort_values(by='taskId')


latex_string = sel.to_latex(index= False, float_format="%.1f")

name = f"TabZilla_datasets_{dataset}.tex"

with open(os.path.join(figure_path(),name), "w") as f:
    f.write(latex_string)


