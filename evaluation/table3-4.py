#!/usr/bin/env python3
"""Boxplot figure script — standalone as possible."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.data_io import read_results_pickle, save
from utils.processing import process_df, pivot_metric
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


df = process_df(df, ref, metric, processing="cheapest")






columns = {}

columns[f"Median relative Acc."] = pivot_metric(df, 'rel_imp', with_sem=False, scale=100, aggfunc='median', pivot="dataset")
columns[f"Mean Acc."] = pivot_metric(df, 'Accuracy__test', with_sem=True, scale=100)
columns[f"Mean AUC"] = pivot_metric(df, 'AUC__test', with_sem=True,scale=100)
columns[f"Mean F1"] = pivot_metric(df, 'F1__test', with_sem=True,scale=100)
columns["Mean time (s)"] = pivot_metric(df, 'cumulated_time_used', with_sem=False)

df_scores_all = pd.DataFrame(columns).sort_values(by="Median relative Acc.", ascending=False)

for c in df_scores_all.columns:
    df_scores_all[c] = df_scores_all[c].map(lambda x: f"{x:.1f}" if not isinstance(x,str) else x)



print(df_scores_all.to_markdown(floatfmt=".1f"))
latex_string = df_scores_all.to_latex(float_format="%.1f")

name = f"TabZilla_metric_{metric}_datasets_{dataset}.tex"

with open(os.path.join(figure_path(),name), "w") as f:
    f.write(latex_string)


