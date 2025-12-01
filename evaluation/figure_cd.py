#!/usr/bin/env python3
"""Critical diagram (autorank) — standalone best-effort."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.data_io import read_results_pickle, save
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
parser.add_argument('--dataset', default='less_10c', type=str)
parser.add_argument("--ref", default="KNN", type=str)
parser.add_argument("--metric", default="Accuracy", type=str)
parser.add_argument("--raw_data_path", default="", type=str)



args = parser.parse_args()
project_root = Path(__file__).resolve().parents[1]
metric = args.metric
ref = args.ref
dataset = args.dataset

df = read_results_pickle(args.raw_data_path, args.dataset, args.metric)
df = process_df(df, ref, metric)


tabpfn_methods = ["EquiTabPFN", "EquiTabPFN$^*$", "TabPFNv1", "TabPFNv2", "TabPFNv2$^*$"]
sub_df = df[df["model"].isin(tabpfn_methods)]
baselines_df = df[~df["model"].isin(tabpfn_methods)]
idx = sub_df.groupby(['dataset_fold_id','dataset','model'])['total_ens'].idxmax()
df_best = sub_df.loc[idx].reset_index(drop=True)
rename_map = {"TabPFNv1": "TabPFNv1+Ens",
           "TabPFNv2": "TabPFNv2+Ens",
           "TabPFNv2$^*$": "TabPFNv2$^*$+Ens"
          }
df_best['model'] = df_best['model'].replace(rename_map)
idx = sub_df.groupby(['dataset_fold_id','dataset','model'])['total_ens'].idxmin()
df_cheapest_equi = sub_df.loc[idx].reset_index(drop=True)
df_cheapest = pd.concat([df_best, df_cheapest_equi], ignore_index=True)

df_cheapest = pd.concat([df_cheapest, baselines_df], ignore_index=True)


wide = df_cheapest.pivot_table(
        index="dataset_fold_id", columns="model", values="AUC__test"
    )


result = autorank(wide.rank(axis=1, ascending=True), force_mode='nonparametric', alpha=0.05, verbose=False)

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
plot_stats(result, ax=ax)
name = f"Tabzilla_CD_metric_{metric}_datasets_{args.dataset}"
save(os.path.join(figure_path(),name))

