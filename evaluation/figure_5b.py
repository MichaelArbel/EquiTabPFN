#!/usr/bin/env python3
"""Efficiency figure — compute speedup and accuracy ratio per ensemble (best-effort)."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.data_io import read_results_pickle, save
from utils.processing import process_df
import os
from equitabpfn.utils import figure_path


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='more_10c', type=str)
parser.add_argument("--ref", default="KNN", type=str)
parser.add_argument("--metric", default="AUC", type=str)
parser.add_argument("--raw_data_path", default="", type=str)



args = parser.parse_args()
project_root = Path(__file__).resolve().parents[1]
metric = args.metric
ref = args.ref


df = read_results_pickle(args.raw_data_path, args.dataset, args.metric)
df = process_df(df, ref, metric, processing="best")



df_plot = df.copy()
# select TabPFN (B) and EquiTabPFN (A)
dfm_tabpfn = df_plot[df_plot['model'] == "TabPFNv2"].set_index('total_ens')
dfm_equitabpfn = df_plot[df_plot['model'] == "EquiTabPFN"].set_index('total_ens')


times = (dfm_tabpfn["cumulated_time_used"])/dfm_equitabpfn.iloc[0]['cumulated_time_used']
accs = dfm_tabpfn['avg_metric']/dfm_equitabpfn.iloc[0]['avg_metric']

ensembles = sorted(dfm_tabpfn.index)
times = times.values
accs  = accs.values



x = np.arange(len(ensembles))
width = 0.35

fig, ax1 = plt.subplots(figsize=(4,4))
ax2 = ax1.twinx()

# Time on ax1
bars1 = ax1.bar(x - width/2, times, width, color='#2ca02c', alpha=0.8, label='Speedup (TabPFNv2/EquiTabPFN)')
ax1.set_ylabel('Speedup (TabPFNv2 / EquitabPFN)', color='#2ca02c', fontsize=8)
ax1.tick_params(axis='y', labelcolor='#2ca02c')

# Accuracy on ax2
bars2 = ax2.bar(x + width/2, accs, width, color='#1f77b4', alpha=0.8, label=f'TabPFNv2 (Rel. {metric})')
ax2.set_ylabel(f' Rel. {metric} ratio (TabPFNv2 / EquitabPFN)', color='#1f77b4',fontsize=8)
ax2.tick_params(axis='y', labelcolor='#1f77b4')
x0, x1 = ax2.get_xlim()

# 3) plot at x1
ax2.scatter([x1], [dfm_equitabpfn.iloc[0]['avg_metric']], marker='*', color='#1f77b4', label=f'EquiTabPFN (Rel. {metric})', zorder=10)

# 4) (optional) reset the x‐limits if you don’t want autoscaling to push them out
ax2.set_xlim(x0, x1)

# Common X formatting
ax1.set_xticks(x)
ensembles = [int(ens) for ens in ensembles]
ax1.set_xticklabels(ensembles)
ax1.set_xlabel('Number of Ensembles used for TabPFNv2')

name = f"Tabzilla_efficiency_{metric}_datasets_{args.dataset}"
save(os.path.join(figure_path(),name))





