#!/usr/bin/env python3
"""Boxplot figure script — standalone as possible."""
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

df = df[df['model']!="EquiTabPFN$^*$"]


tm = (
    df.groupby('model')['time_used']
       .mean()
       .reset_index(name='geom_time')
)

df['rel_imp'] = 100 *df['rel_imp']
# tm = (
#     df.groupby('model')['cumulated_time_used']
#        .mean()
#        .reset_index(name='geom_time')
# )
# ─────────── 3. Build a color map keyed by method ───────────
# we'll color by log10(geom_time)
cmap = plt.get_cmap('plasma_r')
vmin = 0.1#tm["geom_time"].min()/10
vmax = 44 #2*tm["geom_time"].max()
norm = LogNorm(vmin=vmin, vmax=vmax)


sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_clim(vmin, vmax)

method_to_color = {
    m: cmap(norm(t))
    for m, t in zip(tm['model'], tm['geom_time'])
}

# ─────────── 4. Plot ───────────

df = (
    df.groupby(['dataset','model'])['rel_imp']
       .mean()
       .reset_index()    
)

order = (
    df.groupby(['model'],)['rel_imp']
       .median()
       .sort_values(ascending=False)
       .index
)





fig_height = 0.2 * len(order) +0.5
fig, ax = plt.subplots(1, 1, figsize=(3.5, fig_height))


# boxplot
sns.boxplot(
    x='rel_imp', y='model',
    data=df, order=order,
    palette=method_to_color,
    fliersize=1,    # hide the default outliers to avoid overplot
    width=0.8,
    medianprops=dict(color='red', linewidth=2),
)

for i in range(0, len(order), 2):
    ax.axhspan(i - 0.5, i + 0.5, color='gray', alpha=0.2, zorder=0)

ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))  # If your values are in [0,1]
 

# vertical line at zero improvement
plt.axvline(0, color='k', linestyle='--', linewidth=1)

fontsize= 8
plt.xlabel(f'Relative Acc. improvement over {ref}  (↑)', fontsize=fontsize)


plt.ylabel('')

color_supervised = '#9467bd' 
color_self_supervised = '#ff7f0e'
color_ours = '#1f77b4'
method_colors_dict = {
'TabPFNv2': color_self_supervised,
'TabPFNv1': color_self_supervised,
'TabPFNv2$^*$': color_self_supervised,
'TabPFNv2+Ens': color_self_supervised,
'TabPFNv1+Ens': color_self_supervised,
'TabPFNv2$^*$+Ens': color_self_supervised,
'VIME': color_supervised, ## self-supervised method
'EquiTabPFN': color_ours,
'MLP': color_supervised,
'rtdl_MLP': color_supervised,
'rtdl_ResNet': color_supervised,
'TabNet': color_supervised,
}


if dataset=="more_10c":

    regular_order = ['TabPFNv2', 'EquiTabPFN', 'TabPFNv1', 'XGBoost', 'RandomForest',
       'TabPFNv2$^*$', 'rtdl_ResNet', 'LinearModel', 'DecisionTree', 'MLP',
       'rtdl_MLP', 'KNN', 'STG', 'TabNet', 'VIME']
    ref_pos = {method: i for i, method in enumerate(regular_order)}

    diffs = []
    for new_idx, method in enumerate(list(order)):
        old_idx = ref_pos[method]
        diff    = old_idx - new_idx
        diffs.append(diff)



ax.set_yticklabels([])
yticks = ax.get_yticks()
for i, (y, method) in enumerate(zip(yticks, list(order))):
# method name
    color='black'
    if method in method_colors_dict:
        color = method_colors_dict[method]
    
    ax.text(
        x=-0.1,    # in axis coordinates, just left of the axis
        y=y, 
        s=method,
        ha='right', va='center',
        transform=ax.get_yaxis_transform(),
        color=color, fontsize=8
    )

    if dataset=="more_10c":
        # annotation
        diff = diffs[i]
        if diff != 0:
            sign = f"+{diff}" if diff>0 else str(diff)
            col  = 'green' if diff>0 else 'red'

            ax.text(
                x=-0.015, # slightly to the right of the method name
                y=y,
                s=f"({sign})",
                ha='right', va='center',
                transform=ax.get_yaxis_transform(),
                color=col, fontsize=6
            )


if dataset=="more_10c":

    symb = ">"
else:
    symb = "≤"

num_data = len(df["dataset"].unique())
plt.title(f"{num_data} Datasets, {symb} 10 Classes", fontsize=8)

# add a colorbar for geom_time


# draw the colorbar
cbar = plt.colorbar(
    sm,
    orientation='horizontal',
    pad=0.15,
    ax=ax
)

cbar.locator = LogLocator(base=10)         # major ticks at 10^n




superscript = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
ticks = cbar.get_ticks()


ticks = [ el for el in ticks if el <=vmax and  vmin <el ]


labels = [("10" + str(int(np.log10(t))).translate(superscript)) for t in ticks]
cbar.set_ticks(ticks)
cbar.set_ticklabels(labels)

cbar.set_label('Mean training + evaluation time [s]', labelpad=6, fontsize=fontsize)
cbar.ax.xaxis.set_label_position('bottom')


name = f"Tabzilla_box_plot_{metric}_datasets_{dataset}"
save(os.path.join(figure_path(),name))




