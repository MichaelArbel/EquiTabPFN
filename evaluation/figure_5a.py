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
import seaborn as sns
import matplotlib.colors as mcolors

from matplotlib.ticker import PercentFormatter



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
dfb = process_df(df, ref, metric, processing="best")

  # rename for clarity in plotting

# 1) Compute the Pareto front:


dfb = dfb[~(dfb['model'] == "EquiTabPFN$^*$")]




df_sorted = dfb.sort_values('cumulated_time_used')


pareto_mask = []
max_y = -float('inf')
for _, row in df_sorted.iterrows():
    if row['avg_metric'] > max_y:
        pareto_mask.append(True)
        max_y = row['avg_metric']
    else:
        pareto_mask.append(False)

pareto_df = df_sorted[pareto_mask]



base_colors = sns.color_palette('tab20', n_colors=20)


def closest_color_name(requested_rgb):
    min_colors = {}
    for name, hex in mcolors.CSS4_COLORS.items():
        r_c, g_c, b_c = mcolors.to_rgb(hex)
        rd = (r_c - requested_rgb[0]) ** 2
        gd = (g_c - requested_rgb[1]) ** 2
        bd = (b_c - requested_rgb[2]) ** 2
        min_colors[(rd + gd + bd)] = name

    return min_colors[min(min_colors.keys())]

def color_names_dict(color_list):
    used_names = []
    color_names = {}
    named_color_list = []
    for color in color_list:
        name = closest_color_name(color)
        suff = ""
        if name in used_names:
            suff = "_1"
        real_name = name+suff
        color_names[real_name] = color
        named_color_list.append(real_name)
        used_names.append(real_name)
    return color_names, named_color_list

named_colors, color_names = color_names_dict(base_colors)

color_names=['steelblue', 'grey', 'darkorange', 'lightsalmon', 'forestgreen', 'lightgreen', 'crimson', 'lightsalmon_1', 'mediumpurple', 'thistle', 'sienna', 'rosybrown', 'orchid', 'pink',  'silver', 'goldenrod', 'khaki', 'darkturquoise', 'lightblue']


#custom_palette = base_colors


model_names = dfb['model'].unique()


custom_palette = [named_colors[name] for name in color_names]

palette_dict = dict(zip(model_names, custom_palette[:len(model_names)]))




#print(lol)


plt.figure(figsize=(5,5)) 
sns.scatterplot(
    data=dfb, 
    x='cumulated_time_used',
    y='avg_metric',
    hue='model',
    size='ensembles',
    sizes=(50, 400),         # min/max bubble size
    alpha=0.8,
    palette=palette_dict,    
    legend='full'
)


# 3) Overlay Pareto front
plt.plot(
    pareto_df['cumulated_time_used'],
    pareto_df['avg_metric'],
    color='black',
    linestyle='--',
    linewidth=2,
    label='Pareto front'
)
 


plt.xscale('log')
plt.xlabel('Time cost (s)')
plt.ylabel(f'% {metric}. improvement over {ref}')





ax = plt.gca()
fig = plt.gcf()
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))


handles, labels = ax.get_legend_handles_labels()

num_methods = 10



method_handles, method_labels = handles[:num_methods+1]+[handles[-1]], labels[:num_methods+1] + [labels[-1]]
ensemble_handles, ensemble_labels = handles[num_methods+1:-1], labels[num_methods+1:-1] 


legend1 = ax.legend(
    method_handles,method_labels,
    loc='upper left',
    bbox_to_anchor=(1.02, 1),
    ncol=1,   # adjust as needed
    frameon=False,
    labelspacing=0.5,
)
ensemble_labels = [ensemble_labels[0], ensemble_labels[1], ensemble_labels[4]]+ ensemble_labels[5::2]
ensemble_handles = [ensemble_handles[0], ensemble_handles[1], ensemble_handles[4]]+ ensemble_handles[5::2]
# Second row: ensemble size legend (bubbles)
legend2 = ax.legend(
    ensemble_handles, ensemble_labels,
    loc='upper left',
    bbox_to_anchor=(1.02, 0.4),
    ncol=1,
    frameon=False,
    labelspacing=0.5,
)

# Add both legends
#ax.add_artist(legend1)
#ax.add_artist(legend2)

fig.legends.append(legend1)
fig.legends.append(legend2)




name = f"Tabzilla_pareto_{metric}_datasets_{args.dataset}"

save(os.path.join(figure_path(),name), rect=[0,0,0.85,1])













