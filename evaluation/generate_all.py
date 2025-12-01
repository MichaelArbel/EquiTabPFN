#!/usr/bin/env python3
import itertools
import subprocess

import os
os.environ['MPLBACKEND'] = 'Agg'

scripts = { 'figure_boxplot.py': {'dataset': ['more_10c', 'less_10c']},
			'figure_cd.py': {'dataset': ['more_10c', 'less_10c']},
			'figure_5a.py' : {},
			'figure_5b.py' : {},
			'table1-2.py' : {'dataset': ['more_10c', 'less_10c']},
			'table3-4.py': {'dataset': ['more_10c', 'less_10c']}
			}


for script, arg_dict in scripts.items():
    if not arg_dict:  # no arguments
        cmd = ["python", script]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        continue

    # create all combinations of argument values
    keys = list(arg_dict.keys())
    values_product = itertools.product(*(arg_dict[k] for k in keys))

    for combo in values_product:
        cmd = ["python", script]
        for key, val in zip(keys, combo):
            cmd += [f"--{key}", str(val)]

        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)