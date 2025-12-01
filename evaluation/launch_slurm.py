import os
import subprocess
from pathlib import Path
import pandas as pd

import argparse

from utils.data_io import prepare_models, prepare_datasets


def split_into_n_parts(lst, n_parts):
    k, m = divmod(len(lst), n_parts)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_parts)]



def submit_eval_jobs(methods, datasets, script_path, output_dir, tag):
    for i, method in enumerate(methods):
        for dataset in datasets:
            # Construct a unique job name
            job_name = f"{tag}/eval_{dataset}_chunk_{i}"
            
            # Build the sbatch command
            sbatch_cmd = [
                "sbatch",
                f"--job-name={job_name}",
                f"--output={job_name}.out",
                f"--error={job_name}.err",
                script_path,
                dataset,
                method,
                output_dir
            ]
            
            # Submit the job
            print(f"Submitting job {job_name}")
            result = subprocess.run(sbatch_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  Submitted: {result.stdout.strip()}")
            else:
                print(f"  Submission error: {result.stderr.strip()}")

if __name__ == "__main__":



    parser = argparse.ArgumentParser(description="parser for tabzilla experiments")

    parser.add_argument(
        "--output_dir",
        default="",
        type=str,
        help="config file for parameter experiment args",
    )
    parser.add_argument(
        "--dataset_type",
        default="more_10c",
        type=str,
        help="two possible values: more_10c or less_10c",
    ) 



    args = parser.parse_args()
    print(f"ARGS: {args}")
    dataset_type = args.dataset_type
    datasets = prepare_datasets(dataset_type)
    methods = prepare_models(dataset_type)



    print(f"Each job evaluates {len(methods)} methods")

    n_parts = 1

    chunked_methods = split_into_n_parts(methods, n_parts)
    methods = [",".join(chunk) for chunk in chunked_methods]

    # Path to your existing bash script
    bash_script = "eval.sh"

    submit_eval_jobs(methods, datasets, bash_script, args.output_dir, dataset_type)
