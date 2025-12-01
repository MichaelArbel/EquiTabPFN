from pathlib import Path
import pandas as pd
import json
import os
from huggingface_hub import snapshot_download
import matplotlib.pyplot as plt

def read_metadatasets(file_name: str):

    project_root = os.path.join(Path(__file__).parent.parent, "data")
    file_dir = os.path.join(project_root, file_name)
    if not os.path.exists(file_dir):
        snapshot_download(
            repo_id="equitabpfn/tabzilla_evaluation",
            repo_type="dataset",
            allow_patterns=f"{file_name}*",
            local_dir=project_root,
            force_download=False,
        )

    return pd.read_csv(file_dir)

def read_results_pickle(raw_results_path: str, dataset_type: str, metric: str):
    file_name = f"results_{dataset_type}.pkl"

    results_pickled = os.path.join(Path(__file__).parent.parent, "data", file_name )

    if os.path.isfile(results_pickled):

        df = pd.read_pickle(results_pickled)

    else:
        if raw_results_path=="":
            # Download pre-computed results from huggingface
            df = read_metadatasets(file_name)

        else:

            tabpfn_models, datasets, selected_baselines, metric_dict = results_setup(dataset_type, metric)
            
            df = read_results_raw(raw_results_path,
                                  tabpfn_models, 
                                  datasets, 
                                  selected_baselines, 
                                  metric_dict
                                  )

            df.to_pickle(results_pickled)


    rename_map = {
            "EquiTabPFNV1": "EquiTabPFN",
            "EquiTabPFNV2": "EquiTabPFN$^*$",
            "TabPFNV1pretrained": "TabPFNv1",
            "TabPFNV2pretrained":"TabPFNv2",
            "TabPFNV2": "TabPFNv2$^*$",
            "EquiTabPFN (A)": "EquiTabPFN",
            "EquiTabPFN (B*)": "EquiTabPFN$^*$",
            "TabPFN (A)": "TabPFNv1",
            "TabPFN (B)":"TabPFNv2",
            "TabPFN (B*)": "TabPFNv2$^*$",
        }


    df['model'] = df['model'].replace(rename_map)

    return df



def results_setup(dataset_type, metric):

    datasets = prepare_datasets(dataset_type)
    tabpfn_models = prepare_models(dataset_type)
    
        # union our results and the ones available
    selected_baselines = ['DecisionTree', 'KNN',
           'LinearModel', 'MLP', 'RandomForest', 'STG', 'TabNet', 'VIME',
           'XGBoost', 'rtdl_MLP', 'rtdl_ResNet']

    metric_dict = {"test": f"{metric}__test", "val":  f"{metric}__val"}


    return tabpfn_models, datasets, selected_baselines, metric_dict




def read_results_raw(results_path: str, 
                    tabpfn_models: list, 
                    datasets: list, 
                    selected_baselines: list, 
                    metric_dict: dict
                    ):
    
    assert results_path.exists()

    metadataset_df = read_metadatasets("metadataset_clean.csv.zip")
    metafeatures_df = read_metadatasets("metafeatures_clean.csv.zip")
    metadataset_df = metadataset_df[metadataset_df['alg_name'].isin(selected_baselines)]

    #results_path = Path(args.results_path)

    dfs = [
        load_path(
            # paths=[Path("/Users/salinasd/slurmpilot/jobs/tabzilla/multiclassif-fix-class-2025-03-30-22-25-49/results/")],
            paths=[results_path],
            methods=tabpfn_models,
        ),
    ]


    df_results = pd.concat(dfs, ignore_index=True)
    print(df_results.alg_name.value_counts())

    df_union = pd.concat([df_results, metadataset_df], ignore_index=True)

    # pick datasets whose scores are available for the baseline
    sel_datasets = (
        df_union.pivot_table(
            index="dataset_name",
            columns="alg_name",
            values=metric_dict["test"],
            aggfunc="count",
        )[[tabpfn_models[0]]]
        .dropna(axis=0)
        .index
    )

    datasets  = list(set(datasets) & set(sel_datasets.tolist()))

    df_union = df_union[df_union['dataset_name'].isin(datasets)]

    allowed_methods = df_results["alg_name"].unique()
    discarded_datasets, sel_algos, df_pairs = find_missing_values(df_union, allowed_methods, n_missing_max=0)
    sel_algos = list(set(sel_algos).union(set(allowed_methods)))

    datasets = list(set(datasets) - set(discarded_datasets) )
    new_df = df_union[df_union['dataset_name'].isin(datasets)]


    new_df = new_df.rename(columns={
        'dataset_name': 'dataset',
        'alg_name': 'model'
    })

    new_df["model_full"] = new_df["model"]


    new_df["time_used"] = new_df["eval-time__test"] + new_df["training_time"]

    extracted = new_df['model'].str.extract(r'^(?P<model>[^_]+)_(?P<ensembles>\d+)_(?P<multi_class_redundancy>\d+)$')

    new_df.loc[extracted['model'].notna(), 'model'] = extracted['model']
    new_df.loc[extracted['model'].notna(), 'ensembles'] =  pd.to_numeric(extracted['ensembles'], errors='coerce')
    new_df.loc[extracted['model'].notna(), 'multi_class_redundancy'] =  pd.to_numeric(extracted['multi_class_redundancy'], errors='coerce')



    # Create masks
    both_nan = new_df['ensembles'].isna() | new_df['multi_class_redundancy'].isna()
    zero_redundancy = new_df['multi_class_redundancy'] == 0

    # Compute with nested np.where
    new_df['total_ens'] = np.where(
        both_nan,
        np.nan,   # if both are NaN → keep NaN
        np.where(
            zero_redundancy,
            new_df['ensembles'],  # if redundancy==0 → ensembles
            new_df['ensembles'] * new_df['multi_class_redundancy'] * 6  # else → ensembles * redundancy * 6
        )
    )

    df = new_df
    
    return df



def save_pickle(df, project_root: Path, dataset_type: str):
    p = project_root / f"data/results_{dataset_type}.pkl"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(p)
    return p



def find_missing_values(df_union, allowed_methods, n_missing_max=0):

        df_pivot = df_union.pivot_table(
            index="dataset_name", columns="alg_name", values=metric_dict["test"]
        )
        print(df_pivot.isna().sum(axis=0).sort_values())
        sel_algos = df_pivot.isna().sum(axis=0).sort_values()
        sel_algos = sel_algos[sel_algos <= n_missing_max].index.tolist()


        missing_results = df_pivot.isna().stack()

        # keep only the True values
        true_s = missing_results[missing_results]

        # extract the MultiIndex tuples
        df_pairs = true_s.reset_index() 
        
        #df_pairs = df_pairs[df_pairs['alg_name'].isin(allowed_methods)]
        discarded_datasets = df_pairs["dataset_name"].unique().tolist()
        
        return discarded_datasets, sel_algos, df_pairs


def load_path(paths, methods):
    metrics = ["F1", "Accuracy", "AUC"]
    rows = []
    for root in paths:
        for file in root.rglob("*json"):
            with open(file, "r") as f:
                res_dict = json.load(f)
            for i in range(10):
                row = {
                    "dataset_fold_id": res_dict["dataset"]["name"] + f"__fold_{i}",
                    "dataset_name": res_dict["dataset"]["name"],
                    "alg_name": res_dict["model"]["name"],
                }
                if row["alg_name"] in methods and "test" in res_dict["scorers"]:
                    for metric in metrics:
                        row[f"{metric}__test"] = res_dict["scorers"]["test"][metric][i]
                        row[f"{metric}__val"] = res_dict["scorers"]["val"][metric][i]
                    row["training_time"] = res_dict["timers"]["train"][i]
                    row["eval-time__test"] = res_dict["timers"]["test"][i]
                    rows.append(row)
    return pd.DataFrame(rows)





def prepare_datasets(dataset_type):
    if dataset_type=="more_10c":

        return [  'openml__collins__3567', 
                             'openml__chess__3952', 
                             'openml__audiology__7', 
                             'openml__one-hundred-plants-texture__9956', 
                             'openml__isolet__3481', 
                             'openml__primary-tumor__146032', 
                             'openml__arrhythmia__5', 
                             'openml__texture__125922', 
                             'openml__soybean__41', 
                             'openml__vowel__3022']
    else:

        metafeatures_df = read_metadatasets("metafeatures_clean.csv.zip")

        df_datasets_stats = metafeatures_df.drop_duplicates("dataset_name")[
            [
                "dataset_name",
                "f__pymfe.general.nr_inst",
                "f__pymfe.general.nr_class",
                "f__pymfe.general.nr_attr",
            ]
        ].set_index("dataset_name")

        datasets_less_10classes = df_datasets_stats[
            (df_datasets_stats["f__pymfe.general.nr_class"] <= 10) &
            (df_datasets_stats["f__pymfe.general.nr_inst"] <= 3000) &
            (df_datasets_stats["f__pymfe.general.nr_attr"] <= 100)
        ].index.tolist()
        return datasets_less_10classes


def save(n, ext=".pdf", save_figs=True, dirname="", rect=None, **kwargs):
    if save_figs == True:
        kwargs.setdefault("bbox_inches", "tight")
        kwargs.setdefault("pad_inches", 0.1)
        kwargs.setdefault("transparent", True)
        if rect:
            plt.tight_layout(rect=rect)
        else:
            plt.tight_layout()
        plt.grid()
        plt.savefig(os.path.join(dirname, n + ext), **kwargs)
        plt.show()


def prepare_models(dataset_type):


    if dataset_type=="more_10c":

        methods_more_10classes = []


        for multi_class_redundancy in range(0,7):
            for ensembles in range(1,49):

                total_ens = ensembles if multi_class_redundancy==0 else ensembles*multi_class_redundancy*6 
                if total_ens <=48 and (total_ens in [1,2,4] or total_ens%6==0):
                    methods_more_10classes.append(f"EquiTabPFNV1_{ensembles}_{multi_class_redundancy}" )
                    if multi_class_redundancy>0:
                        methods_more_10classes.append(f"TabPFNV1pretrained_{ensembles}_{multi_class_redundancy}" )

                if total_ens <=36 and (total_ens in [1,2,4] or total_ens%6==0):
                    methods_more_10classes.append(f"EquiTabPFNV2_{ensembles}_{multi_class_redundancy}" )
                    if multi_class_redundancy>0:
                        methods_more_10classes.append(f"TabPFNV2_{ensembles}_{multi_class_redundancy}" )
                        methods_more_10classes.append(f"TabPFNV2pretrained_{ensembles}_{multi_class_redundancy}" )

    else:
        
        methods_less_10classes = []

        multi_class_redundancy = 0
        for ensembles in [1,2,3,4,5,6,12,18,24,30,32]:
            methods_less_10classes.append(f"EquiTabPFNV2_{ensembles}_{multi_class_redundancy}" )
            methods_less_10classes.append(f"EquiTabPFNV1_{ensembles}_{multi_class_redundancy}" )
            methods_less_10classes.append(f"TabPFNV2_{ensembles}_{multi_class_redundancy}" )
            methods_less_10classes.append(f"TabPFNV2pretrained_{ensembles}_{multi_class_redundancy}" )
            methods_less_10classes.append(f"TabPFNV1pretrained_{ensembles}_{multi_class_redundancy}" )

