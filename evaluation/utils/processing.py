import pandas as pd
import numpy as np
from autorank import autorank


# import mlxp
import os






TABPFN_METHODS = ["EquiTabPFN", "TabPFNv1", "TabPFNv2", "TabPFNv2$^*$", "EquiTabPFN$^*$"]



# def relative_metric(df: pd.DataFrame, group_cols, ref: str, metric: str, per_split=True):
#     """Compute a relative improvement metric per dataset fold or per dataset.
#     metric argument should be a column name like 'AUC__test' or 'Accuracy__test'"""
#     df = df.copy()
#     if per_split:
#         pivot_key = 'dataset_fold_id' if 'dataset_fold_id' in df.columns else 'dataset'
#     else:
#         pivot_key = 'dataset'
#     # compute reference value per pivot_key
#     ref_vals = df[df['model'] == ref].groupby(pivot_key)[metric].mean().rename('ref_val')
#     df = df.merge(ref_vals, left_on=pivot_key, right_index=True, how='left')
#     # avoid division by zero: if ref_val == 0, set to small number
#     df['ref_val'] = df['ref_val'].replace(0, np.nan)
#     df[f'rel_imp_{metric}'] = 100.0 * (df[metric] - df['ref_val']) / df['ref_val']
#     # also create short name rel_imp for convenience
#     df['rel_imp'] = df[f'rel_imp_{metric}']
#     return df.drop(columns=['ref_val'])

# def avg_results(df: pd.DataFrame, metric_id: str, groupby_cols):
#     """Compute average and SEM across datasets (or dataset_fold_id) for the given grouping."""
#     grp = df.groupby(groupby_cols)[metric_id].agg(['mean', 'sem']).reset_index()
#     grp = grp.rename(columns={'mean':'avg_metric','sem':'avg_sem'})
#     return grp



def process_df(df, ref, metric, processing="default"):

    tabpfn_methods = TABPFN_METHODS
    df_baselines = df[~df["model"].isin(tabpfn_methods)]
    df_tabpfn = df[df["model"].isin(tabpfn_methods)]



    group_cols = ['model', 'total_ens', 'multi_class_redundancy']

    ref_df = df[df['model'] == ref] 

    df_processed_tabpfn, avg_metric_tabpfn = compute_average_metric(df_tabpfn, group_cols, ref_df, metric)
    df_processed_baselines, _ = compute_average_metric(df_baselines, ["model"], ref_df, metric)


    if processing=="best":

        avg_metric_tabpfn = avg_metric_tabpfn[~(avg_metric_tabpfn['multi_class_redundancy'].isin([4,5,6]))]


        idx = avg_metric_tabpfn.groupby(['model', 'total_ens'])[f'avg_metric_val'].idxmax()
        df_best_variant = avg_metric_tabpfn.loc[idx].reset_index(drop=True)
        df_best_variant['ensembles'] = df_best_variant['total_ens'].astype(int) 

        selected_baselines = df_processed_baselines[df_processed_baselines["avg_metric"]>0.]
        selected_baselines["ensembles"] = 1


        return pd.concat([df_best_variant, selected_baselines], axis=0, sort=False, ignore_index=True)


    if processing=='cheapest':


        idx = df_processed_tabpfn.groupby(['dataset_fold_id','dataset','model'])['total_ens'].idxmin()
        df_cheapest = df_processed_tabpfn.loc[idx].reset_index(drop=True)
        return  pd.concat([df_cheapest, df_processed_baselines], ignore_index=True)

    return pd.concat([df_processed_tabpfn, df_processed_baselines], ignore_index=True)



def compute_average_metric(df, group_cols, ref_df, metric):

    metric_dict = {"test": f"{metric}__test", "val":  f"{metric}__val"}

    cum_time = (
        df
        .groupby(group_cols)['time_used']
        .mean()
        .reset_index(name='cumulated_time_used')
    )
    df = relative_metric(df, group_cols, ref_df, metric=metric_dict["test"],per_split=True)
    df = relative_metric(df, group_cols, ref_df, metric=metric_dict["val"],per_split=True)

    #print(lol)
    df["rel_imp"] = df[f"rel_imp_{metric_dict["test"]}" ]

    metric_id=  f"rel_imp_{metric_dict["val"]}" 
    avg_metric_tmp = avg_results(df, metric_id, group_cols)
    
    metric_id=  f"rel_imp_{metric_dict["test"]}" 
    avg_metric = avg_results(df, metric_id, group_cols)

    avg_metric['avg_metric'] = avg_metric[f'avg_metric_rel_imp_{metric_dict["test"]}']

    avg_metric['avg_sem'] = avg_metric[f'avg_sem_rel_imp_{metric_dict["test"]}']

    avg_metric['avg_metric_val'] = avg_metric_tmp[f'avg_metric_rel_imp_{metric_dict["val"]}']

    avg_metric['avg_sem_val'] = avg_metric_tmp[f'avg_sem_rel_imp_{metric_dict["val"]}']




    to_add = cum_time.columns.difference(avg_metric.columns)

    avg_metric = pd.concat([avg_metric, cum_time[to_add]], axis=1)


    df_processed = df.merge(
        avg_metric,
        on=group_cols,
        how='left'
    )
    return df_processed, avg_metric

def pivot_metric(df, metric, with_sem=False ,scale=1, aggfunc='mean', pivot="dataset_fold_id"):
    df_pivot = df.pivot_table(
            index=pivot, columns="model", values=metric
        )
    if with_sem:
        if pivot =="dataset_fold_id":
            return pd.Series(
                [f"{scale*mean:.1f} +/- {scale*sem:.1f}" for mean, sem in zip(df_pivot.mean(), df_pivot.sem())], index=df_pivot.columns
            )
        else: 
            df_pivot_sem = df.pivot_table(
                index=pivot,
                columns="model",
                values=metric,            # the column you want the SEM of
                aggfunc=lambda x: x.sem() # Pandas Series.sem()
            )
            return pd.Series(
                [f"{scale*mean:.2f} +/- {scale*sem:.2f}" for mean, sem in zip(df_pivot.mean(), df_pivot_sem.mean())], index=df_pivot.columns
            )
    else:
        if aggfunc=='mean':
            return pd.Series(
                [scale*mean for mean in df_pivot.mean()], index=df_pivot.columns
            )
        else:
            return pd.Series(
                [scale*mean  for mean in df_pivot.median()], index=df_pivot.columns
            )








def avg_results(df,metric, group_cols, var_mode="per_split"):
    if var_mode=="per_split":
        avg_metric = (
            df
            .groupby(["dataset"] + group_cols,as_index=False)[metric]
            .agg(
                avg_metric='mean',
                avg_sem ='sem'
            )
        )
        avg_metric[f'avg_metric_{metric}']= avg_metric['avg_metric']
        avg_metric[f'avg_sem_{metric}']= avg_metric['avg_sem']

        summary = (
            avg_metric
            .groupby(group_cols)[[f'avg_metric_{metric}',f'avg_sem_{metric}']]
            .mean()
            .reset_index()                  # takes the mean of each column
        )
    else:
        summary = (
            df
            .groupby(group_cols)[metric]
            .mean()
            .reset_index()                  # takes the mean of each column
        )
    return summary

def normalize_score(df, group_cols, metric="mean_metric"):
    # 1. Per (dataset, model): mean & std over splits
    agg = (
        df
        .groupby(['dataset'] + group_cols, as_index=False)
        [metric]
        .agg(
            split_avg_metric='mean',
            split_sem='sem',           # std of the *raw* split scores
        )
        .reset_index()
    )

    # 2. Normalize the per‐dataset means
    min_vals   = agg.groupby('dataset')['split_avg_metric'].transform('min')
    max_vals   = agg.groupby('dataset')['split_avg_metric'].transform('max')
    range_vals = max_vals - min_vals

    agg['norm_avg'] = (agg['split_avg_metric'] - min_vals) / range_vals
    agg['norm_sem_split'] = agg['split_sem'] / range_vals
    agg[['norm_avg','norm_sem_split']] = agg[['norm_avg','norm_sem_split']].fillna(0.0)


    # 3. Per‐model across datasets: mean and std of the *normalized* averages
    
    global_norm = (
        agg
        .groupby(group_cols, as_index=False)['norm_avg']
        .agg(
            avg_metric='mean',
            avg_sem ='sem'
        )
    )
    global_norm['avg_sem'] = global_norm['avg_sem'].fillna(0.0)
    return global_norm


def find_best_model(df, metric, best_over_cols, group_cols=[], dataset="dataset"):
    df_avgs = (
        df
        .groupby([dataset, 'model'] +group_cols + best_over_cols, as_index=False)[metric]
        .mean()
    )
    if best_over_cols:
        df_best = df_avgs.loc[df_avgs.groupby([dataset,'model']+group_cols)[metric].idxmax()]
    else:
        df_best = df_avgs
    return df_best


def relative_metric(df, group_cols, ref_df, metric="mean_metric",best_over_cols = [], per_split=False):
    # 1) Compute the per‐dataset average over splits for the reference model
    if per_split:
        dataset = 'dataset_fold_id'
    else:
        dataset ='dataset'

    #df_best = df[df['model'] == ref]
    ref_map = find_best_model(ref_df,metric, best_over_cols=best_over_cols, dataset=dataset)



    df_ref_lookup = ref_map[[dataset, metric]].rename(
        columns={metric:f'ref_metric_{metric}'}
    )

    df = df.merge(
        df_ref_lookup,
        on=dataset,
        how='left'
    )
    #print(lol)
    df[f'rel_imp_{metric}'] = (df[metric] - df[f'ref_metric_{metric}']) / df[f'ref_metric_{metric}']

    return df
