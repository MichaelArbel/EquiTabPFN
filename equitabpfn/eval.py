from typing import List
import os
from pathlib import Path

import numpy as np
import openml
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from tqdm import tqdm


from autogluon.features import AutoMLPipelineFeatureGenerator
from sklearn.impute import SimpleImputer
import torch

from equitabpfn.evaluation.equivariance_error import average_equivariance_error
from equitabpfn.model_builder import load_model, load_model_from_name
from equitabpfn.utils import get_original_state_dict, batched_pca
from mothernet.prediction.tabpfn import TabPFNClassifier


from mothernet.evaluation.tabular_evaluation import _eval_single_dataset_wrapper
from mothernet.evaluation.baselines.tabular_baselines import transformer_metric
from tqdm import tqdm
from sklearn.base import BaseEstimator
from joblib import Parallel, delayed
from functools import partial
import itertools
import pickle
#openml.config.server = "http://145.38.195.79/api/v1/xml"
import torch
from torch import autograd
default_root = os.path.join(Path(__file__).parent.parent, "data")

open_cc_ood_taskids =[168300, 146802, 3481, 9972, 2076, 6, 359953, 41, 125922]
open_cc_ood_dids =   [41084,  40971,  300,  1481, 184, 6, 1515, 42, 40499]


#open_cc_ood_dids = [1515]
from equitabpfn.model_builder import get_model_info, get_model, load_model_from_states

from equitabpfn.models.tabpfnv2 import TabPFNv2
from equitabpfn.models.equitabpfnv2 import EquiTabPFNv2


import yaml



def eval_on_datasets(
    task_type,
    model,
    model_name,
    datasets,
    eval_positions,
    max_times, 
    metric_used,
    split_numbers,
    n_samples,
    base_path,
    overwrite=False,
    append_metric=True,
    fetch_only=False,
    verbose=False,
    n_jobs=-1,
    save=False,
    device="auto",
    force=False
):
    if callable(model):
        model_callable = model
        if device == "auto":
            device = "cpu"
    elif isinstance(model, BaseEstimator):
        model_callable = partial(transformer_metric, classifier=model)
        device_param = [v for k, v in model.get_params().items() if "device" in k]
        device = device_param[0] if len(device_param) > 0 else "cpu"
    else:
        raise ValueError(
            f"Got model {model} of type {type(model)} which is not callable or a BaseEstimator"
        )
    if model_name in ["knn", "rf_new_params", "xgb", "logistic"]:
        device="cpu"
    if model_name=="xgb":
        n_jobs = 64

    results_dir = os.path.join(default_root, "results_per_model")
    local_path = os.path.join(results_dir, model_name)
    if not Path(local_path).exists() or force:

        os.makedirs(results_dir, exist_ok=True)
        print(f"evaluating {model_name} on {device}")
        if "cuda" in device:
            results = []
            for ds, max_time, split_number in tqdm(
                list(itertools.product(datasets, max_times, split_numbers))
            ):
                result = _eval_single_dataset_wrapper(
                    datasets=[ds],
                    model=model_callable,
                    model_name=model_name,
                    n_samples=n_samples,
                    base_path=base_path,
                    eval_positions=eval_positions,
                    device=device,
                    max_splits=1,
                    overwrite=overwrite,
                    save=save,
                    metric_used=metric_used,
                    path_interfix=task_type,
                    fetch_only=fetch_only,
                    split_number=split_number,
                    verbose=verbose,
                    max_time=max_time,
                    append_metric=append_metric,
                )
                results.append(result)
        else: 
            results = Parallel(n_jobs=n_jobs, verbose=2)(
               delayed(_eval_single_dataset_wrapper)(
            #results  = [_eval_single_dataset_wrapper(
                    datasets=[ds],
                    model=model_callable,
                    model_name=model_name,
                    n_samples=n_samples,
                    base_path=base_path,
                    eval_positions=eval_positions,
                    device=device,
                    max_splits=1,
                    overwrite=overwrite,
                    save=save,
                    metric_used=metric_used, 
                    path_interfix=task_type,
                    fetch_only=fetch_only,
                    split_number=split_number,
                    verbose=verbose,
                    max_time=max_time,
                    append_metric=append_metric,
                )
                for ds in datasets
                for max_time in max_times
                for split_number in split_numbers
            )
            with open(local_path, "wb") as f:
                pickle.dump(results, f)
    else:
        print(f"Loading evaluation for {model_name}")
        with open(local_path, "rb") as f:
            results = pickle.load(f)

    return results


def get_openml_classification(
    did, max_samples, root=default_root, multiclass=True, shuffled=True
):
    X, y, categorical_indicator, attribute_names = get_dataset(
        did, dataset_format="array", root=root
    )
    # dataset = openml.datasets.get_dataset(did, download_all_files=True)
    # X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute, dataset_format="array")
    X = np.array(X)
    y = np.array(y)
    if not multiclass:
        X = X[y < 2]
        y = y[y < 2]

    if multiclass and not shuffled:
        raise NotImplementedError(
            "This combination of multiclass and shuffling isn't implemented"
        )

    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        print("Not a NP Array, skipping")
        return None, None, None, None

    if not shuffled:
        sort = np.argsort(y) if y.mean() < 0.5 else np.argsort(-y)
        pos = int(y.sum()) if y.mean() < 0.5 else int((1 - y).sum())
        X, y = X[sort][-pos * 2 :], y[sort][-pos * 2 :]
        y = torch.tensor(y).reshape(2, -1).transpose(0, 1).reshape(-1).flip([0]).float()
        X = (
            torch.tensor(X)
            .reshape(2, -1, X.shape[1])
            .transpose(0, 1)
            .reshape(-1, X.shape[1])
            .flip([0])
            .float()
        )
    else:
        order = np.arange(y.shape[0])
        np.random.seed(13)
        np.random.shuffle(order)
        X, y = torch.tensor(X[order]), torch.tensor(y[order])
    if max_samples:
        X, y = X[:max_samples], y[:max_samples]

    return X, y, list(np.where(categorical_indicator)[0]), attribute_names


def load_openml_list(
    dids,
    filter_for_nan=False,
    num_feats=100,
    min_samples=100,
    max_samples=400,
    multiclass=True,
    max_num_classes=10,
    shuffled=True,
    return_capped=False,
    verbose=0,
    root=default_root,
    datasets_name="datalist.pkl"
):
    data_path = root
    local_path = os.path.join(data_path, datasets_name)

    datasets = []
    if not Path(local_path).exists():
        os.makedirs(data_path, exist_ok=True)

        openml_list = openml.datasets.list_datasets(dids)
        import pickle

        with open(local_path, "wb") as f:
            pickle.dump(openml_list, f)
    else:
        import pickle

        with open(local_path, "rb") as f:
            openml_list = pickle.load(f)
            openml_list = {key:value for key,value in openml_list.items() if key in dids}

    print(f"Number of datasets: {len(openml_list)}")
    datalist = pd.DataFrame.from_dict(openml_list, orient="index")
    if filter_for_nan:
        datalist = datalist[datalist["NumberOfInstancesWithMissingValues"] == 0]
        print(
            f"Number of datasets after Nan and feature number filtering: {len(datalist)}"
        )

    for ds in datalist.index:
        modifications = {
            "samples_capped": False,
            "classes_capped": False,
            "feats_capped": False,
        }
        entry = datalist.loc[ds]
        if verbose > 0:
            print("Loading", entry["name"], entry.did, "..")

        if entry["NumberOfClasses"] == 0.0:
            raise Exception("Regression not supported")
            # X, y, categorical_feats, attribute_names = get_openml_regression(int(entry.did), max_samples)
        else:
            X, y, categorical_feats, attribute_names = get_openml_classification(
                int(entry.did),
                max_samples,
                root=root,
                multiclass=multiclass,
                shuffled=shuffled,
            )
        if X is None:
            continue

        if X.shape[1] > num_feats:
            if return_capped:
                X = X[:, 0:num_feats]
                # X = X.unsqueeze(1)
                # X = batched_pca(X.cuda()).cpu()
                # X = X.squeeze(1)
                categorical_feats = [c for c in categorical_feats if c < num_feats]
                modifications["feats_capped"] = True
            #else:
            #    print("Too many features")
            #    continue
        if X.shape[0] == max_samples:
            modifications["samples_capped"] = True

        if X.shape[0] < min_samples:
            print("Too few samples left")
            continue

        if len(np.unique(y)) > max_num_classes:
            if return_capped:
                X = X[y < np.unique(y)[max_num_classes]]
                y = y[y < np.unique(y)[max_num_classes]]
                modifications["classes_capped"] = True
            else:
                print("Too many classes")
                continue

        datasets += [
            [entry["name"], X, y, categorical_feats, attribute_names, modifications]
        ]

    return datasets, datalist




class Evaluator:
    def __init__(
        self, model, c, device, N_ensemble_configurations=32, root=default_root
    ):
        model_string = "train"
        epoch = -1
        model_key = model_string + "|" + str(device) + "|" + str(epoch)
        self.root = root
        self.c = c
        self.selected_metadatas = generate_metadata(root)

        # if isinstance(model, EquiTabPFNv2) or isinstance(model, TabPFNv2):
        #     from tabpfn import TabPFNClassifier

        #     self.classifier = TabPFNClassifier()


        # else:

        self.classifier = TabPFNClassifier(
            device=device,
            base_path="",
            model_string=model_string,
            N_ensemble_configurations=N_ensemble_configurations,
            feature_shift_decoder=False,
            multiclass_decoder="",
            no_preprocess_mode=True,
            epoch=epoch,
        )

        self.classifier.models_in_memory[model_key] = (model, self.c, "")
        self.model_key = model_key
        self.device = device

    def eval(self):
        # self.classifier.models_in_memory[self.model_key][0].eval()
        results = []
        for row in tqdm(self.selected_metadatas):
            tid = row["tid"]
            target_feature = row["target_feature"]
            dataset = load_default_dataset(
                tid, target_feature, download=True, root=self.root
            )
            acc = eval_model(self.classifier, dataset)
            results.append({"tid": tid, "accuracy": acc})
        # self.classifier.models_in_memory[self.model_key][0].train()
        equivariance_results = average_equivariance_error(self.classifier, n_seeds=100)
        return pd.DataFrame(results), equivariance_results

    def set_model(self, model, c):
        self.classifier.models_in_memory[self.model_key] = (model, c, "")


class Evaluator_OpenML_CC30:
    def __init__(
        self,
        model,
        c,
        device,
        N_ensemble_configurations=32,
        feature_shift_decoder=False,
        multiclass_decoder="",
        no_preprocess_mode=True,
        root=default_root,
    ):
        model_string = "train"
        epoch = -1
        model_key = self.make_model_key(model_string, device, epoch)
        self.device = device
        self.datasets = prepare_OpenML_CC30_datasets(root=root)
        self.classifier = TabPFNClassifier(
            device=device,
            base_path="",
            model_string=model_string,
            N_ensemble_configurations=N_ensemble_configurations,
            epoch=epoch,
            feature_shift_decoder=feature_shift_decoder,
            multiclass_decoder=multiclass_decoder,
            no_preprocess_mode=no_preprocess_mode,
        )

        self.classifier.models_in_memory[model_key] = (model, c, "")
        self.model_key = model_key

    def set_model(self, model, c):
        self.classifier.models_in_memory[self.model_key] = (model, c, "")

    def make_model_key(self, model_string, device, epoch):
        return model_string + "|" + str(device) + "|" + str(epoch)

    def eval(self):
        openml_results = eval_OpenML_CC30(
            self.classifier, "equitabpfn", self.datasets, self.device
        )

        openml_results = [res for per_dataset in openml_results for res in per_dataset]
        openml_results = pd.DataFrame(openml_results)
        equivariance_results = average_equivariance_error(self.classifier, n_seeds=100)
        return openml_results, equivariance_results





def make_classifier(loading_info = None,  
                    classifier_info={"type":"old",
                                    "ensembles": 1,
                                    "num_classes":10},
                    device="cuda:0",):
        
    from tabpfn import TabPFNClassifier as TabPFNClassifierV2  # or from tabpfn_client
    import tabpfn_extensions
    from tabpfn_extensions.many_class import ManyClassClassifier
    
    from equitabpfn.models.equitabpfnv2 import EquiTabPFNv2
    from equitabpfn.models.equitabpfn import EquiTabPFN

    if loading_info["pretrained"]:
        from equitabpfn.models.tabpfnv2 import TabPFNv2
        model = TabPFNv2(load_dict=True, model_version=loading_info["model_version"])
        args  = model.config_
        cfg = model.config_
    else:
        root, model_name = os.path.split(loading_info["root"])
        model, args = load_model_from_name(root, model_name , device=device, uncompiled_model_keys=True)
        cfg = args
        
    if classifier_info["type"]=="TabPFNV2":


        # Create a base TabPFN classifier
        base_clf = TabPFNClassifierV2(device=device, 
                                        ignore_pretraining_limits=True)
        base_clf.model_ = model
        base_clf.config_ = args
        num_classes = classifier_info["num_classes"]
        # Wrap it with ManyClassClassifier to handle more classes
        if num_classes  >10:
            if classifier_info["multi_class_redundancy"]:
                return ManyClassClassifier(
                    estimator=base_clf,
                    alphabet_size=10,
                    n_estimators_redundancy=classifier_info["multi_class_redundancy"], # Use TabPFN's maximum class limit
                    n_classes=num_classes,
                ), model
            from equitabpfn.utils import OneHot
            if isinstance(model, EquiTabPFNv2):
                base_clf.model_.transformer_encoder.y_encoder[1].one_hot = OneHot(num_classes)
            else:
                base_clf.model_.y_encoder.one_hot = OneHot(num_classes)
            #base_clf.model_.transformer_encoder.y_encoder.one_hot = OneHot(num_classes)
            base_clf.config_['prior']['classification']['max_num_classes']=num_classes
            base_clf.config_['max_num_classes']=num_classes
            base_clf.inference_config = {"MAX_NUMBER_OF_CLASSES": num_classes,
                                         "POLYNOMIAL_FEATURES": "all",
                                         "FINGERPRINT_FEATURE": True}
        
        if isinstance(model, EquiTabPFN):
            from tabpfn.model.encoders import SequentialEncoder, NanHandlingEncoderStep, InputNormalizationEncoderStep

            preprocessor = SequentialEncoder(
                NanHandlingEncoderStep(
                                keep_nans = False,
                                in_keys=("main",),
                                out_keys=("main","nan_indicators"),
                                ),
                InputNormalizationEncoderStep(normalize_on_train_only=True,
                                      normalize_to_ranking=False,
                                      normalize_x=True,
                                      remove_outliers=False,
                                      in_keys=("main",),
                                      out_keys=("output",)
                                      ),
                )
            base_clf.model_.preprocessor = preprocessor

        return base_clf, model


    elif classifier_info["type"]=="TabPFNV1":
        
        num_classes = classifier_info["num_classes"]
        model_string = "train"
        epoch = -1
        model_key = model_string + "|" + str(device) + "|" + str(epoch)
        N_ens = classifier_info["ensembles"]
        assert N_ens >0
        if N_ens ==1:
            feature_shift_decoder = False
            multiclass_decoder = ""
            no_preprocess_mode = True
        else:
            feature_shift_decoder = True
            no_preprocess_mode = False

            if isinstance(model, EquiTabPFN) or isinstance(model, EquiTabPFNv2):
                multiclass_decoder = ''
            else:
                multiclass_decoder = 'permutation'

        classifier = TabPFNClassifier(
            device=device,
            base_path="",
            model_string=model_string,
            N_ensemble_configurations=N_ens,
            epoch=epoch,
            feature_shift_decoder=feature_shift_decoder,
            multiclass_decoder=multiclass_decoder,
            no_preprocess_mode=no_preprocess_mode,
            batch_size_inference=1,
        )

        if num_classes >10:
            if not (isinstance(model, EquiTabPFNv2) or isinstance(model, EquiTabPFN)):
                if classifier_info["multi_class_redundancy"]==0:
                    return None, None
            if classifier_info["multi_class_redundancy"]>0:
                classifier.models_in_memory[model_key] = (model, cfg, "")
                return ManyClassClassifier(
                    estimator=classifier,
                    alphabet_size=10,
                    n_estimators_redundancy=classifier_info["multi_class_redundancy"],  # Use TabPFN's maximum class limit
                    n_classes=num_classes,
                ), model
            cfg['prior']['classification']['max_num_classes']=num_classes
            cfg['max_num_classes']=num_classes
            classifier.num_classes = num_classes

            from equitabpfn.utils import OneHot
            if isinstance(model, EquiTabPFNv2):
                model.transformer_encoder.y_encoder[1].one_hot = OneHot(num_classes)
            else:
                model.y_encoder.one_hot = OneHot(num_classes)

        classifier.models_in_memory[model_key] = (model, cfg, "")
        if isinstance(model, EquiTabPFN):
            from tabpfn.model.encoders import SequentialEncoder, NanHandlingEncoderStep, InputNormalizationEncoderStep

            preprocessor = SequentialEncoder(
                NanHandlingEncoderStep(
                                keep_nans = False,
                                in_keys=("main",),
                                out_keys=("main","nan_indicators"),
                                ),
                InputNormalizationEncoderStep(normalize_on_train_only=True,
                                      normalize_to_ranking=False,
                                      normalize_x=True,
                                      remove_outliers=False,
                                      in_keys=("main",),
                                      out_keys=("output",)
                                      ),
                )
            model.preprocessor = preprocessor


        return classifier, model
    else:
        raise NotImplementedError

class Evaluator_OpenML:
    def __init__(self, classifier, device, dids_type="regular", root=default_root, return_capped=True,num_feats=100):
        self.classifier = classifier
        self.datasets = prepare_OpenML_CC30_datasets(root=root,dids_type=dids_type, return_capped=return_capped, num_feats=num_feats)
        self.device = device
    def eval(self):
        openml_results = eval_OpenML_CC30(
            self.classifier, "equitabpfn", self.datasets, self.device, force=True
        )

        openml_results = [res for per_dataset in openml_results for res in per_dataset]
        openml_results = pd.DataFrame(openml_results)
        #equivariance_results = average_equivariance_error(self.classifier, n_seeds=100)
        equivariance_results = -1.
        return openml_results, equivariance_results

 

class Evaluator_OpenML_OOD:
    def __init__(
        self,
        model,
        c,
        device,
        N_ensemble_configurations=32,
        feature_shift_decoder=False,
        multiclass_decoder="",
        no_preprocess_mode=True,
        root=default_root,
    ):
        model_string = "train"
        epoch = -1
        model_key = self.make_model_key(model_string, device, epoch)
        self.device = device
        self.datasets = prepare_OpenML_CC30_datasets(root=root,dids_type="ood")
        self.classifier = TabPFNClassifier(
            device=device,
            base_path="",
            model_string=model_string,
            N_ensemble_configurations=N_ensemble_configurations,
            epoch=epoch,
            feature_shift_decoder=feature_shift_decoder,
            multiclass_decoder=multiclass_decoder,
            no_preprocess_mode=no_preprocess_mode,
        )

        max_num_classes = 30
        c['prior']['classification']['max_num_classes']=max_num_classes
        c['max_num_classes']=max_num_classes
        self.classifier.models_in_memory[model_key] = (model, c, "")
        self.classifier.max_num_classes = max_num_classes
        from equitabpfn.utils import OneHot
        model.y_encoder.one_hot = OneHot(max_num_classes)
        self.model_key = model_key

    def set_model(self, model, c):
        self.classifier.models_in_memory[self.model_key] = (model, c, "")

    def make_model_key(self, model_string, device, epoch):
        return model_string + "|" + str(device) + "|" + str(epoch)

    def eval(self):
        openml_results = eval_OpenML_CC30(
            self.classifier, "equitabpfn", self.datasets, self.device
        )

        openml_results = [res for per_dataset in openml_results for res in per_dataset]
        openml_results = pd.DataFrame(openml_results)
        equivariance_results = average_equivariance_error(self.classifier, n_seeds=100)
        return openml_results, equivariance_results



# data_path = Path(__file__).parent.parent / "data"
def tabrepo_metadata(root):
    local_path = os.path.join(root, "task_metadata_289.csv")
    # data_path = Path(__file__).parent.parent / "data"
    # local_path = data_path / "task_metadata_289.csv"
    if not Path(local_path).exists():
        os.makedirs(root, exist_ok=True)
        import subprocess

        link = "https://raw.githubusercontent.com/autogluon/tabrepo/main/data/metadata/task_metadata_289.csv"
        result = subprocess.run(
            ["wget", link, "-O", local_path], capture_output=True, text=True
        )

        # Check if the command was successful
        if result.returncode == 0:
            print("Download successful")
        else:
            print("Download failed")
            print("Error:", result.stderr)

    return pd.read_csv(local_path)


def generate_metadata(root):
    df_metadata = tabrepo_metadata(root)

    # tabpfn only supports classification with up to 10 classes and up to 1024 rows
    df_metadata = df_metadata[df_metadata.NumberOfClasses < 10]
    df_metadata = df_metadata[df_metadata.NumberOfInstances < 1024]
    df_metadata = df_metadata[df_metadata.task_type == "Supervised Classification"]

    # exclude tids with bug/errors in openml
    excluded_tids = [49]
    df_metadata = df_metadata[~df_metadata.tid.isin(excluded_tids)]

    selected_metadatas = list(df_metadata.to_dict(orient="records"))[:4]
    return selected_metadatas


def get_dataset(tid, target_feature=None, dataset_format=None, root=default_root):
    data_path = os.path.join(root, "datasets")
    os.makedirs(data_path, exist_ok=True)
    openml.config.cache_directory = os.path.expanduser(data_path)

    if target_feature is None:
        local_path = os.path.join(data_path, f"tid_{tid}")
    else:
        local_path = os.path.join(
            data_path, f"tid_{tid}_target_feature_{target_feature}"
        )

    if not Path(local_path).exists():
        dataset = openml.datasets.get_dataset(
            tid,
            download_data=True,
            download_qualities=True,
            download_features_meta_data=True,
        )
        if target_feature is None:
            target_feature = dataset.default_target_attribute
        else:
            feature_names = [x.name for x in dataset.features.values()]
            if target_feature not in feature_names:
                raise ValueError(f"target {target_feature} not found")
        if dataset_format is None:
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                target=target_feature
            )
        else:
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                target=target_feature, dataset_format=dataset_format
            )
        data = {
            "X": X,
            "y": y,
            "categorical_indicator": categorical_indicator,
            "attribute_names": attribute_names,
        }
        with open(local_path, "wb") as f:
            pickle.dump(data, f)
    else:
        with open(local_path, "rb") as f:
            data = pickle.load(f)
        X = data["X"]
        y = data["y"]
        categorical_indicator = data["categorical_indicator"]
        attribute_names = data["attribute_names"]

    return X, y, categorical_indicator, attribute_names


def load_default_dataset(tid, target_feature, root=default_root, download=False):
    X, y, categorical_indicator, attribute_names = get_dataset(
        tid, target_feature, root=root
    )

    cat_features = [
        col for col, is_cat in zip(attribute_names, categorical_indicator) if is_cat
    ]
    dataset = {"X": X, "y": y, "cat_features": cat_features}
    return dataset


def eval_model(model, dataset):
    X_train, X_test, y_train, y_test = train_test_split(
        dataset["X"], dataset["y"], test_size=0.33, random_state=0
    )
    X_train, X_test = featurize(X_train, X_test, cat_cols=dataset["cat_features"])
    scaler = MinMaxScaler().fit(X_train)

    # TODO this should be optional as TabPFN does the normalization IIRC
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    with torch.no_grad():
        model.fit(X_train, y_train)
        y_eval, p_eval = model.predict(
            X_test,
            return_winning_probability=True,
        )
    acc = accuracy_score(y_test, y_eval)
    return acc


def featurize(
    df_train: pd.DataFrame, df_test: pd.DataFrame, cat_cols: List[str] = None
):
    feature_generator = AutoMLPipelineFeatureGenerator()

    X_train = feature_generator.fit_transform(X=df_train)
    imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
    imp_mean.fit(X_train)
    X_train = imp_mean.transform(X_train)

    X_test = feature_generator.transform(df_test)
    X_test = imp_mean.transform(X_test)

    return X_train, X_test




def prepare_OpenML_CC30_datasets(root=default_root, dids_type='regular', return_capped=True,num_feats=100):
    from mothernet.datasets import open_cc_dids
    if dids_type =="regular":
        dids = open_cc_dids
        datasets_name="datalist.pkl"
        max_num_classes = 10

    elif dids_type=="ood":
        dids = open_cc_ood_dids
        datasets_name="ood_datalist.pkl"
        max_num_classes = 30

    else:
        raise NotImplementedError

    datasets, datasets_df = load_openml_list(
        dids,
        multiclass=True,
        shuffled=True,
        filter_for_nan=False,
        max_samples=10000,
        num_feats=num_feats,
        max_num_classes=max_num_classes,
        return_capped=return_capped,
        root=root,
        datasets_name=datasets_name
    )
    #print(lol)
    # datasets = [datasets[0]]
    datasets_df["isNumeric"] = (datasets_df.NumberOfSymbolicFeatures == 1) & (
        datasets_df.NumberOfInstancesWithMissingValues == 0
    )
    datasets_df["NumberOfInstances"] = datasets_df["NumberOfInstances"].astype(int)
    datasets_df["NumberOfFeatures"] = datasets_df["NumberOfFeatures"].astype(int)
    datasets_df["NumberOfClasses"] = datasets_df["NumberOfClasses"].astype(int)
    return datasets


def eval_OpenML_CC30(model, model_name, datasets, device, force=False):
    clf_dict = {model_name: model}

    results = eval_models_on_datasets(
        clf_dict,
        datasets,
        eval_positions=[1000],
        n_samples=2000,
        max_times=[1],
        split_numbers=[1, 2, 3, 4, 5],
        #split_numbers=[4],
        n_jobs=1,
        device=device,
        force=force
    )
    return results


def eval_OpenML_CC30_baselines(
    datasets, device, methods=[],  fname="", deep_models=False, N_ens=1
    ):

    ensembling=N_ens>1
    results = []
    max_times = [1, 15, 30, 60, 60 * 5, 60 * 15, 60 * 60]
    split_numbers = [1, 2, 3, 4, 5]
    #max_times = [60 * 60]
    #split_numbers = [1]
    
    
    
    if deep_models:
    
    # Eval NN baselines
        from mothernet.prediction.tabpfn import TabPFNClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from mothernet.evaluation.baselines.distill_mlp import DistilledTabPFNMLP
        from functools import partial
        from mothernet.prediction.mothernet import EnsembleMeta, MotherNetClassifier

        if ensembling:
            feature_shift_decoder = True
            multiclass_decoder = ""
            no_preprocess_mode = False
            power = True
            label_shift = True
            feature_shift = True
        else:
            feature_shift_decoder = (False,)
            multiclass_decoder = ("",)
            no_preprocess_mode = True
            power = False
            label_shift = False
            feature_shift = False
            N_ens = 1

        tabpfn = TabPFNClassifier(
            device=device,
            N_ensemble_configurations=N_ens,
            feature_shift_decoder=feature_shift_decoder,
            multiclass_decoder=multiclass_decoder,
            no_preprocess_mode=no_preprocess_mode,
        )

        # mlp_distill_model_string = "tabpfn_nooptimizer_emsize_512_nlayers_12_steps_2048_bs_32ada_lr_0.0001_1_gpu_07_24_2023_01_43_33"
        # mlp_distill = make_pipeline(StandardScaler(), DistilledTabPFNMLP(n_epochs=1000,
        # device=device,
        # hidden_size=128,
        # n_layers=2,
        # dropout_rate=.1,
        # learning_rate=0.01,
        # model_string=mlp_distill_model_string,
        # epoch=1650,
        # N_ensemble_configurations=3))

        from mothernet.utils import get_mn_model

        mothernet_model_string = "mn_d2048_H4096_L2_W32_P512_1_gpu_warm_08_25_2023_21_46_25_epoch_3940_no_optimizer.pickle"
        mothernet = EnsembleMeta(
            MotherNetClassifier(path=get_mn_model(mothernet_model_string), device=device),
            n_estimators=N_ens,
            power=power,
            label_shift=label_shift,
            feature_shift=feature_shift,
        )

        clf_dict = {
            f"tabpfn_{N_ens}": tabpfn,
            f"mothernet_{N_ens}": partial(
                transformer_metric, classifier=mothernet, onehot=True
            ),
            #'mlp_distill': mlp_distill,
        }

        results += eval_models_on_datasets(
            clf_dict,
            datasets,
            eval_positions=[1000],
            n_samples=2000,
            max_times=[1],
            split_numbers=split_numbers,
            n_jobs=1,
            device=device,
        )

    else:
        # Eval tabular_baselines baselines
        from mothernet.evaluation.baselines.tabular_baselines import (
            knn_metric,
            logistic_metric,
            xgb_metric,
            random_forest_metric,
            mlp_metric,
        )

        clf_dict = {
            "logistic": logistic_metric,
            "knn": knn_metric,
            "rf_new_params": random_forest_metric,
            "xgb": xgb_metric,
            "mlp": mlp_metric,
        }
        if methods:
            clf_dict = { key : value for key, value in clf_dict.items() if key in methods }

        results += eval_models_on_datasets(
            clf_dict,
            datasets,
            eval_positions=[1000],
            n_samples=2000,
            max_times=max_times,
            split_numbers=split_numbers,
            fetch_only=False,
            device=device,
        )

    import pickle

    parent_dir = os.path.dirname(fname)
    os.makedirs(parent_dir, exist_ok=True)
    with open(fname, "wb") as f:
        pickle.dump(results, f)

    return results


def eval_models_on_datasets(clf_dict, datasets, **kwargs):
    from mothernet.evaluation import tabular_metrics

    metric_used = tabular_metrics.auc_metric

    results = [
        format(
            eval_on_datasets(
                "multiclass",
                model,
                model_name,
                datasets,
                base_path="",
                metric_used=metric_used,
                overwrite=True,
                save=False,
                **kwargs,
            )
        )
        for model_name, model in clf_dict.items()
    ]

    return results


def format(results):
    out = len(results) * [None]
    for i, result in enumerate(results):
        metric = (
            result["mean_metric"].item()
            if isinstance(result["mean_metric"], torch.Tensor)
            else result["mean_metric"]
        )
        out[i] = {
            "model": result["model"],
            "dataset": result["dataset"],
            "split": result["split"],
            "max_time": result["max_time"],
            "mean_metric": metric,
            "time_used": result["time_used"]
        }
    return out


def models_from_ckpts(ckpt_dict, device):
    model_string = "train"
    epoch = -1
    model_key = model_string + "|" + str(device) + "|" + str(epoch)

    def get_classifier(ckpt):
        # states = torch.load(ckpt, map_location='cpu')
        model, args = load_model(ckpt, device=device, verbose=False)
        classifier = TabPFNClassifier(
            device=device,
            base_path="",
            model_string=model_string,
            N_ensemble_configurations=32,
            epoch=epoch,
        )
        classifier.models_in_memory[model_key] = (model, args, "")
        return classifier

    return {key: get_classifier(ckpt) for key, ckpt in ckpt_dict.items()}, model_key


def eval_baselines(
    datasets,
    device,
    methods = [],
    base_path=os.path.join("data/baselines"),
    name="all",
    deep_models=False,
    force=False,
    N_ens=3,
):
    fname = os.path.join(base_path, f"{name}.pickle")
    if not os.path.exists(fname) or force:
        results = eval_OpenML_CC30_baselines(
            datasets,
            device,
            methods= methods,
            fname=fname,
            deep_models=deep_models,
            N_ens=N_ens,
        )
        results = [res for per_dataset in results for res in per_dataset]
        results = pd.DataFrame(results)
        import pickle

        with open(fname, "wb") as file:
            pickle.dump(results, file)
    else:
        import pickle

        with open(fname, "rb") as file:
            results = pickle.load(file)
    return results


def eval_models(datasets, ckpt_dict, device, base_path=os.path.join("data/baselines")):
    results = []

    for key, ckpt in ckpt_dict.items():
        model_dict, model_key = models_from_ckpts({key: ckpt}, device)

        # for key, classifier in model_dict.items():
        #    model = classifier.models_in_memory[model_key][0]
        #    model.decoder.kwarg["bw"] = 1.

        result = eval_models_on_datasets(
            model_dict,
            datasets,
            eval_positions=[1000],
            n_samples=2000,
            max_times=[1],
            split_numbers=[1, 2, 3, 4, 5],
            n_jobs=1,
            device=device,
        )
        results += result

    results = [res for per_dataset in results for res in per_dataset]
    results = pd.DataFrame(results)

    return results


if __name__ == "__main__":
    import sys
    from pathlib import Path

    ckpt_path = Path(sys.argv[1:][0]).expanduser()
    assert ckpt_path.exists()
    ckpt_dict = {"ckpt": ckpt_path}
    device = "cpu"
    datasets = prepare_OpenML_CC30_datasets()

    results = eval_models(datasets, ckpt_dict, device)
    print(results["mean_metric"].mean())
