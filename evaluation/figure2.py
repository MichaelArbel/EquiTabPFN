# Trying example from the repo
import pickle

import pandas as pd

from equitabpfn.models.equitabpfn_classifier import EquiTabPFNClassifier
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from equitabpfn.evaluation.equivariance_error import permute_y

from equitabpfn.utils import figure_path

checkpoint_path = Path(__file__).parent.parent / "checkpoints/"
cmap = "Set1"
device = "cpu"
data_root = Path(__file__).parent.parent / "data" / "multiclass"
data_root.mkdir(exist_ok=True, parents=True)


def draw_random_examples(n_classes: int = 9):
    # draw points uniformly in [0, 1]^2
    X_train = np.random.rand(n_classes, 2)
    y_train = np.arange(len(X_train))
    return X_train, y_train


def make_S2(n: int):
    pts = [(np.cos(2 * np.pi * k / n), np.sin(2 * np.pi * k / n)) for k in range(n)]
    # center points, apply 0.9 for plotting
    pts = 0.9 * np.array(pts)
    pts = 0.5 + pts / 2
    return pts


def make_grid(n: int):
    # build a grid of points of n in [0, 1]^2
    n_sqrt = int(np.sqrt(n))
    pts = []
    for i in range(n_sqrt):
        pts.append(
            np.stack([np.arange(n_sqrt) / n_sqrt, i * np.ones(n_sqrt) / n_sqrt]).T
        )
    pts = np.concatenate(pts) + 1 / (2 * n_sqrt)
    return pts


def compute_predictions(classifier, X_train, X_test, y_train, sigma):
    """Compute predictions for a given classifier with permuted labels."""
    y_train_perm = permute_y(torch.Tensor(y_train), sigma).numpy()

    classifier.fit(X_train, y_train_perm)

    # Try predict with return_winning_probability for compatibility
    try:
        y_eval, p_eval = classifier.predict(X_test, return_winning_probability=True)
    except TypeError:
        # If classifier doesn't support return_winning_probability, use simple predict
        y_eval = classifier.predict(X_test)

    # Unpermute the predictions
    y_eval = permute_y(torch.Tensor(y_eval), np.argsort(sigma)).numpy()
    return y_eval


def plot(name: str, generate_fun, N: int = 1600):
    seeds = [0, 1, 2]
    classifiers = {
        "TabPFN-v2": TabPFNClassifier.create_default_for_version(ModelVersion.V2),
        "TabPFN-v2.5": TabPFNClassifier.create_default_for_version(ModelVersion.V2_5),
        "EquiTabPFN": EquiTabPFNClassifier(epoch=1200),
    }
    n_methods = len(classifiers)

    fig, axes = plt.subplots(n_methods, 3, figsize=(7, 4.5))

    # X_test = make_grid(1600)
    X_test = make_grid(N)
    N = 9
    X_train = generate_fun(N)
    y_train = np.arange(N)

    recompute = True
    if recompute:
        # compute predictions for all methods
        y_eval_dict = {method_name: {} for method_name in classifiers.keys()}

        for i, seed in enumerate(seeds):
            np.random.seed(i)
            sigma = np.random.permutation(N)

            for method_name, classifier in classifiers.items():
                print(f"Computing {method_name} for seed {i}")
                y_eval_dict[method_name][i] = compute_predictions(
                    classifier=classifier,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    sigma=sigma
                )

        with open(data_root / f"{name}-multiclass-voronoi.pkl", "wb") as f:
            pickle.dump(y_eval_dict, f)

    with open(data_root / f"{name}-multiclass-voronoi.pkl", "rb") as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        y_eval_dict = pickle.load(f)

    # plot for all methods
    method_names = list(classifiers.keys())
    for row_idx, method_name in enumerate(method_names):
        for i, seed in enumerate(seeds):
            np.random.seed(i)

            # axes[row_idx][i].set_box_aspect(aspect=1)
            axes[row_idx][i].scatter(
                X_test[:, 0], X_test[:, 1], c=y_eval_dict[method_name][i], cmap=cmap,
                marker=".", s=20,
            )
            axes[row_idx][i].scatter(X_train[:, 0], X_train[:, 1], c="black", marker="x")

            if row_idx == 0:
                axes[row_idx][i].set_title(f"Targets permuted with $\sigma_{i}$")

            axes[row_idx][i].set_xticks([])
            axes[row_idx][i].set_yticks([])

            if i == 0:
                axes[row_idx][i].set_ylabel(method_name)


if __name__ == "__main__":

    # load equipfn model

    plot(name="S2", generate_fun=make_S2)
    plt.tight_layout()
    plt.savefig(figure_path() / "boundary-s2.pdf")
    plt.show()

    plot(
        name="grid",
        generate_fun=make_grid,
        N=900,
    )
    plt.rcParams.update({"font.size": 10})
    plt.tight_layout()
    plt.savefig(figure_path() / "boundary-grid.pdf")
    plt.show()
