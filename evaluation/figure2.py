# Trying example from the repo
import pickle

import pandas as pd

from equitabpfn.models.equitabpfn_classifier import EquiTabPFNClassifier
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mothernet import TabPFNClassifier

from equitabpfn.evaluation.equivariance_error import permute_y
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import Voronoi, voronoi_plot_2d

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


def pred_equi(classifier, X_train, y_train, X_test):
    # compute predictions from equipfn

    classifier.fit(X_train, y_train)
    with torch.no_grad():
        y_eval, p_eval = classifier.predict(
            X_test,
            return_winning_probability=True,
        )
    return y_eval


def compute_tabpfn(X_train, y_train, sigma, X_test):
    y_train_perm = permute_y(torch.Tensor(y_train), sigma).numpy()
    from mothernet.prediction.tabpfn import TabPFNClassifier as TabPFNv1

    classifier = TabPFNv1(device="cpu", N_ensemble_configurations=1)
    classifier.fit(X_train, y_train_perm)
    y_eval, p_eval = classifier.predict(X_test, return_winning_probability=True)
    y_eval = permute_y(torch.Tensor(y_eval), np.argsort(sigma)).numpy()
    return y_eval


def compute_tabpfn_v2(X_train, X_test, y_train, sigma):
    # from tabpfn import TabPFNClassifier as TabPFNv2
    y_train_perm = permute_y(torch.Tensor(y_train), sigma).numpy()

    from tabpfn_client import init
    from tabpfn_client import TabPFNClassifier as TabPFNv2

    classifier = TabPFNv2(n_estimators=1, paper_version=True)
    classifier.fit(X_train, y_train_perm)
    y_eval = classifier.predict(X_test)
    y_eval = permute_y(torch.Tensor(y_eval), np.argsort(sigma)).numpy()
    return y_eval


def compute_equi_tabpfn(X_train, X_test, y_train, sigma):
    equipfn_classifier = EquiTabPFNClassifier(epoch=1200)

    y_train_perm = permute_y(torch.Tensor(y_train), sigma).numpy()

    equipfn_classifier.fit(X_train, y_train_perm)
    with torch.no_grad():
        y_eval, p_eval = equipfn_classifier.predict(
            X_test,
            return_winning_probability=True,
        )

    return permute_y(torch.Tensor(y_eval), np.argsort(sigma)).numpy()


def plot(name: str, generate_fun, N: int = 1600):
    seeds = [0, 1, 2]
    fig, axes = plt.subplots(3, 3, figsize=(7, 4.5))

    # X_test = make_grid(1600)
    X_test = make_grid(N)
    N = 9
    X_train = generate_fun(N)
    y_train = np.arange(N)

    do_compute_tabpfn_v2 = True
    recompute = True
    if recompute:
        # compute predictions for all methods
        y_eval_tabpfn = {}
        y_eval_tabpfn_v2 = {}
        y_eval_equitabpfn = {}
        for i, seed in enumerate(seeds):
            np.random.seed(i)
            sigma = np.random.permutation(N)
            y_eval_tabpfn[i] = compute_tabpfn(
                X_train=X_train, X_test=X_test, y_train=y_train, sigma=sigma
            )

            if do_compute_tabpfn_v2:
                y_eval_tabpfn_v2[i] = compute_tabpfn_v2(
                    X_train=X_train, X_test=X_test, y_train=y_train, sigma=sigma
                )

            y_eval_equitabpfn[i] = compute_equi_tabpfn(
                X_train,
                X_test,
                y_train,
                sigma,
            )

        with open(data_root / f"{name}-multiclass-voronoi.pkl", "wb") as f:
            pickle.dump(
                {
                    "y_eval_equitabpfn": y_eval_equitabpfn,
                    "y_eval_tabpfn_v2": y_eval_tabpfn_v2,
                    "y_eval_tabpfn": y_eval_tabpfn,
                },
                f,
            )

    with open(data_root / f"{name}-multiclass-voronoi.pkl", "rb") as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data = pickle.load(f)
        y_eval_equitabpfn = data["y_eval_equitabpfn"]
        y_eval_tabpfn_v2 = data["y_eval_tabpfn_v2"]
        y_eval_tabpfn = data["y_eval_tabpfn"]

    # plot for all methods
    for i, seed in enumerate(seeds):
        np.random.seed(i)
        sigma = np.random.permutation(N)
        axes[0][i].set_box_aspect(aspect=1)
        axes[0][i].scatter(
            X_test[:, 0], X_test[:, 1], c=y_eval_tabpfn[i], cmap=cmap, marker="."
        )
        axes[0][i].scatter(X_train[:, 0], X_train[:, 1], c="black", marker="x")
        axes[0][i].set_title(f"Targets permuted with $\sigma_{i}$")
        axes[0][i].set_xticks([])
        axes[0][i].set_yticks([])
        if i == 0:
            axes[0][i].set_ylabel("TabPFN")

        if do_compute_tabpfn_v2:
            axes[1][i].scatter(
                X_test[:, 0], X_test[:, 1], c=y_eval_tabpfn_v2[i], cmap=cmap, marker="."
            )
            axes[1][i].scatter(X_train[:, 0], X_train[:, 1], c="black", marker="x")
            axes[1][i].set_xticks([])
            axes[1][i].set_yticks([])
            if i == 0:
                axes[1][i].set_ylabel("TabPFNv2")

        axes[2][i].scatter(
            X_test[:, 0], X_test[:, 1], c=y_eval_equitabpfn[i], cmap=cmap, marker="."
        )
        axes[2][i].scatter(X_train[:, 0], X_train[:, 1], c="black", marker="x")
        # axes[1][i].set_title(f"EquiTabPFN Seed = {i}")
        axes[2][i].set_xticks([])
        axes[2][i].set_yticks([])
        if i == 0:
            axes[2][i].set_ylabel("EquiTabPFN")


if __name__ == "__main__":

    # load equipfn model

    # plot(make_S2)
    # plt.tight_layout()
    # plt.savefig(figure_path() / "boundary-s2.pdf")
    # plt.show()

    plot(
        name="grid",
        generate_fun=make_grid,
        N=900,
    )
    plt.rcParams.update({"font.size": 10})
    plt.tight_layout()
    plt.savefig(figure_path() / "boundary-grid.pdf")
    plt.show()
