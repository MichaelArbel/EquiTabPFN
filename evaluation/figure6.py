# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import mothernet
from equitabpfn.models.model_builder import load_model_from_name
from equitabpfn.utils import figure_path
from tabpfn import TabPFNClassifier

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "Random Forest",
    "Neural Net",
    "TabPFN",
    "TabPFN + ens.",
    "EquiTabPFN",
    "EquiTabPFN + ens.",
]

device = "cpu"
checkpoint_path = Path(__file__).parent.parent / "data/models"
model, config = load_model_from_name(root=checkpoint_path, model_name="equitabpfn")

epoch = -1
model_string = "train"
model_key = model_string + "|" + str(device) + "|" + str(epoch)

equipfn_classifier = mothernet.TabPFNClassifier(
    device=device,
    base_path="",
    model_string=model_string,
    N_ensemble_configurations=1,
    # feature_shift_decoder=False,
    # multiclass_decoder="",
    # no_preprocess_mode=True,
    epoch=epoch,
)
equipfn_classifier.models_in_memory[model_key] = (model, config, "")

equipfn_classifier_ens = mothernet.TabPFNClassifier(
    device=device,
    base_path="",
    model_string=model_string,
    N_ensemble_configurations=32,
    # feature_shift_decoder=False,
    # multiclass_decoder="",
    # no_preprocess_mode=True,
    epoch=epoch,
)
equipfn_classifier_ens.models_in_memory[model_key] = (model, config, "")

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    TabPFNClassifier(n_estimators=1),
    TabPFNClassifier(n_estimators=32),
    equipfn_classifier,
    equipfn_classifier_ens,
]

assert len(classifiers) == len(names)
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=2, n_clusters_per_class=1
)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [
    make_moons(noise=0.3, random_state=0),
    make_circles(noise=0.2, factor=0.5, random_state=1),
    linearly_separable,
]

figure = plt.figure(figsize=(12, 4))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        cmap=cm_bright,
        edgecolors="k",
        s=15,
        linewidth=0.3,
    )
    # Plot the testing points
    ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap=cm_bright,
        alpha=0.6,
        edgecolors="k",
        s=15,
        linewidth=0.3,
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1
    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        print(f"Processing dataset#{ds_cnt} and {name}")
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        if name in ["TabPFN", "EquiTabPFN"]:
            DecisionBoundaryDisplay.from_estimator(
                clf,
                X,
                cmap=cm,
                alpha=0.8,
                ax=ax,
                eps=1,
                response_method="predict_proba",
            )
        else:
            DecisionBoundaryDisplay.from_estimator(
                clf, X, cmap=cm, alpha=0.8, ax=ax, eps=1
            )

        # Plot the training points
        ax.scatter(
            X_train[:, 0],
            X_train[:, 1],
            c=y_train,
            cmap=cm_bright,
            edgecolors="k",
            s=15,
            linewidth=0.3,
        )
        # Plot the testing points
        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_test,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.6,
            s=15,
            linewidth=0.3,
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(
            x_max - 0.3,
            y_min + 0.3,
            ("%.2f" % score).lstrip("0"),
            size=15,
            horizontalalignment="right",
        )
        i += 1

plt.tight_layout()
plt.show()
figure.savefig(figure_path() / "classification_boundary_comparison.pdf", dpi=300)
