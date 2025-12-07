import numpy as np
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from equitabpfn.models.equitabpfn_classifier import EquiTabPFNClassifier


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


def main():
    # fit tabpfn-v2 and tabpfn-v2.5
    # Look at this https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/classifier.py
    classifiers = {
        "TabPFN-v2": TabPFNClassifier.create_default_for_version(ModelVersion.V2),
        "TabPFN-v2.5": TabPFNClassifier.create_default_for_version(ModelVersion.V2_5),
        "EquiTabPFN": EquiTabPFNClassifier(epoch=1200),
    }

    n_train = 9
    n_test = 100
    for name, classifier in classifiers.items():
        print(f"Fitting {name}")
        # X_test = make_grid(1600)
        X_test = make_grid(n_test)
        X_train = make_grid(n_train)
        y_train = np.arange(n_train)
        model = classifier.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(y_pred)


if __name__ == '__main__':
    main()