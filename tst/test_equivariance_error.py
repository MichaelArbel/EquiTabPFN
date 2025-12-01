import numpy as np
import torch

from equitabpfn.evaluation.equivariance_error import permute_y, permute_y_prob


def test_permute_y():
    n_rows = 12
    batch_size = 1
    n_classes = 10
    perm = np.array([0, 2, 1, 3, 4, 5, 6, 7, 8, 9])
    y = torch.randint(low=0, high=n_classes, size=(n_rows, batch_size))
    y_perm = permute_y(y, perm).numpy()

    assert y_perm.shape == (n_rows, batch_size)

    for x1, x2 in zip(y_perm.reshape(-1), y.reshape(-1)):
        assert x1 == perm[x2]


def test_permute_y_prob():
    n_rows = 12
    batch_size = 1
    n_classes = 2
    perm = np.array([0, 2, 1])
    y_prob = torch.rand(size=(n_rows, batch_size, n_classes + 1))
    y_prob_perm = permute_y_prob(y_prob, perm).numpy()

    assert y_prob_perm.shape == (n_rows, batch_size, n_classes + 1)
    y_prob = y_prob.numpy()
    for i in range(n_rows):
        for j in range(batch_size):
            prob1 = y_prob[i][j]
            prob2 = y_prob_perm[i][j]
            for k in range(n_classes):
                assert prob1[k] == prob2[perm[k]]
