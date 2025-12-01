import numpy as np
import pandas as pd
from mothernet import TabPFNClassifier
import torch
from torch import nn
from tqdm import tqdm

from equitabpfn.utils import Criterion


def average_equivariance_error(classifier, n_seeds: int = 100) -> float:
    # TODO make sampling deterministic to have less variance when comparing over epochs
    errors = []
    for i in range(n_seeds):
        # TODO sample from prior and not from this artificial dataset
        X_train, y_train, X_test = draw_random_examples()
        errors.append(equivariance_error(classifier, X_train, y_train, X_test))
    return np.mean(errors)


def equivariance_error(
    classifier, X_train: np.array, y_train: np.array, X_test: np.array
) -> float:
    assert len(X_train) == len(y_train)
    assert X_train.ndim == 2
    assert X_test.ndim == 2
    assert X_train.shape[1] == X_test.shape[1]

    m = y_train.max()
    perm = np.random.permutation(m + 1)

    def apply_perm(y, perm):
        return np.array([perm[y_] for y_ in y])

    classifier.fit(X_train, apply_perm(y_train, perm))
    y_pred_perm_prob = classifier.predict_proba(X_test)
    y_pred_perm = np.argmax(y_pred_perm_prob, axis=1)
    classifier.fit(X_train, y_train)
    y_pred_prob = classifier.predict_proba(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    equivariance_error_class = 1 - (y_pred_perm == apply_perm(y_pred, perm)).mean()
    return equivariance_error_class


def draw_random_examples(n_classes: int = 9):
    X_train = np.random.rand(n_classes, 2)
    y_train = np.arange(len(X_train))
    n_test = 20
    X_test = []
    for i in range(n_test):
        X_test.append(
            np.stack([np.arange(n_test) / n_test, i * np.ones(n_test) / n_test]).T
        )
    X_test = np.concatenate(X_test)
    return X_train, y_train, X_test


def permute_y_prob(y_prob: torch.Tensor, perm: np.array):
    """
    :param y: labels (n_rows, batch_size, n_class)
    :param perm: array one to one from n_class to n_class
    :return: (n_rows, batch_size, n_class) where classes are permuted with perm
    """
    return torch.stack([y_prob[:, :, perm[i]] for i in range(len(perm))], dim=2)


def permute_y(y: torch.Tensor, perm: np.array):
    """
    :param y: labels (n_rows, batch_size,)
    :param perm: array one to one from n_class to n_class
    :return: (n_rows, batch_size) where classes are permuted with perm
    """
    inv_perm = np.argsort(perm)
    with torch.no_grad():
        # TODO is -100 for missing values?
        return y.clone().apply_(
            lambda x: float(inv_perm[int(x)]) if x != -100 else -100
        )


def sample_non_identity_permutation(
    num_classes: int, max_num_classes: int, num_tries: int = 10
):
    perm = np.random.permutation(num_classes)
    # exclude identity with ugly reject sampling
    if np.all(perm == np.arange(num_classes)):
        for i in range(num_tries):
            perm = np.random.permutation(num_classes)
            if not np.all(perm == np.arange(num_classes)):
                break
    # completes the permutation with unused classes to be able to stack the prediction probabilities later on
    # eg if the permutation is [0, 2, 1] then [0, 2, 1, 3, 4, ..., max_num_classes] is returned
    if num_classes < max_num_classes:
        perm = np.array([perm[i] if i < num_classes else i for i in range(10)])
    return perm


def compute_equivariance_metrics(
    dataloader,
    module: nn.Module,
    n_batch: int = 10,
    device: str = "cuda",
    n_ensemble: int = 1,
) -> pd.DataFrame:
    # module = module.double()

    clf_errors = []
    l2_logits = []
    # equi_errors = []
    list_num_classes = []
    for batch, (data, targets, single_eval_pos) in tqdm(
        enumerate(dataloader), total=n_batch
    ):
        # for batch, (data, targets, single_eval_pos) in enumerate(dataloader):
        with torch.no_grad():
            max_num_classes = 10
            if batch >= n_batch:
                break
            # x_src: (n_rows, batch_size, feature_dim)
            # y_src: (n_rows, batch_size,)
            (style_src, x_src, y_src) = (
                tuple(e.to(device) if torch.is_tensor(e) else e for e in data)
                if isinstance(data, tuple)
                else data.to(device)
            )
            # x_src = x_src.double()
            # y_src = y_src.double()

            # (n_rows, batch_size, feature_dim) = x_src.shape
            # n_test_rows = n_rows - single_eval_pos
            num_classes = int(y_src.max().item()) + 1

            def pred_fun(
                module,
                style_src,
                x_src,
                y_src,
                single_eval_pos,
                n_ensemble: int,
                num_classes: int,
                max_num_classes: int,
            ):
                if n_ensemble == 1:
                    return module(
                        (style_src, x_src, y_src), single_eval_pos=single_eval_pos
                    )
                else:
                    # compute  E_σ[σ−1f(X,σ(Y),X⋆)] where the expectation is approximated with n_ensemble members
                    preds = []
                    for _ in range(n_ensemble):
                        # draws permutations, for each permute y before asking for prediction
                        # and then apply inverse permutation to the predictions
                        sigma = sample_non_identity_permutation(
                            num_classes, max_num_classes=max_num_classes
                        )

                        y_src_sigma = permute_y(y_src.to("cpu"), sigma).to(device)
                        f_pred = module(
                            (style_src, x_src, y_src_sigma),
                            single_eval_pos=single_eval_pos,
                        )
                        inv_perm = np.argsort(sigma)
                        preds.append(permute_y_prob(f_pred, inv_perm))
                    # average all permutations
                    return torch.stack(preds, dim=0).mean(dim=0)

            sigma = sample_non_identity_permutation(
                num_classes, max_num_classes=max_num_classes
            )
            # TypeError: apply_ is only implemented on CPU tensors
            sigma_y_src = permute_y(y_src.to("cpu"), sigma).to(device)
            # sigma_y_src = sigma_y_src.double()

            # compute f(X, σ(Y), X∗)
            #  (n_rows - single_eval_pos, batch_size, n_class)
            prob_output1 = pred_fun(
                module=module,
                style_src=style_src,
                x_src=x_src,
                y_src=sigma_y_src,
                single_eval_pos=single_eval_pos,
                n_ensemble=n_ensemble,
                max_num_classes=max_num_classes,
                num_classes=num_classes,
            )

            # compute σ(f(X, Y, X∗))
            #  (n_rows - single_eval_pos, batch_size, n_class)
            prob_output2 = pred_fun(
                module=module,
                style_src=style_src,
                x_src=x_src,
                y_src=y_src,
                single_eval_pos=single_eval_pos,
                n_ensemble=n_ensemble,
                max_num_classes=max_num_classes,
                num_classes=num_classes,
            )
            prob_output2 = permute_y_prob(prob_output2, sigma)

            # all batch_size tensors
            clf_error = 1 - (
                (prob_output1.argmax(dim=2) == prob_output2.argmax(dim=2))
                .float()
                .mean(axis=(0))
            )
            clf_errors += clf_error.tolist()
            # averages over dataset and dimensions, gets
            # (n_rows - single_eval_pos) list
            l2_logits += torch.sqrt(
                torch.square((prob_output1 - prob_output2)).mean(axis=(0, 2))
            ).tolist()

            # loss_fun = Criterion(
            #     max_num_classes,
            #     logits=True,
            # )
            # loss1 = (
            #     loss_fun(
            #         prob_output1,
            #         sigma_y[single_eval_pos:].long(),
            #     )
            #     .reshape(n_test_rows, batch_size)
            #     .mean(axis=0)
            # )
            # loss2 = (
            #     loss_fun(
            #         prob_output2,
            #         sigma_y[single_eval_pos:].long(),
            #     )
            #     .reshape(n_test_rows, batch_size)
            #     .mean(axis=0)
            # )
            # equi_errors += (loss1 - loss2).square().tolist()
            list_num_classes += [num_classes] * len(clf_error)

    return pd.DataFrame(
        {
            # "loss_difference": equi_errors,
            "clf_errors": clf_errors,
            "l2_logit": l2_logits,
            "num_classes": list_num_classes,
            "num_ensemble": [n_ensemble] * len(list_num_classes),
        }
    )


if __name__ == "__main__":
    classifier = TabPFNClassifier()
    X_train, y_train, X_test = draw_random_examples()
    print(equivariance_error(classifier, X_train, y_train, X_test))
