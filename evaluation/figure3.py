import numpy as np

from equitabpfn.utils import figure_path
import pandas as pd
import seaborn as sns
from pathlib import Path

from matplotlib import pyplot as plt

data_dir = Path(__file__).parent / "data" / "figure3"

def load_df():
    dfs = []
    for file in data_dir.rglob("equivariance-errors-*.csv"):
        df = pd.read_csv(file)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


df = load_df()

# TODO fix
# df = df[df.num_classes < 10]

metric = "clf_error"
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4, 4.2))
axes = np.ravel(axes)
df["Number of gradient iterations"] = df["epoch"] * 384
ax = sns.barplot(
    df[(df.model == "tabpfn") & (df.num_ensemble == 1)],
    x="Number of gradient iterations",
    y=metric,
    ax=axes[0],
)
ax.set_yscale("log")
# ax.set_xlabel("Number of gradient iterations")
ax.set_ylabel("Equivariance error")

df["# ensemble"] = df["num_ensemble"]
ax = sns.barplot(
    df[(df.model == "tabpfn") & (df.epoch == 1200)],
    x="num_classes",
    y=metric,
    hue="# ensemble",
    ax=axes[1],
    estimator="mean",
)
ax.set_xlabel("Number of classes")
ax.set_ylabel("Equivariance error")
# ax.set_ylabel(None)
plt.tight_layout()
plt.savefig(figure_path() / "equivariance-error.pdf")
plt.show()
