import pytest
import torch

from equitabpfn.models.encoders import OneHotAndLinear
from equitabpfn.models.ktabpfn_model import equitabpfn
from equitabpfn.models.biattention_ktabpfn import BiAttentionTabPFN

batch_size = 5
n_rows = 3
n_features = 7
num_classes = 2
single_eval_pos = 3
emsize = 6


@pytest.mark.parametrize(
    "model_cls",
    [equitabpfn,
     #BiAttentionTabPFN
     ],
)
def test_forward_shape(model_cls):
    x = torch.rand(batch_size, n_rows, n_features)
    y = torch.randint(low=0, high=num_classes, size=(batch_size, n_rows))
    model = model_cls(
        n_out=num_classes,
        emsize=emsize,
        nhead=2,
        nhid_factor=12,
        nlayers=2,
        n_features=n_features,
        y_encoder_layer=OneHotAndLinear(emsize=emsize, num_classes=num_classes),
    )
    pred = model.forward(src=(x, y), single_eval_pos=single_eval_pos)
    assert pred.shape == (batch_size - single_eval_pos, n_rows, num_classes)
