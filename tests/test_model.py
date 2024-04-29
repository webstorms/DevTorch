import pytest
import torch
import torch.nn as nn

from devtorch import DevModel


class Model(DevModel):

    def __init__(self, init_type, **kwargs):
        super().__init__()
        self.tensor = nn.Parameter(torch.rand(10, 10))
        self.init_weight(self.tensor, init_type, **kwargs)


parameter_sets = [
    ("constant", {"val": 0.5}),
    ("uniform", {"a": -1, "b": 1}),
    ("glorot_uniform", {}),
    ("normal", {"mean": 0, "std": 1}),
    ("glorot_normal", {}),
    ("identity", {"scale": 1}),
    ("identity", {"scale": 0.5}),
]


@pytest.mark.parametrize("init_type, kwargs", parameter_sets)
def test_all_inits(init_type, kwargs):
    model = Model(init_type, **kwargs)
    assert model.hyperparams["weights"]["tensor"]["init_type"] == init_type


def test_for_duplicate():
    model = Model("constant", val=0.5)

    with pytest.raises(ValueError):
        model.init_weight(model.tensor, "constant", val=0.5)
