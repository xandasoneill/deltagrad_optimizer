import pytest
import torch

from deltagrad.optimizers import make_baseline_optimizer
from deltagrad.optimizers.baselines import _BASELINE_REGISTRY


@pytest.mark.parametrize("name", list(_BASELINE_REGISTRY))
def test_constructs_and_steps(name):
    p = torch.nn.Parameter(torch.randn(3, 2))
    optimizer = make_baseline_optimizer(name, [p])
    p.grad = torch.randn_like(p)
    optimizer.step()  # must not raise


def test_case_insensitive_name():
    p = torch.nn.Parameter(torch.randn(2))
    optimizer = make_baseline_optimizer("Adam", [p])
    assert isinstance(optimizer, torch.optim.Adam)


def test_overrides_take_precedence_over_defaults():
    p = torch.nn.Parameter(torch.randn(2))
    optimizer = make_baseline_optimizer("adam", [p], lr=0.5)
    assert optimizer.param_groups[0]["lr"] == 0.5


def test_unknown_name_raises():
    p = torch.nn.Parameter(torch.randn(2))
    with pytest.raises(ValueError):
        make_baseline_optimizer("not_a_real_optimizer", [p])
