import pytest
import torch
import torch.nn as nn

from deltagrad.training import get_grad_variance, _measure_grad_variance_and_R


def test_get_grad_variance_returns_nonnegative_float():
    torch.manual_seed(0)
    model = nn.Linear(6, 3)
    criterion = nn.CrossEntropyLoss()
    inputs = torch.randn(8, 6)
    labels = torch.randint(0, 3, (8,))

    variance = get_grad_variance(model, criterion, inputs, labels, num_samples=4)
    assert isinstance(variance, float)
    assert variance >= 0.0


class _FakeOptimizer:
    """Minimal stand-in exposing just enough of the Optimizer interface for
    _measure_grad_variance_and_R's R-averaging logic."""

    def __init__(self, params, r_value=None):
        params = list(params)
        self.param_groups = [{"params": params}]
        self.state = {p: ({"R": torch.full_like(p, r_value)} if r_value is not None else {})
                      for p in params}


def test_measure_wrapper_restores_original_grads():
    torch.manual_seed(0)
    model = nn.Linear(6, 3)
    criterion = nn.CrossEntropyLoss()
    inputs = torch.randn(8, 6)
    labels = torch.randint(0, 3, (8,))

    model.zero_grad()
    criterion(model(inputs), labels).backward()
    real_grads = [p.grad.clone() for p in model.parameters()]

    optimizer = _FakeOptimizer(model.parameters())
    variance, r_value = _measure_grad_variance_and_R(model, optimizer, criterion, inputs, labels)

    assert isinstance(variance, float) and variance >= 0.0
    assert r_value is None  # no 'R' tracked by this fake optimizer

    restored = [p.grad for p in model.parameters()]
    assert all(torch.allclose(a, b) for a, b in zip(real_grads, restored))


def test_measure_wrapper_averages_R_across_params():
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    criterion = nn.CrossEntropyLoss()
    inputs = torch.randn(5, 4)
    labels = torch.randint(0, 2, (5,))
    model.zero_grad()
    criterion(model(inputs), labels).backward()

    optimizer = _FakeOptimizer(model.parameters(), r_value=0.5)
    _, r_value = _measure_grad_variance_and_R(model, optimizer, criterion, inputs, labels)
    assert r_value == pytest.approx(0.5)
