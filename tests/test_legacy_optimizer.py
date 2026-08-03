import glob

import pytest
import joblib
import torch

from deltagrad.optimizers import DeltaGradWindowedLegacy

_BEST_PARAMS_FIXTURES = sorted(glob.glob("best_params/best_params_DeltaGrad_fixed_*.pkl"))


@pytest.mark.skipif(not _BEST_PARAMS_FIXTURES, reason="no legacy best_params/*.pkl fixtures found")
@pytest.mark.parametrize("path", _BEST_PARAMS_FIXTURES)
def test_loads_real_best_params_pkl(path):
    """DeltaGradWindowedLegacy must keep accepting the old gamma-shaped best_params
    schema (lr, gamma, alpha, beta, K, smoothing, batch_size) via the documented
    lr *= gamma / pop workaround used by experiments/final_benchmark.py."""
    params = dict(joblib.load(path))
    params["lr"] = params["lr"] * params.pop("gamma")
    params.pop("batch_size", None)

    model = torch.nn.Linear(4, 3)
    optimizer = DeltaGradWindowedLegacy(model.parameters(), **params)

    inputs = torch.randn(2, 4)
    loss = model(inputs).sum()
    loss.backward()
    optimizer.step()  # must not raise


def test_regression_guard():
    """Freezes the legacy optimizer's exact (quirky, paper-deviating) output for a
    fixed input sequence, so a future refactor can't silently 'fix' behavior that's
    preserved here specifically for old-result reproducibility."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    optimizer = DeltaGradWindowedLegacy([p], lr=0.1, K=3, alpha=0.3, beta=0.7, smoothing=0.6)
    grads = [
        torch.tensor([2.0, -1.0]),
        torch.tensor([1.0, 0.5]),
        torch.tensor([-3.0, 2.0]),
        torch.tensor([0.5, -0.5]),
        torch.tensor([4.0, 1.0]),
    ]

    expected_p = [
        [0.80000001, -0.89999998],
        [0.62222224, -0.84285712],
        [0.56045753, -0.82021004],
        [0.51416439, -0.81381476],
        [0.48774436, -0.82739741],
    ]
    expected_R = [
        [1.0, 1.0],
        [0.88888890, 0.57142866],
        [0.35000002, 0.35000002],
        [0.54506421, 0.66693491],
        [0.63761622, 0.85812283],
    ]

    for step, (g, exp_p, exp_r) in enumerate(zip(grads, expected_p, expected_R), start=1):
        p.grad = g.clone()
        optimizer.step()
        assert p.detach().tolist() == pytest.approx(exp_p, abs=1e-5), f"step {step} theta regressed"
        assert optimizer.state[p]["R"].tolist() == pytest.approx(exp_r, abs=1e-5), f"step {step} R regressed"
