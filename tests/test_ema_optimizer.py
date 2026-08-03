import copy

import pytest
import torch

from deltagrad.optimizers import DeltaGradEMA, R_TRANSFORMS


def _manual_steps(optimizer, param, grads):
    history = []
    for g in grads:
        param.grad = torch.tensor([g])
        optimizer.step()
        history.append({"p": param.item(), "R": optimizer.state[param]["R"].item()})
    return history


def test_hand_computed_three_step():
    """Cross-checked against an independent pure-Python re-derivation of Sec. 3's
    formula (r_transform="linear", i.e. Option 0, for arithmetic simplicity) --
    this also exercises S_hat's bias correction end-to-end."""
    p = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = DeltaGradEMA([p], lr=0.1, sigma=0.5, beta_phi=0.5, beta_m=0.5,
                             r_transform="linear", A=0.0, B=1.0, epsilon=1e-8)
    history = _manual_steps(optimizer, p, [2.0, 1.0, 3.0])

    expected_p = [0.8, 0.66428571, 0.47418367]
    expected_R = [1.0, 0.90476191, 0.84489796]

    for step, (h, exp_p, exp_r) in enumerate(zip(history, expected_p, expected_R), start=1):
        assert h["p"] == pytest.approx(exp_p, abs=1e-5), f"step {step} theta mismatch"
        assert h["R"] == pytest.approx(exp_r, abs=1e-5), f"step {step} R mismatch"


@pytest.mark.parametrize("r_transform", list(R_TRANSFORMS))
def test_all_six_transforms_bounded(r_transform):
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(6, 5))
    optimizer = DeltaGradEMA([p], lr=0.01, r_transform=r_transform, A=0.1, B=1.0)
    for _ in range(30):
        p.grad = torch.randn_like(p) * 3
        optimizer.step()
        R = optimizer.state[p]["R"]
        assert torch.isfinite(R).all(), f"{r_transform} produced non-finite R"
        assert torch.all(R >= 0.1 - 1e-6)
        assert torch.all(R <= 1.0 + 1e-6)


def test_state_shapes_default_is_3d():
    p = torch.nn.Parameter(torch.randn(3, 4))
    optimizer = DeltaGradEMA([p])  # default r_transform="exp"
    p.grad = torch.randn_like(p)
    optimizer.step()

    state = optimizer.state[p]
    assert set(state) & {"g_tilde", "S", "m"} == {"g_tilde", "S", "m"}
    assert "mu_S" not in state and "var_S" not in state
    core_elements = state["g_tilde"].numel() + state["S"].numel() + state["m"].numel()
    assert core_elements == 3 * p.numel()


def test_state_shapes_zscore_is_5d():
    p = torch.nn.Parameter(torch.randn(3, 4))
    optimizer = DeltaGradEMA([p], r_transform="zscore")
    p.grad = torch.randn_like(p)
    optimizer.step()

    state = optimizer.state[p]
    core_keys = {"g_tilde", "S", "m", "mu_S", "var_S"}
    assert core_keys.issubset(state)
    core_elements = sum(state[k].numel() for k in core_keys)
    assert core_elements == 5 * p.numel()


def test_state_dict_round_trip():
    torch.manual_seed(1)
    p1 = torch.nn.Parameter(torch.randn(4))
    p2 = torch.nn.Parameter(p1.detach().clone())

    opt1 = DeltaGradEMA([p1], lr=0.05)
    opt2 = DeltaGradEMA([p2], lr=0.05)

    grads = [torch.randn(4) for _ in range(4)]
    for g in grads[:2]:
        p1.grad, p2.grad = g.clone(), g.clone()
        opt1.step()
        opt2.step()

    state_dict = copy.deepcopy(opt2.state_dict())
    opt2_reloaded = DeltaGradEMA([p2], lr=0.05)
    opt2_reloaded.load_state_dict(state_dict)

    for g in grads[2:]:
        p1.grad, p2.grad = g.clone(), g.clone()
        opt1.step()
        opt2_reloaded.step()

    assert torch.allclose(p1, p2, atol=1e-6)


def test_convergence_toy_quadratic(toy_quadratic_target):
    torch.manual_seed(2)
    p = torch.nn.Parameter(torch.zeros(3))
    optimizer = DeltaGradEMA([p], lr=0.5)

    initial_loss = ((p - toy_quadratic_target) ** 2).sum().item()
    for _ in range(200):
        optimizer.zero_grad()
        loss = ((p - toy_quadratic_target) ** 2).sum()
        loss.backward()
        optimizer.step()

    final_loss = ((p - toy_quadratic_target) ** 2).sum().item()
    assert final_loss < initial_loss * 0.1


def test_invalid_hyperparameters_raise():
    p = [torch.nn.Parameter(torch.randn(2))]
    with pytest.raises(ValueError):
        DeltaGradEMA(p, lr=-1.0)
    with pytest.raises(ValueError):
        DeltaGradEMA(p, beta_phi=1.0)
    with pytest.raises(ValueError):
        DeltaGradEMA(p, r_transform="not_a_real_transform")
    with pytest.raises(ValueError):
        DeltaGradEMA(p, A=0.9, B=0.1)
