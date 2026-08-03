import copy

import pytest
import torch

from deltagrad.optimizers import DeltaGradWindowed


def _manual_steps(optimizer, param, grads):
    history = []
    for g in grads:
        param.grad = torch.tensor([g])
        optimizer.step()
        history.append({"p": param.item(), "R": optimizer.state[param]["R"].item()})
    return history


def test_hand_computed_three_step():
    """Cross-checked against an independent pure-Python re-derivation of Sec. 2's formula."""
    p = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = DeltaGradWindowed([p], lr=0.1, K=2, alpha=0.5, sigma=0.5, A=0.0, B=1.0, epsilon=1e-8)
    history = _manual_steps(optimizer, p, [2.0, 1.0, 3.0])

    expected_p = [0.8, 0.65535714, 0.44326155]
    expected_R = [1.0, 0.96428571, 0.94264706]

    for step, (h, exp_p, exp_r) in enumerate(zip(history, expected_p, expected_R), start=1):
        assert h["p"] == pytest.approx(exp_p, abs=1e-5), f"step {step} theta mismatch"
        assert h["R"] == pytest.approx(exp_r, abs=1e-5), f"step {step} R mismatch"


@pytest.mark.parametrize("K,alpha,sigma,A,B", [
    (4, 0.1, 0.9, 0.1, 1.0),
    (2, 0.5, 0.5, 0.2, 0.8),
    (8, 0.9, 0.1, 0.0, 1.0),
])
def test_R_bounds_stay_in_AB(K, alpha, sigma, A, B):
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(6, 5))
    optimizer = DeltaGradWindowed([p], lr=0.01, K=K, alpha=alpha, sigma=sigma, A=A, B=B)
    for _ in range(30):
        p.grad = torch.randn_like(p) * 3  # occasional large grads stress-test the clamp
        optimizer.step()
        R = optimizer.state[p]["R"]
        assert torch.all(R >= A - 1e-6)
        assert torch.all(R <= B + 1e-6)


def test_state_shapes_match_k_plus_1_d():
    K = 5
    p = torch.nn.Parameter(torch.randn(3, 4))
    optimizer = DeltaGradWindowed([p], K=K)
    p.grad = torch.randn_like(p)
    optimizer.step()

    state = optimizer.state[p]
    assert state["g_s"].shape == p.shape
    assert state["history"].shape == (K,) + p.shape

    core_elements = state["g_s"].numel() + state["history"].numel()
    assert core_elements == (K + 1) * p.numel()


def test_history_ages_stay_aligned_past_k_steps():
    """Regression guard for the legacy fixed-slice-index bug: after step > K, the
    oldest value must be evicted and ages must stay in order (index 0 = newest)."""
    K = 3
    p = torch.nn.Parameter(torch.tensor([0.0]))
    optimizer = DeltaGradWindowed([p], lr=0.0, K=K, sigma=0.0)  # sigma=0 -> g_s == grad exactly
    grad_sequence = [1.0, 2.0, 3.0, 4.0, 5.0]
    for g in grad_sequence:
        p.grad = torch.tensor([g])
        optimizer.step()

    state = optimizer.state[p]
    # After 5 steps with K=3, the most recent 3 g_s values (3,4,5) should be held,
    # oldest-first eviction of (1,2); index 0 = newest.
    assert state["history"].flatten().tolist() == pytest.approx([5.0, 4.0, 3.0])
    assert state["history_count"] == K


def test_state_dict_round_trip():
    torch.manual_seed(1)
    p1 = torch.nn.Parameter(torch.randn(4))
    p2 = torch.nn.Parameter(p1.detach().clone())

    opt1 = DeltaGradWindowed([p1], lr=0.05, K=3)
    opt2 = DeltaGradWindowed([p2], lr=0.05, K=3)

    grads = [torch.randn(4) for _ in range(4)]
    for g in grads[:2]:
        p1.grad, p2.grad = g.clone(), g.clone()
        opt1.step()
        opt2.step()

    state_dict = copy.deepcopy(opt2.state_dict())
    opt2_reloaded = DeltaGradWindowed([p2], lr=0.05, K=3)
    opt2_reloaded.load_state_dict(state_dict)

    for g in grads[2:]:
        p1.grad, p2.grad = g.clone(), g.clone()
        opt1.step()
        opt2_reloaded.step()

    assert torch.allclose(p1, p2, atol=1e-6)


def test_convergence_toy_quadratic(toy_quadratic_target):
    torch.manual_seed(2)
    p = torch.nn.Parameter(torch.zeros(3))
    optimizer = DeltaGradWindowed([p], lr=0.3, K=4)

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
        DeltaGradWindowed(p, lr=-1.0)
    with pytest.raises(ValueError):
        DeltaGradWindowed(p, K=0)
    with pytest.raises(ValueError):
        DeltaGradWindowed(p, A=0.9, B=0.1)
