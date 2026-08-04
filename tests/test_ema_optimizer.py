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
    with pytest.raises(ValueError):
        DeltaGradEMA(p, sample_every=0)
    with pytest.raises(ValueError):
        DeltaGradEMA(p, sample_size=0)


# --------------------------------------------------------------------------
# R-transform sampling (diagnostics for notebooks/analyze_results.ipynb Sec. 7)
# --------------------------------------------------------------------------


def test_sampling_off_by_default():
    p = torch.nn.Parameter(torch.randn(4))
    optimizer = DeltaGradEMA([p])
    for _ in range(5):
        p.grad = torch.randn_like(p)
        optimizer.step()
    assert optimizer.transform_samples == []


def test_sampling_fires_on_schedule():
    p = torch.nn.Parameter(torch.randn(10, 10))
    optimizer = DeltaGradEMA([p], sample_every=3, sample_size=16)
    for _ in range(10):
        p.grad = torch.randn_like(p)
        optimizer.step()

    assert [c["step"] for c in optimizer.transform_samples] == [3, 6, 9]
    for capture in optimizer.transform_samples:
        assert capture["S_hat"].shape == capture["R"].shape == capture["param_index"].shape
        assert capture["S_hat"].size <= 16


def test_sampled_pairs_lie_on_the_clamped_transform():
    """The whole point of the plot: a sampled R must be what the analytic curve
    (post-clamp) says for its sampled S_hat, or the overlay would be a lie."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8, 8))
    optimizer = DeltaGradEMA([p], r_transform="exp", gamma=2.0, A=0.1, B=1.0,
                             sample_every=1, sample_size=64)
    for _ in range(6):
        p.grad = torch.randn_like(p) * 5
        optimizer.step()

    for capture in optimizer.transform_samples:
        expected = torch.exp(-2.0 * torch.from_numpy(capture["S_hat"])).clamp(0.1, 1.0)
        assert torch.allclose(expected, torch.from_numpy(capture["R"]), atol=1e-5)


@pytest.mark.parametrize("r_transform", list(R_TRANSFORMS))
def test_sampling_does_not_perturb_training(r_transform):
    """Sampling is a diagnostic, so an identically-seeded run must land on exactly
    the same parameters with it on as with it off -- including consuming no global
    RNG (hence the torch.randn draws interleaved between steps, which would
    desynchronise if the sampler drew from the global stream)."""
    def run(sample_every):
        torch.manual_seed(7)
        p = torch.nn.Parameter(torch.randn(6, 5))
        optimizer = DeltaGradEMA([p], lr=0.05, r_transform=r_transform,
                                 sample_every=sample_every, sample_size=32)
        for _ in range(12):
            p.grad = torch.randn_like(p) * 2
            optimizer.step()
        return p.detach().clone()

    assert torch.equal(run(None), run(2)), f"{r_transform}: sampling changed the trajectory"


def test_transform_spec_round_trips_shape_parameters():
    p = torch.nn.Parameter(torch.randn(3))
    optimizer = DeltaGradEMA([p], r_transform="sigmoid", tau=0.3, s=0.05, A=0.2, B=0.9)
    spec = optimizer.transform_spec()
    assert spec["r_transform"] == "sigmoid"
    assert spec["tau"] == 0.3 and spec["s"] == 0.05
    assert spec["A"] == 0.2 and spec["B"] == 0.9


def test_samples_stay_out_of_state_dict():
    """Diagnostics must not ride along into checkpoints, and must not count
    toward the Sec. 4.2 3d state-memory footprint."""
    p = torch.nn.Parameter(torch.randn(5))
    optimizer = DeltaGradEMA([p], sample_every=1)
    p.grad = torch.randn_like(p)
    optimizer.step()

    assert optimizer.transform_samples, "expected a capture to have happened"
    assert "sample_every" not in optimizer.param_groups[0]
    assert not any(key.startswith("sample") or key == "S_hat"
                   for key in optimizer.state[p])
    assert "transform_samples" not in optimizer.state_dict()
