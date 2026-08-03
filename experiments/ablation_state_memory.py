import argparse

import torch

from deltagrad.optimizers import DeltaGradWindowed, DeltaGradWindowedLegacy, DeltaGradEMA, make_baseline_optimizer

# Per-optimizer allowlist of "core" state keys -- the ones that scale with parameter
# count d and count toward deltagradpaperplan.pdf Sec 4.2's state-memory claims.
# Excludes bookkeeping (`step`, `history_count`) and diagnostic-only (`R`) entries
# that aren't part of the paper's stated footprint -- note even torch.optim.Adam's
# own `step` is a 0-dim tensor in this torch version, not a python int, so a naive
# "count every tensor" approach would over-count the baselines too.
CORE_STATE_KEYS = {
    "windowed": {"g_s", "history"},
    "windowed_legacy": {"smooth_grad", "history_buffer"},
    "ema": {"g_tilde", "S", "m", "mu_S", "var_S"},  # mu_S/var_S only populated for r_transform="zscore"
    "adam": {"exp_avg", "exp_avg_sq"},
    "adamw": {"exp_avg", "exp_avg_sq"},
    "sgd_momentum": {"momentum_buffer"},
    "adagrad": {"sum"},
    "rmsprop": {"square_avg"},
}


def _core_elements_per_param(optimizer, family, num_params):
    allowed = CORE_STATE_KEYS[family]
    total = 0
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state.get(p, {})
            for key in allowed:
                if key in state and torch.is_tensor(state[key]):
                    total += state[key].numel()
    return total / num_params


def measure_footprint(build_fn, family, num_steps=3, d=97):
    """d=97 is an arbitrary prime parameter count, chosen so shape-broadcasting
    bugs can't hide behind a suspiciously round number."""
    p = torch.nn.Parameter(torch.randn(d))
    optimizer = build_fn([p])
    for _ in range(num_steps):
        optimizer.zero_grad()
        (p ** 2).sum().backward()
        optimizer.step()
    return _core_elements_per_param(optimizer, family, d)


def main():
    K = 4
    configs = [
        ("windowed",        lambda params: DeltaGradWindowed(params, K=K),                 "windowed",        K + 1),
        ("windowed_legacy", lambda params: DeltaGradWindowedLegacy(params, K=K),            "windowed_legacy", K + 1),
        ("ema (default)",   lambda params: DeltaGradEMA(params),                            "ema",             3),
        ("ema (zscore)",    lambda params: DeltaGradEMA(params, r_transform="zscore"),       "ema",             5),
        ("adam",            lambda params: make_baseline_optimizer("adam", params),         "adam",            2),
        ("adamw",           lambda params: make_baseline_optimizer("adamw", params),         "adamw",           2),
        ("sgd_momentum",    lambda params: make_baseline_optimizer("sgd_momentum", params),  "sgd_momentum",    1),
        ("adagrad",         lambda params: make_baseline_optimizer("adagrad", params),       "adagrad",         1),
        ("rmsprop",         lambda params: make_baseline_optimizer("rmsprop", params),       "rmsprop",         1),
    ]

    print(f"{'optimizer':<20}{'measured (xd)':<16}{'expected (xd)':<16}{'match'}")
    all_match = True
    for label, build_fn, family, expected in configs:
        measured = measure_footprint(build_fn, family)
        match = abs(measured - expected) < 1e-6
        all_match = all_match and match
        print(f"{label:<20}{measured:<16.3f}{expected:<16}{'OK' if match else 'MISMATCH'}")

    if not all_match:
        raise SystemExit("State-memory footprint did not match deltagradpaperplan.pdf Sec 4.2's claims.")
    print("\nAll state-memory footprints match deltagradpaperplan.pdf Sec 4.2's claims "
          "((K+1)d windowed, 3d EMA [5d for zscore], 2d Adam-family, 1d SGD/Adagrad/RMSProp).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="State memory footprint ablation (Sec 4.2).")
    parser.add_argument("--smoke", action="store_true",
                         help="No-op here (this ablation is already instant) -- kept for CLI consistency.")
    parser.parse_args()
    main()
