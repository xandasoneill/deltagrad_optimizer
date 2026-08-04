"""Runs one task with each of DeltaGradEMA's 6 Sec. 3.2 R-transforms.

Each transform is saved as its own "optimizer" (`ema_exp`, `ema_sigmoid`, ...)
under the task's results directory, so every comparison in
notebooks/analyze_results.ipynb -- leaderboard, learning curves, gradient
variance, seed stability, significance -- treats them as competitors and works
on them unchanged. Section 7 of that notebook additionally plots each
transform's sampled (S_hat, R) operating points over its analytic curve, which
is what `--sample-transform-every` records.

    python -m experiments.sweep_r_transforms --task mnist_logreg
    python -m experiments.sweep_r_transforms --task cifar100_noise_20 --epochs 20
    python -m experiments.sweep_r_transforms --task mnist_logreg --transform exp sigmoid
"""

import argparse
from dataclasses import replace

import torch

from deltagrad.optimizers import R_TRANSFORMS

from experiments._cli import add_smoke_args
from experiments.configs import TASK_REGISTRY
from experiments.final_benchmark import run_benchmark, save_results


def main():
    parser = argparse.ArgumentParser(
        description="Compare DeltaGradEMA's 6 R-transform options (deltagradpaperplan.pdf "
                    "Sec. 3.2) on one task.")
    parser.add_argument("--task", default="mnist_logreg", choices=sorted(TASK_REGISTRY),
                        help="Default mnist_logreg: cheap enough to sweep 6 times over.")
    parser.add_argument("--transform", nargs="+", default=list(R_TRANSFORMS),
                        choices=list(R_TRANSFORMS),
                        help="Transforms to run (default: all 6).")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override the task's epoch count -- this is a diagnostic "
                             "sweep, so bounding its compute is usually worth it.")
    parser.add_argument("--sample-transform-every", type=int, default=50, metavar="N",
                        help="Record (S_hat, R) pairs every N optimizer steps for the "
                             "notebook's curve overlay (default: 50; 0 disables).")
    parser.add_argument("--grad-variance-every", type=int, default=10, metavar="N",
                        help="Measure gradient variance every N batches. Each measurement "
                             "costs 8 extra fwd/bwd passes, which dominates this sweep's "
                             "runtime on cheap tasks -- raise it to go faster (default: 10).")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    add_smoke_args(parser)
    args = parser.parse_args()

    config = TASK_REGISTRY[args.task]
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    base_kwargs = config.optimizer_kwargs_for("ema")
    n_runs = args.n_runs if args.n_runs is not None else (1 if args.smoke else 3)

    print(f"R-transform sweep on {args.task} | {device} | {n_runs} runs x "
          f"{config.effective_epochs(args.smoke)} epochs\n")
    print(f"{'transform':<12}{'option':<9}{'final metric (mean)':<22}{'mean R'}")

    for transform in args.transform:
        option_number, _ = R_TRANSFORMS[transform]
        results = run_benchmark(
            config, "ema", n_runs=n_runs, smoke=args.smoke, device=device,
            optimizer_kwargs_override={**base_kwargs, "r_transform": transform},
            grad_variance_every=args.grad_variance_every,
            sample_transform_every=args.sample_transform_every or None)
        # Saved as a distinct "optimizer" so the transforms sit side by side in
        # one task directory and every notebook comparison picks them up.
        results["optimizer"] = f"ema_{transform}"
        path = save_results(results)

        finals = [history[-1] for history in results["acc_history"]]
        mean_final = sum(finals) / len(finals)
        r_finals = [r[-1] for r in results["r_history"] if r]
        mean_r = sum(r_finals) / len(r_finals) if r_finals else float("nan")
        print(f"{transform:<12}{option_number:<9}{mean_final:<22.3f}{mean_r:.4f}"
              f"   -> {path}")


if __name__ == "__main__":
    main()
