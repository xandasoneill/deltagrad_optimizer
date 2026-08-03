import argparse
import statistics

from scipy.stats import pearsonr

from experiments.configs import CIFAR100_LR_STRESS_BASE, OPTIMIZER_KEYS
from experiments.final_benchmark import run_benchmark, save_results
from experiments._cli import add_smoke_args

LR_MULTIPLIERS = (1, 3, 10)


def main():
    parser = argparse.ArgumentParser(
        description="CIFAR-100 5-CNN learning-rate stress test: sweeps LR multipliers "
                    "{1x,3x,10x} over the base LR x 5 seeds, per deltagradpaperplan.pdf Sec 4.1.")
    parser.add_argument("--optimizer", required=True, choices=OPTIMIZER_KEYS)
    add_smoke_args(parser)
    args = parser.parse_args()

    config = CIFAR100_LR_STRESS_BASE
    base_kwargs = config.optimizer_kwargs_for(args.optimizer)
    base_lr = base_kwargs.get("lr")
    if base_lr is None:
        raise ValueError(f"No base lr configured for optimizer '{args.optimizer}'")

    n_runs = args.n_runs if args.n_runs is not None else (1 if args.smoke else 5)

    summary = {}
    for multiplier in LR_MULTIPLIERS:
        override = {**base_kwargs, "lr": base_lr * multiplier}
        results = run_benchmark(config, args.optimizer, n_runs=n_runs, smoke=args.smoke,
                                 optimizer_kwargs_override=override)
        results["lr_multiplier"] = multiplier
        results["task"] = f"cifar100_lr_stress_{multiplier}x"
        path = save_results(results)

        final_accs = [h[-1] for h in results["acc_history"]]
        std_dev = statistics.pstdev(final_accs) if len(final_accs) > 1 else 0.0

        r_flat = [v for run in results["r_history"] for v in run]
        var_flat = [v for run in results["variance_history"] for v in run]
        corr = pearsonr(r_flat, var_flat)[0] if len(r_flat) > 1 else float("nan")

        summary[multiplier] = {"final_accs": final_accs, "std_dev": std_dev,
                                "r_vs_variance_corr": corr, "path": path}
        print(f"[{multiplier}x lr={base_lr * multiplier:.5g}] final accs={final_accs} "
              f"std_dev={std_dev:.3f} R-vs-variance corr={corr:.3f}")

    print("\nSummary (validation-accuracy std-dev across 1x/3x/10x, per deltagradpaperplan.pdf Sec 4.1):")
    for multiplier, info in summary.items():
        print(f"  {multiplier}x: std_dev={info['std_dev']:.3f}  corr={info['r_vs_variance_corr']:.3f}  "
              f"(saved to {info['path']})")


if __name__ == "__main__":
    main()
