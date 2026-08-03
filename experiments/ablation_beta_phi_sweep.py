import argparse
from dataclasses import replace

from experiments.configs import MNIST_MLP_DROPOUT
from experiments.final_benchmark import run_benchmark, save_results

BETA_PHI_VALUES = (0.70, 0.80, 0.90, 0.95)


def main():
    parser = argparse.ArgumentParser(
        description="beta_phi decay sensitivity sweep (Sec 4.2), run on MNIST MLP "
                    "(cheap + non-convex) rather than 50-epoch CIFAR-100, to keep "
                    "the ablation's compute bounded.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n-runs", type=int, default=None)
    args = parser.parse_args()

    config = MNIST_MLP_DROPOUT
    if not args.smoke:
        # This sweep is a diagnostic, not a full 200-epoch run -- the paper doesn't
        # specify a duration, so cap epochs to bound total sweep compute.
        config = replace(config, epochs=20)

    base_kwargs = config.optimizer_kwargs_for("ema")
    n_runs = args.n_runs if args.n_runs is not None else (1 if args.smoke else 3)

    print(f"{'beta_phi':<10}{'K_eff':<8}{'final_acc (mean)':<18}{'final_R (mean)'}")
    for beta_phi in BETA_PHI_VALUES:
        override = {**base_kwargs, "beta_phi": beta_phi}
        results = run_benchmark(config, "ema", n_runs=n_runs, smoke=args.smoke,
                                 optimizer_kwargs_override=override)
        results["task"] = f"beta_phi_sweep_{beta_phi}"
        path = save_results(results)

        final_accs = [h[-1] for h in results["acc_history"]]
        mean_acc = sum(final_accs) / len(final_accs)
        final_rs = [r[-1] for r in results["r_history"] if r]
        mean_r = sum(final_rs) / len(final_rs) if final_rs else float("nan")
        k_eff = 1 / (1 - beta_phi)

        print(f"{beta_phi:<10}{k_eff:<8.1f}{mean_acc:<18.3f}{mean_r:<14.3f}(saved to {path})")


if __name__ == "__main__":
    main()
