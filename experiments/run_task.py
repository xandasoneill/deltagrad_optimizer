import argparse

from experiments.configs import TASK_REGISTRY, OPTIMIZER_KEYS
from experiments.final_benchmark import run_benchmark, save_results
from experiments._cli import add_smoke_args


def main():
    parser = argparse.ArgumentParser(
        description="Run one deltagradpaperplan.pdf core benchmark task against one optimizer.")
    parser.add_argument("--task", required=True, choices=sorted(TASK_REGISTRY))
    parser.add_argument("--optimizer", required=True, choices=OPTIMIZER_KEYS)
    add_smoke_args(parser)
    args = parser.parse_args()

    config = TASK_REGISTRY[args.task]
    results = run_benchmark(config, args.optimizer, n_runs=args.n_runs, smoke=args.smoke)
    path = save_results(results)

    print(f"Saved results to {path}")
    print(f"Final-epoch metric per run: {[h[-1] for h in results['acc_history']]}")


if __name__ == "__main__":
    main()
