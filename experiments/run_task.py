import argparse

from experiments.configs import TASK_REGISTRY, OPTIMIZER_KEYS
from experiments.final_benchmark import run_benchmark, save_results, load_tuned_kwargs
from experiments._cli import add_smoke_args


def main():
    parser = argparse.ArgumentParser(
        description="Run one deltagradpaperplan.pdf core benchmark task against one optimizer.")
    parser.add_argument("--task", required=True, choices=sorted(TASK_REGISTRY))
    parser.add_argument("--optimizer", required=True, choices=OPTIMIZER_KEYS)
    parser.add_argument("--use-tuned", action="store_true",
                        help="Use the kwargs from best_params/{task}/{optimizer}.pkl "
                             "(written by experiments.tune_hyperparams) instead of the "
                             "config's untuned defaults.")
    add_smoke_args(parser)
    args = parser.parse_args()

    config = TASK_REGISTRY[args.task]

    override = None
    if args.use_tuned:
        override = load_tuned_kwargs(args.task, args.optimizer)
        # Hard error rather than a silent fall back to defaults: an untuned run
        # quietly labelled as tuned would poison the comparison it feeds.
        if override is None:
            parser.error(f"No tuned params at best_params/{args.task}/{args.optimizer}.pkl -- run "
                         f"`python -m experiments.tune_hyperparams --task {args.task} "
                         f"--optimizer {args.optimizer}` first, or drop --use-tuned.")
        print(f"Using tuned hyperparameters: {override}")

    results = run_benchmark(config, args.optimizer, n_runs=args.n_runs, smoke=args.smoke,
                             optimizer_kwargs_override=override)
    path = save_results(results)

    print(f"Saved results to {path}")
    print(f"Final-epoch metric per run: {[h[-1] for h in results['acc_history']]}")


if __name__ == "__main__":
    main()
