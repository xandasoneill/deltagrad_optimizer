import argparse

import torch

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
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"],
                        help="Default: cuda when available, else cpu.")
    parser.add_argument("--grad-variance-every", type=int, default=10,
                        help="Measure gradient variance (8 extra fwd/bwd passes on a "
                             "subsample) every N batches; raise to cut instrumentation "
                             "overhead, e.g. --grad-variance-every 50 (default: 10).")
    parser.add_argument("--sample-transform-every", type=int, default=None, metavar="N",
                        help="Record (S_hat, R) pairs every N optimizer steps so the "
                             "R-transform's operating range can be plotted against its "
                             "analytic curve (--optimizer ema only; off by default).")
    add_smoke_args(parser)
    args = parser.parse_args()

    config = TASK_REGISTRY[args.task]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    if args.sample_transform_every and args.optimizer != "ema":
        parser.error("--sample-transform-every only applies to --optimizer ema "
                     "(it is the only optimizer with an R-transform to sample).")

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
                             optimizer_kwargs_override=override, device=device,
                             grad_variance_every=args.grad_variance_every,
                             sample_transform_every=args.sample_transform_every)
    path = save_results(results)

    print(f"Saved results to {path}")
    print(f"Final-epoch metric per run: {[h[-1] for h in results['acc_history']]}")


if __name__ == "__main__":
    main()
