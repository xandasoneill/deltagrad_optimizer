import argparse

from experiments.configs import MNIST_LOGREG
from experiments.final_benchmark import build_optimizer
from deltagrad.training import train_classifier

BATCH_SIZES = (16, 128, 512)


def measure_wallclock(config, optimizer_key, batch_size, epochs, subset_size):
    loader_kwargs = dict(config.loader_kwargs)
    loader_kwargs["subset_size"] = subset_size
    train_loader, test_loader = config.loader_fn(batch_size=batch_size, **loader_kwargs)
    model = config.model_cls(**config.model_kwargs)
    kwargs = config.optimizer_kwargs_for(optimizer_key)
    optimizer = build_optimizer(optimizer_key, model.parameters(), **kwargs)
    result = train_classifier(model, optimizer, optimizer_key, train_loader, test_loader, epochs=epochs)
    return result["total_net_time"] / epochs


def main():
    parser = argparse.ArgumentParser(
        description="Wall-clock per-epoch overhead ablation across batch sizes (Sec 4.2). "
                    "Smoke mode only checks the pipeline runs and returns sane shapes -- "
                    "a hard <0.5%% assertion here would be flaky on shared/CPU hardware "
                    "regardless of code correctness; verify the real claim from a "
                    "full-scale (Colab) run.")
    parser.add_argument("--optimizers", nargs="+", default=["windowed", "ema", "adam"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = MNIST_LOGREG
    subset_size = 256 if args.smoke else None
    epochs = 1 if args.smoke else args.epochs
    batch_sizes = (8,) if args.smoke else BATCH_SIZES

    header = f"{'batch':<8}" + "".join(f"{opt:<14}" for opt in args.optimizers)
    print(header)
    for batch_size in batch_sizes:
        times = {opt: measure_wallclock(config, opt, batch_size, epochs, subset_size)
                 for opt in args.optimizers}
        row = f"{batch_size:<8}" + "".join(f"{times[opt]:<14.4f}" for opt in args.optimizers)
        if "adam" in times and times["adam"] > 0:
            overhead = {opt: (times[opt] / times["adam"] - 1) * 100
                        for opt in args.optimizers if opt != "adam"}
            row += "  " + ", ".join(f"{opt}:{v:+.2f}%" for opt, v in overhead.items())
        print(row)


if __name__ == "__main__":
    main()
