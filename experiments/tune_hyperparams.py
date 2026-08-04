"""Optuna hyperparameter tuning for any optimizer in OPTIMIZER_KEYS against any
task in TASK_REGISTRY.

Scoring always happens on a validation split carved out of the *training* set --
the task's test loader is discarded here and never evaluated, so tuned
hyperparameters cannot leak test-set information into the benchmark that later
reports on them.

Results land in `best_params/{task}/{optimizer}.pkl`, which
`experiments/run_task.py --use-tuned` reads back.

    python -m experiments.tune_hyperparams --task cifar100_noise_20            # all 7 optimizers
    python -m experiments.tune_hyperparams --task mnist_vae --optimizer ema adam
    python -m experiments.tune_hyperparams --task cifar10_conv --optimizer ema --fixed r_transform=sigmoid
"""

import argparse
import ast
import os

import joblib
import optuna
import torch
from torch.utils.data import DataLoader, random_split

from deltagrad.training import train_classifier, train_vae

from experiments._cli import set_seed
from experiments.configs import TASK_REGISTRY, OPTIMIZER_KEYS
from experiments.final_benchmark import build_optimizer

# optimizer key -> {kwarg: spec}, where spec is ("float", low, high, log) or
# ("int", low, high). Anything an optimizer accepts but that isn't listed here
# keeps its config/class default -- pin it explicitly with --fixed to override.
SEARCH_SPACES = {
    "windowed": {
        "lr":       ("float", 1e-4, 0.5, True),
        "K":        ("int", 2, 8),
        "alpha":    ("float", 0.1, 0.9, False),
        "sigma":    ("float", 0.5, 0.99, False),
    },
    "ema": {
        "lr":       ("float", 1e-4, 0.5, True),
        "sigma":    ("float", 0.5, 0.99, False),
        "beta_phi": ("float", 0.5, 0.999, False),
        "beta_m":   ("float", 0.5, 0.999, False),
    },
    "adam": {
        "lr":       ("float", 1e-5, 1e-1, True),
        "beta1":    ("float", 0.5, 0.99, False),
        "beta2":    ("float", 0.9, 0.9999, False),
    },
    "adamw": {
        "lr":           ("float", 1e-5, 1e-1, True),
        "beta1":        ("float", 0.5, 0.99, False),
        "beta2":        ("float", 0.9, 0.9999, False),
        "weight_decay": ("float", 1e-5, 1e-1, True),
    },
    "sgd_momentum": {
        "lr":       ("float", 1e-4, 0.5, True),
        "momentum": ("float", 0.5, 0.99, False),
    },
    "adagrad": {
        "lr":       ("float", 1e-4, 0.5, True),
        "lr_decay": ("float", 0.0, 1e-2, False),
    },
    "rmsprop": {
        "lr":       ("float", 1e-5, 1e-1, True),
        "alpha":    ("float", 0.8, 0.999, False),
        "momentum": ("float", 0.0, 0.99, False),
    },
}

# DeltaGradEMA's Sec. 3.2 R-transforms each read their own shape parameters, and
# ignore the others -- searching a knob the active transform never reads would
# just waste trials, so only the live ones join the space.
_EMA_TRANSFORM_SPACES = {
    "linear":  {},
    "exp":     {"gamma": ("float", 0.1, 10.0, True)},
    "inverse": {"gamma": ("float", 0.1, 10.0, True)},
    "power":   {"power_p": ("float", 0.1, 3.0, False)},
    "sigmoid": {"tau": ("float", 0.0, 1.0, False), "s": ("float", 0.01, 1.0, True)},
    "zscore":  {"zscore_k": ("float", 0.5, 4.0, False)},
}


def search_space_for(optimizer_key, base_kwargs):
    """Tunable kwargs for `optimizer_key`, given the non-tuned kwargs it starts from
    (which decide the EMA transform, and so which shape parameters are live)."""
    space = dict(SEARCH_SPACES[optimizer_key])
    if optimizer_key == "ema":
        transform = base_kwargs.get("r_transform", "exp")
        if transform not in _EMA_TRANSFORM_SPACES:
            raise ValueError(f"Unknown r_transform '{transform}'. "
                             f"Choices: {list(_EMA_TRANSFORM_SPACES)}")
        space.update(_EMA_TRANSFORM_SPACES[transform])
    return space


def _suggest(trial, name, spec):
    kind = spec[0]
    if kind == "float":
        _, low, high, log = spec
        return trial.suggest_float(name, low, high, log=log)
    if kind == "int":
        _, low, high = spec
        return trial.suggest_int(name, low, high)
    raise ValueError(f"Unknown search-space spec kind '{kind}' for '{name}'")


def finalize_kwargs(optimizer_key, kwargs):
    """torch's Adam/AdamW take one `betas` tuple, but Optuna has to search its two
    components as separate scalars -- recombine them into the constructor's shape."""
    kwargs = dict(kwargs)
    if optimizer_key in ("adam", "adamw"):
        beta1 = kwargs.pop("beta1", None)
        beta2 = kwargs.pop("beta2", None)
        if beta1 is not None and beta2 is not None:
            kwargs["betas"] = (beta1, beta2)
    return kwargs


def train_val_loaders(config, batch_size, loader_kwargs, val_fraction, seed):
    """Splits the task's *training* set into train/validation loaders. The test
    loader the task builds is dropped on the floor -- tuning must never see it.

    Returns the loaders plus the shuffle generator, so each trial can re-seed it
    and thus see an identical batch sequence (otherwise trials differ by batch
    order as well as by hyperparameters, which is noise the sampler reads as signal).
    """
    train_loader, _ = config.loader_fn(batch_size=batch_size, **loader_kwargs)
    dataset = train_loader.dataset

    n_val = max(1, round(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    if n_train < 1:
        raise ValueError(f"val_fraction={val_fraction} leaves no training examples for "
                         f"'{config.name}' ({len(dataset)} available)")

    train_split, val_split = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    shuffle_generator = torch.Generator().manual_seed(seed)
    return (DataLoader(train_split, batch_size=batch_size, shuffle=True, generator=shuffle_generator),
            DataLoader(val_split, batch_size=batch_size, shuffle=False),
            shuffle_generator)


def objective(trial, config, optimizer_key, base_kwargs, epochs, device,
              train_loader, val_loader, shuffle_generator, seed):
    set_seed(seed)
    shuffle_generator.manual_seed(seed)

    model = config.model_cls(**config.model_kwargs).to(device)

    kwargs = dict(base_kwargs)
    for name, spec in search_space_for(optimizer_key, base_kwargs).items():
        kwargs[name] = _suggest(trial, name, spec)
    optimizer = build_optimizer(optimizer_key, model.parameters(),
                                 **finalize_kwargs(optimizer_key, kwargs))
    scheduler = config.lr_scheduler_fn(optimizer) if config.lr_scheduler_fn else None

    def report(epoch, metric):
        trial.report(metric, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    train_fn = train_vae if config.task_type == "vae" else train_classifier
    result = train_fn(model, optimizer, f"{optimizer_key} trial{trial.number}",
                       train_loader, val_loader, epochs=epochs, device=device,
                       scheduler=scheduler, epoch_callback=report)

    # Final epoch, not best epoch: the benchmark reports final-epoch performance
    # (run_task.py), so tuning for a lucky mid-run peak would optimize the wrong thing.
    return result["acc_history"][-1]


def tune_optimizer(config, optimizer_key, n_trials, epochs, device, seed,
                    fixed_kwargs, loaders):
    """Runs one Optuna study. Returns (study, base_kwargs, direction)."""
    # config defaults (e.g. mnist_logreg's weight_decay) < --fixed < what Optuna suggests.
    base_kwargs = config.optimizer_kwargs_for(optimizer_key)
    base_kwargs.update(fixed_kwargs)

    # VAE tasks score reconstruction loss, where lower is better; everything else
    # scores accuracy.
    direction = "minimize" if config.task_type == "vae" else "maximize"

    train_loader, val_loader, shuffle_generator = loaders
    study = optuna.create_study(direction=direction,
                                 sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(
        lambda trial: objective(trial, config, optimizer_key, base_kwargs, epochs,
                                 device, train_loader, val_loader, shuffle_generator, seed),
        n_trials=n_trials)
    return study, base_kwargs, direction


def save_study(study, config, optimizer_key, base_kwargs, direction, epochs, batch_size,
                best_params_root="best_params", study_root="optuna_studies"):
    """Writes the tuned kwargs (constructor-ready, so betas are already assembled)
    plus the run's provenance. Returns the best_params path."""
    out_dir = os.path.join(best_params_root, config.name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{optimizer_key}.pkl")

    joblib.dump({
        "task": config.name,
        "optimizer": optimizer_key,
        "best_params": finalize_kwargs(optimizer_key, {**base_kwargs, **study.best_params}),
        "tuned_params": dict(study.best_params),  # only what Optuna searched
        "best_value": study.best_value,
        "direction": direction,
        "epochs": epochs,
        "batch_size": batch_size,
        "n_trials": len(study.trials),
    }, path)

    study_dir = os.path.join(study_root, config.name)
    os.makedirs(study_dir, exist_ok=True)
    joblib.dump(study, os.path.join(study_dir, f"{optimizer_key}.pkl"))
    return path


def parse_fixed(pairs):
    """--fixed lr=0.05 r_transform=sigmoid -> {"lr": 0.05, "r_transform": "sigmoid"}"""
    fixed = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep:
            raise ValueError(f"--fixed expects key=value pairs, got '{pair}'")
        try:
            fixed[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            fixed[key] = raw  # bare strings, e.g. r_transform=exp
    return fixed


def main():
    parser = argparse.ArgumentParser(
        description="Optuna-tune any optimizer against any task in TASK_REGISTRY, "
                    "scoring on a held-out split of the training set.")
    parser.add_argument("--task", required=True, choices=sorted(TASK_REGISTRY))
    parser.add_argument("--optimizer", nargs="+", default=OPTIMIZER_KEYS, choices=OPTIMIZER_KEYS,
                        help="Optimizers to tune (default: all of them, one study each).")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override the task's epoch count (default: the task's own).")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override the task's batch size (default: the task's own).")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Fraction of the training set held out for scoring (default: 0.1).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seeds model init, the train/val split, batch order, and the sampler.")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"],
                        help="Default: cuda when available, else cpu.")
    parser.add_argument("--fixed", nargs="+", default=[], metavar="KEY=VALUE",
                        help="Pin optimizer kwargs instead of searching them, e.g. r_transform=sigmoid.")
    parser.add_argument("--smoke", action="store_true",
                        help="Fast wiring check: the task's smoke config plus 2 trials.")
    args = parser.parse_args()

    config = TASK_REGISTRY[args.task]
    fixed_kwargs = parse_fixed(args.fixed)

    n_trials = 2 if args.smoke else args.n_trials
    epochs = args.epochs if args.epochs is not None else config.effective_epochs(args.smoke)
    batch_size = args.batch_size if args.batch_size is not None else config.effective_batch_size(args.smoke)
    loader_kwargs = config.effective_loader_kwargs(args.smoke)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Built once and shared by every trial and every optimizer: the data doesn't
    # depend on either, and rebuilding it per trial is pure overhead (minutes, for
    # IMDB's re-tokenization).
    loaders = train_val_loaders(config, batch_size, loader_kwargs, args.val_fraction, args.seed)

    print(f"Tuning {args.task} on {device} | {epochs} epochs | batch {batch_size} | "
          f"{n_trials} trials/optimizer | {len(loaders[0].dataset)} train / "
          f"{len(loaders[1].dataset)} val examples")

    for optimizer_key in args.optimizer:
        print(f"\n=== {optimizer_key} ===")
        study, base_kwargs, direction = tune_optimizer(
            config, optimizer_key, n_trials, epochs, device, args.seed, fixed_kwargs, loaders)
        path = save_study(study, config, optimizer_key, base_kwargs, direction, epochs, batch_size)
        print(f"{optimizer_key}: best value={study.best_value:.4f} ({direction}) "
              f"params={study.best_params}\n  -> {path}")


if __name__ == "__main__":
    main()
