import os
import time

import joblib
import torch

from deltagrad.optimizers import DeltaGradWindowed, DeltaGradEMA, make_baseline_optimizer
from deltagrad.training import train_classifier, train_vae

from experiments._cli import set_seed


def build_optimizer(optimizer_key, params, **kwargs):
    """Constructs an optimizer by key ("windowed", "ema", or any baseline name from
    deltagrad.optimizers.baselines)."""
    if optimizer_key == "windowed":
        return DeltaGradWindowed(params, **kwargs)
    if optimizer_key == "ema":
        return DeltaGradEMA(params, **kwargs)
    return make_baseline_optimizer(optimizer_key, params, **kwargs)


def load_tuned_kwargs(task_name, optimizer_key, root="best_params"):
    """Reads the kwargs experiments/tune_hyperparams.py tuned for this task/optimizer
    pair, or None if that pair was never tuned. Also accepts a hand-written pkl
    holding a bare kwargs dict."""
    path = os.path.join(root, task_name, f"{optimizer_key}.pkl")
    if not os.path.isfile(path):
        return None
    payload = joblib.load(path)
    return payload.get("best_params", payload)


def run_benchmark(config, optimizer_key, n_runs=None, smoke=False, device=None,
                   optimizer_kwargs_override=None, grad_variance_every=10,
                   sample_transform_every=None):
    """Runs `n_runs` seeded repetitions of `config` (an experiments.configs.
    ExperimentConfig) with the optimizer named by `optimizer_key`. Returns a
    results dict ready for joblib.dump / deltagrad.viz plotting."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_runs = n_runs if n_runs is not None else (1 if smoke else 5)

    batch_size = config.effective_batch_size(smoke)
    epochs = config.effective_epochs(smoke)
    loader_kwargs = config.effective_loader_kwargs(smoke)
    train_fn = train_vae if config.task_type == "vae" else train_classifier

    acc_history, loss_history, r_history, variance_history = [], [], [], []
    total_net_time_history, time_stamps_history = [], []
    experiment_start_time_history = []
    seeds_used = []
    optimizer_hyperparameters = {}
    transform_samples_history, transform_spec = [], None

    for _ in range(n_runs):
        seed = set_seed()
        seeds_used.append(seed)

        train_loader, test_loader = config.loader_fn(batch_size=batch_size, **loader_kwargs)
        model = config.model_cls(**config.model_kwargs).to(device)

        optimizer_hyperparameters = config.optimizer_kwargs_for(optimizer_key)
        if optimizer_kwargs_override:
            optimizer_hyperparameters.update(optimizer_kwargs_override)
        # Only DeltaGradEMA has an R-transform to sample; asking any other
        # optimizer for one would just be a TypeError from its constructor.
        if sample_transform_every and optimizer_key == "ema":
            optimizer_hyperparameters["sample_every"] = sample_transform_every
        optimizer = build_optimizer(optimizer_key, model.parameters(), **optimizer_hyperparameters)
        scheduler = config.lr_scheduler_fn(optimizer) if config.lr_scheduler_fn else None

        result = train_fn(model, optimizer, optimizer_key, train_loader, test_loader,
                           epochs=epochs, device=device, scheduler=scheduler,
                           grad_variance_every=grad_variance_every)

        # Pulled off the optimizer rather than out of `result` so the sampling
        # stays invisible to deltagrad/training.py, which has no reason to know
        # any particular optimizer keeps diagnostics.
        if getattr(optimizer, "transform_samples", None):
            transform_samples_history.append(optimizer.transform_samples)
            transform_spec = optimizer.transform_spec()

        acc_history.append(result["acc_history"])
        loss_history.append(result["loss_history"])
        r_history.append(result["r_values"])
        variance_history.append(result["variance_values"])
        total_net_time_history.append(result["total_net_time"])
        time_stamps_history.append(result["time_stamps"])
        experiment_start_time_history.append(time.ctime(result["experiment_start_time"]))

    return {
        "optimizer": optimizer_key,
        "task": config.name,
        "epochs": epochs,
        "batch_size": batch_size,
        "number_runs": n_runs,
        "model_name": config.model_cls.__name__,
        "acc_history": acc_history,
        "loss_history": loss_history,
        "r_history": r_history,
        "variance_history": variance_history,
        "all_timestamps": time_stamps_history,
        "transform_samples": transform_samples_history,
        "transform_spec": transform_spec,
        "optimizer_hyperparameters": optimizer_hyperparameters,
        "all_total_times": total_net_time_history,
        "seeds": seeds_used,
        "device": str(device),
        "start_time": experiment_start_time_history,
        "smoke": smoke,
    }


def save_results(results, root="results"):
    subdir = "_smoke_runs" if results.get("smoke") else ""
    out_dir = os.path.join(root, subdir, results["task"]) if subdir else os.path.join(root, results["task"])
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{results['optimizer']}_results.pkl")
    joblib.dump(results, path)
    return path
