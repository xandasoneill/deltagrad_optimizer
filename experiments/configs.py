import math
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

from torch.optim.lr_scheduler import LambdaLR

from deltagrad.models import ConvNet5Layer, ConvNet3Stage, LogisticRegression, MLP2Layer, MNISTVAE
from deltagrad.data.cifar import get_cifar100_loaders, get_cifar10_loaders
from deltagrad.data.mnist import get_mnist_loaders
from deltagrad.data.imdb import get_imdb_bow_loaders

OPTIMIZER_KEYS = ["windowed", "ema", "adam", "adamw", "sgd_momentum", "adagrad", "rmsprop"]


@dataclass
class ExperimentConfig:
    """One row of deltagradpaperplan.pdf Table 1. `default_optimizer_kwargs` are
    reasonable, untuned starting points (only the CIFAR-100/ConvNet task has an
    actual Optuna-tuned LR, via experiments/tune_hyperparams.py + best_params/) --
    not literature-exact values for every task.
    """
    name: str
    task_type: str  # "classification" | "vae"
    model_cls: type
    model_kwargs: dict
    loader_fn: Callable
    loader_kwargs: dict
    batch_size: int
    epochs: int
    default_optimizer_kwargs: dict  # {optimizer_key: {kwarg: value}}
    lr_scheduler_fn: Optional[Callable] = None  # (optimizer) -> LRScheduler
    smoke_loader_kwargs: dict = field(default_factory=dict)
    smoke_batch_size: int = 8
    smoke_epochs: int = 1
    smoke_subset_size: int = 64

    def effective_batch_size(self, smoke):
        return self.smoke_batch_size if smoke else self.batch_size

    def effective_epochs(self, smoke):
        return self.smoke_epochs if smoke else self.epochs

    def effective_loader_kwargs(self, smoke):
        kwargs = dict(self.loader_kwargs)
        if smoke:
            kwargs["subset_size"] = self.smoke_subset_size
            kwargs.update(self.smoke_loader_kwargs)
        return kwargs

    def optimizer_kwargs_for(self, optimizer_key):
        return dict(self.default_optimizer_kwargs.get(optimizer_key, {}))


def _sqrt_decay_scheduler(optimizer):
    """alpha_t = alpha / sqrt(t+1), per-step -- deltagradpaperplan.pdf Sec. 4.1's
    MNIST LogReg replication, applied uniformly to every optimizer compared."""
    return LambdaLR(optimizer, lr_lambda=lambda t: 1 / math.sqrt(t + 1))


_STANDARD_OPTIMIZER_KWARGS = {
    "windowed": {"lr": 0.05},
    "ema": {"lr": 0.01},
    "adam": {"lr": 1e-3},
    "adamw": {"lr": 1e-3},
    "sgd_momentum": {"lr": 0.05},
    "adagrad": {"lr": 0.01},
    "rmsprop": {"lr": 1e-3},
}


def _with_weight_decay(kwargs_by_optimizer, weight_decay):
    return {k: {**v, "weight_decay": weight_decay} for k, v in kwargs_by_optimizer.items()}


CIFAR100_NOISE_0 = ExperimentConfig(
    name="cifar100_noise_0",
    task_type="classification",
    model_cls=ConvNet5Layer,
    model_kwargs={"num_classes": 100},
    loader_fn=get_cifar100_loaders,
    loader_kwargs={"noise_rate": 0.0, "augment": False},
    batch_size=512,
    epochs=50,
    default_optimizer_kwargs=_STANDARD_OPTIMIZER_KWARGS,
)

CIFAR100_NOISE_20 = replace(
    CIFAR100_NOISE_0, name="cifar100_noise_20",
    loader_kwargs={"noise_rate": 0.20, "augment": False},
)

# Base config for the LR-stress-test replication (experiments/run_cifar100_lr_stress.py
# sweeps LR multipliers x seeds on top of this; kept out of TASK_REGISTRY since its
# shape doesn't fit run_task.py's generic one-task-one-optimizer dispatch).
CIFAR100_LR_STRESS_BASE = replace(
    CIFAR100_NOISE_0, name="cifar100_lr_stress",
    batch_size=16, epochs=50,
)

MNIST_LOGREG = ExperimentConfig(
    name="mnist_logreg",
    task_type="classification",
    model_cls=LogisticRegression,
    model_kwargs={"input_dim": 784, "num_classes": 10, "dropout_p": 0.0},
    loader_fn=get_mnist_loaders,
    loader_kwargs={"noise_rate": 0.0},
    batch_size=128,
    epochs=45,
    lr_scheduler_fn=_sqrt_decay_scheduler,
    default_optimizer_kwargs=_with_weight_decay(_STANDARD_OPTIMIZER_KWARGS, 1e-4),
)

IMDB_BOW = ExperimentConfig(
    name="imdb_bow",
    task_type="classification",
    model_cls=LogisticRegression,
    model_kwargs={"input_dim": 10_000, "num_classes": 2, "dropout_p": 0.5},
    loader_fn=get_imdb_bow_loaders,
    loader_kwargs={"vocab_size": 10_000},
    batch_size=128,
    epochs=175,
    default_optimizer_kwargs=_STANDARD_OPTIMIZER_KWARGS,
)

MNIST_MLP_DETERMINISTIC = ExperimentConfig(
    name="mnist_mlp_deterministic",
    task_type="classification",
    model_cls=MLP2Layer,
    model_kwargs={"input_dim": 784, "hidden": 1000, "num_classes": 10, "dropout_p": 0.0},
    loader_fn=get_mnist_loaders,
    loader_kwargs={"noise_rate": 0.0},
    batch_size=128,
    epochs=200,
    default_optimizer_kwargs=_STANDARD_OPTIMIZER_KWARGS,
)

MNIST_MLP_DROPOUT = replace(
    MNIST_MLP_DETERMINISTIC, name="mnist_mlp_dropout",
    model_kwargs={"input_dim": 784, "hidden": 1000, "num_classes": 10, "dropout_p": 0.5},
)

CIFAR10_CONV = ExperimentConfig(
    name="cifar10_conv",
    task_type="classification",
    model_cls=ConvNet3Stage,
    model_kwargs={"num_classes": 10},
    loader_fn=get_cifar10_loaders,
    loader_kwargs={"noise_rate": 0.0, "augment": False},
    batch_size=128,
    epochs=45,
    default_optimizer_kwargs=_STANDARD_OPTIMIZER_KWARGS,
)

MNIST_VAE = ExperimentConfig(
    name="mnist_vae",
    task_type="vae",
    model_cls=MNISTVAE,
    model_kwargs={},
    loader_fn=get_mnist_loaders,
    loader_kwargs={"normalize": False},
    batch_size=128,
    epochs=100,
    default_optimizer_kwargs=_STANDARD_OPTIMIZER_KWARGS,
)

# The 7 "core" tasks with a uniform (one model, one optimizer) shape -- covered by
# the generic experiments/run_task.py CLI. cifar100_lr_stress has its own script
# (run_cifar100_lr_stress.py) since it sweeps LR multipliers x seeds instead.
TASK_REGISTRY = {
    cfg.name: cfg for cfg in [
        CIFAR100_NOISE_0,
        CIFAR100_NOISE_20,
        MNIST_LOGREG,
        IMDB_BOW,
        MNIST_MLP_DETERMINISTIC,
        MNIST_MLP_DROPOUT,
        CIFAR10_CONV,
        MNIST_VAE,
    ]
}
