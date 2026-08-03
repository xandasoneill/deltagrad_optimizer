import torch.optim as optim

_BASELINE_REGISTRY = {
    "adam":         (optim.Adam,    dict(lr=1e-3)),
    "adamw":        (optim.AdamW,   dict(lr=1e-3, weight_decay=1e-2)),
    "sgd_momentum": (optim.SGD,     dict(lr=1e-2, momentum=0.9)),
    "adagrad":      (optim.Adagrad, dict(lr=1e-2)),
    "rmsprop":      (optim.RMSprop, dict(lr=1e-3, alpha=0.99)),
}


def make_baseline_optimizer(name, params, **overrides):
    """Builds a standard torch.optim baseline by name: adam, adamw, sgd_momentum,
    adagrad, rmsprop. `overrides` take precedence over the (unconditioned) sane
    defaults -- experiment configs should always supply their own tuned `lr`.
    """
    key = name.lower()
    if key not in _BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline optimizer '{name}'. Choices: {list(_BASELINE_REGISTRY)}")
    cls, defaults = _BASELINE_REGISTRY[key]
    return cls(params, **{**defaults, **overrides})
