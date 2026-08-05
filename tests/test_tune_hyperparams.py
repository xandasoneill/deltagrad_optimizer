"""Covers the Optuna plumbing in experiments/tune_hyperparams.py: which kwargs a
study searches, and the trial budget of a resumed study. Both fail silently when
broken -- a study that quietly stops searching a parameter, or one that re-runs its
whole budget on every resume, still produces plausible-looking output."""

import optuna
import pytest

from experiments.configs import TASK_REGISTRY
from experiments.tune_hyperparams import (
    finalize_kwargs,
    search_space_for,
    sqlite_storage,
    train_val_loaders,
    tune_optimizer,
)

TASK = "mnist_logreg"


@pytest.fixture(autouse=True)
def _quiet_optuna():
    optuna.logging.set_verbosity(optuna.logging.WARNING)


@pytest.fixture(scope="module")
def smoke_loaders():
    """The task's smoke config (64 examples, batch 8), so a trial costs a fraction
    of a second and the tests below are about wiring, not about learning anything."""
    config = TASK_REGISTRY[TASK]
    loaders = train_val_loaders(config, config.effective_batch_size(smoke=True),
                                config.effective_loader_kwargs(smoke=True),
                                val_fraction=0.2, seed=0)
    return config, loaders


def _tune(config, loaders, n_trials, storage):
    return tune_optimizer(config, "adam", n_trials, epochs=1, device="cpu", seed=0,
                          fixed_kwargs={}, loaders=loaders, storage=storage)


def test_resume_tops_up_to_the_budget_instead_of_repeating_it(smoke_loaders, tmp_path):
    config, loaders = smoke_loaders
    storage = sqlite_storage(TASK, study_root=str(tmp_path))

    study, _, _ = _tune(config, loaders, 2, storage)
    assert len(study.trials) == 2

    # Same budget again: the study is already paid up, so nothing more should run.
    study, _, _ = _tune(config, loaders, 2, storage)
    assert len(study.trials) == 2

    # A raised budget searches further rather than starting over.
    study, _, _ = _tune(config, loaders, 4, storage)
    assert len(study.trials) == 4


def test_study_without_storage_starts_empty_every_time(smoke_loaders):
    """The default path must keep its old semantics: an in-memory study runs the
    trials it was asked for and remembers nothing between calls."""
    config, loaders = smoke_loaders

    first, _, _ = _tune(config, loaders, 2, None)
    second, _, _ = _tune(config, loaders, 2, None)
    assert len(first.trials) == len(second.trials) == 2


def test_ema_space_follows_the_active_r_transform():
    """Each Sec. 3.2 transform reads its own shape parameters and ignores the rest;
    searching a knob the active transform never reads would waste trials."""
    assert "gamma" in search_space_for("ema", {"r_transform": "exp"})

    sigmoid = search_space_for("ema", {"r_transform": "sigmoid"})
    assert {"tau", "s"} <= set(sigmoid) and "gamma" not in sigmoid

    # Shared across every transform, so always searched.
    assert {"lr", "sigma", "beta_phi", "beta_m"} <= set(sigmoid)

    with pytest.raises(ValueError, match="Unknown r_transform"):
        search_space_for("ema", {"r_transform": "not_a_transform"})


def test_finalize_kwargs_recombines_adam_betas():
    """Optuna has to search beta1/beta2 as separate scalars; torch's constructor
    only accepts the pair."""
    finalized = finalize_kwargs("adam", {"lr": 0.01, "beta1": 0.8, "beta2": 0.99})
    assert finalized == {"lr": 0.01, "betas": (0.8, 0.99)}

    # Optimizers without a betas tuple are passed through untouched.
    assert finalize_kwargs("sgd_momentum", {"lr": 0.01, "momentum": 0.9}) == \
        {"lr": 0.01, "momentum": 0.9}


def test_validation_split_comes_out_of_the_training_set(smoke_loaders):
    """Scoring on the test set would leak it into the benchmark that later reports
    on these hyperparameters. Train + validation adding up to exactly the training
    set is what says the split was carved from it and nothing else was pulled in."""
    config, (train_loader, val_loader, _) = smoke_loaders
    full_train, _ = config.loader_fn(batch_size=config.effective_batch_size(smoke=True),
                                     **config.effective_loader_kwargs(smoke=True))

    assert len(train_loader.dataset) + len(val_loader.dataset) == len(full_train.dataset)
    assert len(val_loader.dataset) > 0


def test_search_spaces_cover_every_registered_optimizer():
    """A new optimizer key without a search space would tune nothing at all and
    still write a best_params file."""
    from experiments.configs import OPTIMIZER_KEYS
    from experiments.tune_hyperparams import SEARCH_SPACES

    assert set(SEARCH_SPACES) == set(OPTIMIZER_KEYS)
    for key, space in SEARCH_SPACES.items():
        assert "lr" in space, f"{key} does not search a learning rate"
