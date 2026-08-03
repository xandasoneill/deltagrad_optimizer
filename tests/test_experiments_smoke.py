import time

import pytest

from experiments.configs import TASK_REGISTRY
from experiments.final_benchmark import run_benchmark

EXPECTED_KEYS = {
    "optimizer", "task", "epochs", "batch_size", "number_runs", "model_name",
    "acc_history", "loss_history", "r_history", "variance_history",
    "all_timestamps", "optimizer_hyperparameters", "all_total_times",
    "seeds", "device", "start_time", "smoke",
}


@pytest.mark.parametrize("task_name", sorted(TASK_REGISTRY))
@pytest.mark.parametrize("optimizer_key", ["windowed", "adam"])
def test_smoke_run_completes_quickly_with_expected_shape(task_name, optimizer_key):
    """Not the full 7-optimizer x 8-task matrix (too slow even at smoke scale) --
    windowed + adam is enough to prove the wiring works end to end for every task.
    The full matrix is runnable manually via `python -m experiments.run_task`."""
    config = TASK_REGISTRY[task_name]
    start = time.time()
    results = run_benchmark(config, optimizer_key, n_runs=1, smoke=True)
    elapsed = time.time() - start

    assert elapsed < 60, f"{task_name}/{optimizer_key} smoke run took {elapsed:.1f}s (>60s)"
    assert EXPECTED_KEYS.issubset(results)
    assert len(results["acc_history"]) == 1
    assert len(results["acc_history"][0]) == config.smoke_epochs
