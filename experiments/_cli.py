import numpy as np
import torch


def add_smoke_args(parser):
    """Shared --smoke / --n-runs flags for every experiment script."""
    parser.add_argument("--smoke", action="store_true",
                         help="Run a fast local smoke test (tiny subset, few epochs) "
                              "instead of the full Table-1-accurate config.")
    parser.add_argument("--n-runs", type=int, default=None,
                         help="Override the number of seeded repetitions "
                              "(default: 5 full-scale, 1 smoke).")
    return parser


def set_seed(seed=None):
    seed = seed if seed is not None else torch.seed()
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    return seed
