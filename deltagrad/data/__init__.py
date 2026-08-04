import torch
from torch.utils.data import Subset


def maybe_subset(dataset, subset_size=None, generator=None):
    """Returns `dataset` unchanged, or a random Subset of `subset_size` examples.

    Shared by every loader function so smoke-mode subsetting works identically
    across CIFAR/MNIST/IMDB instead of being reimplemented per dataset.
    """
    if subset_size is None or subset_size >= len(dataset):
        return dataset
    generator = generator or torch.Generator().manual_seed(0)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size]
    return Subset(dataset, indices.tolist())
