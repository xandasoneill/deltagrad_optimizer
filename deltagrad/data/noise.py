import torch


def inject_label_noise(dataset, noise_rate, num_classes, generator=None):
    """Randomly reassigns `noise_rate` fraction of `dataset.targets` to a different
    class (mutates in place). Works for any torchvision dataset exposing `.targets`
    as a list/tensor of ints (CIFAR10/100, MNIST). noise_rate<=0 is a no-op."""
    if noise_rate <= 0:
        return dataset

    num_samples = len(dataset.targets)
    num_noisy = int(noise_rate * num_samples)
    generator = generator or torch.Generator().manual_seed(0)
    noisy_indices = torch.randperm(num_samples, generator=generator)[:num_noisy]

    for idx in noisy_indices.tolist():
        current_label = int(dataset.targets[idx])
        new_label = torch.randint(0, num_classes, (1,), generator=generator).item()
        while new_label == current_label:
            new_label = torch.randint(0, num_classes, (1,), generator=generator).item()
        dataset.targets[idx] = new_label

    return dataset
