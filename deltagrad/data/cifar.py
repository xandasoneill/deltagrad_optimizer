import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from . import maybe_subset
from .noise import inject_label_noise

CIFAR_MEAN = (0.5, 0.5, 0.5)
CIFAR_STD = (0.5, 0.5, 0.5)


def _cifar_loaders(dataset_cls, num_classes, batch_size, noise_rate=0.0, augment=False,
                    subset_size=None, root="data", num_workers=0):
    train_tf = []
    if augment:
        train_tf += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    train_tf += [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
    test_tf = [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]

    trainset = dataset_cls(root=root, train=True, download=True, transform=transforms.Compose(train_tf))
    testset = dataset_cls(root=root, train=False, download=True, transform=transforms.Compose(test_tf))

    if noise_rate > 0:
        inject_label_noise(trainset, noise_rate, num_classes)

    trainset = maybe_subset(trainset, subset_size)
    testset = maybe_subset(testset, subset_size)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return trainloader, testloader


def get_cifar100_loaders(batch_size, noise_rate=0.0, augment=False, subset_size=None,
                          root="data", num_workers=0):
    return _cifar_loaders(torchvision.datasets.CIFAR100, 100, batch_size, noise_rate,
                           augment, subset_size, root, num_workers)


def get_cifar10_loaders(batch_size, noise_rate=0.0, augment=False, subset_size=None,
                         root="data", num_workers=0):
    return _cifar_loaders(torchvision.datasets.CIFAR10, 10, batch_size, noise_rate,
                           augment, subset_size, root, num_workers)
