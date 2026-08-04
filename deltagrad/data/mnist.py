import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from . import maybe_subset
from .noise import inject_label_noise

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def get_mnist_loaders(batch_size, noise_rate=0.0, subset_size=None, normalize=True,
                       root="data", num_workers=0):
    """normalize=False keeps pixels in [0,1] (needed for the VAE's BCE reconstruction
    loss); normalize=True standardizes for the classification tasks."""
    tf = [transforms.ToTensor()]
    if normalize:
        tf.append(transforms.Normalize(MNIST_MEAN, MNIST_STD))
    transform = transforms.Compose(tf)

    trainset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)

    if noise_rate > 0:
        inject_label_noise(trainset, noise_rate, 10)

    trainset = maybe_subset(trainset, subset_size)
    testset = maybe_subset(testset, subset_size)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return trainloader, testloader
