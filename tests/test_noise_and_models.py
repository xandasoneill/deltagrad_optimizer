import torch
import torch.nn as nn

from deltagrad.data.noise import inject_label_noise
from deltagrad.models import LogisticRegression, MLP2Layer, MNISTVAE, vae_loss


class _FakeDataset:
    def __init__(self, targets):
        self.targets = list(targets)


def test_noise_rate_zero_is_noop():
    targets = [0, 1, 2, 3, 4]
    ds = _FakeDataset(targets)
    inject_label_noise(ds, 0.0, num_classes=5)
    assert ds.targets == targets


def test_noise_rate_flips_expected_fraction():
    targets = [i % 5 for i in range(100)]
    ds = _FakeDataset(targets)
    inject_label_noise(ds, 0.3, num_classes=5, generator=torch.Generator().manual_seed(0))
    n_changed = sum(1 for a, b in zip(targets, ds.targets) if a != b)
    assert n_changed == 30  # the while-resample loop guarantees every selected index changes


def test_noise_never_assigns_same_label():
    targets = [0] * 50
    ds = _FakeDataset(targets)
    inject_label_noise(ds, 1.0, num_classes=3, generator=torch.Generator().manual_seed(0))
    assert all(t != 0 for t in ds.targets)


def test_dropout_p_constructs_dropout_submodule():
    model = LogisticRegression(10, 2, dropout_p=0.5)
    assert isinstance(model.dropout, nn.Dropout)
    assert model.dropout.p == 0.5

    mlp = MLP2Layer(dropout_p=0.3)
    assert isinstance(mlp.dropout, nn.Dropout)
    assert mlp.dropout.p == 0.3


def test_vae_forward_shapes_and_loss():
    model = MNISTVAE(input_dim=784, hidden=64, latent_dim=8)
    x = torch.rand(5, 1, 28, 28)
    recon, mu, logvar = model(x)
    assert recon.shape == (5, 784)
    assert mu.shape == (5, 8)
    assert logvar.shape == (5, 8)

    loss = vae_loss(recon, x, mu, logvar)
    assert loss.dim() == 0
    assert loss.item() > 0
