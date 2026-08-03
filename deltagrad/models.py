import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet5Layer(nn.Module):
    """3 conv layers + 2 FC layers -- the CIFAR-100 "5-Layer CNN" DeltaGrad Abstract
    setup (also reused for CIFAR-10 by overriding num_classes)."""

    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ConvNet3Stage(nn.Module):
    """3 alternating stages of 5x5 conv + 3x3 maxpool, then a 1000-unit FC layer --
    the CIFAR-10 ConvNet (Adam Paper Setup) replication. Channel widths (64/128/256)
    aren't specified in deltagradpaperplan.pdf; chosen as a reasonable fill-in."""

    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        self.stage1 = self._stage(in_channels, 64)
        self.stage2 = self._stage(64, 128)
        self.stage3 = self._stage(128, 256)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 4 * 4, 1000)
        self.fc2 = nn.Linear(1000, num_classes)
        self.relu = nn.ReLU()

    @staticmethod
    def _stage(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class LogisticRegression(nn.Module):
    """Plain multi-class logistic regression, with an optional dropout layer on the
    input features -- used for both MNIST LogReg (dropout_p=0.0) and IMDB BoW
    (dropout_p=0.5, per Table 1's "+ 50% Dropout")."""

    def __init__(self, input_dim, num_classes, dropout_p=0.0):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_p)
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout(x)
        return self.linear(x)


class MLP2Layer(nn.Module):
    """2 hidden layers x 1000 ReLU units (MNIST MLP, Adam Paper Setup). dropout_p=0.0
    reproduces the "deterministic loss" variant; dropout_p=0.5 the "dropout noise"
    variant Table 1 asks for."""

    def __init__(self, input_dim=784, hidden=1000, num_classes=10, dropout_p=0.0):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, num_classes)
        self.dropout = nn.Dropout(dropout_p)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)


class MNISTVAE(nn.Module):
    """500 softplus units, 50-dim Gaussian latent (MNIST VAE, Adam Paper Setup)."""

    def __init__(self, input_dim=784, hidden=500, latent_dim=50):
        super().__init__()
        self.flatten = nn.Flatten()
        self.encoder = nn.Linear(input_dim, hidden)
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.decoder_hidden = nn.Linear(latent_dim, hidden)
        self.decoder_out = nn.Linear(hidden, input_dim)
        self.softplus = nn.Softplus()

    def encode(self, x):
        h = self.softplus(self.encoder(self.flatten(x)))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        h = self.softplus(self.decoder_hidden(z))
        return torch.sigmoid(self.decoder_out(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    """Standard ELBO loss (BCE reconstruction + KL divergence), summed over the
    batch -- divide by batch size for a per-example average if needed."""
    bce = F.binary_cross_entropy(recon_x, x.view(recon_x.shape), reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld
