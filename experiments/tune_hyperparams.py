import argparse
import os

import joblib
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from deltagrad.data import maybe_subset
from deltagrad.models import ConvNet5Layer
from deltagrad.optimizers import DeltaGradWindowed

CIFAR_MEAN = (0.5, 0.5, 0.5)
CIFAR_STD = (0.5, 0.5, 0.5)


def _cifar100_train_val_loaders(batch_size, subset_size=None, root="data"):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    full_trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
    # Held-out validation split for tuning only -- never touches the real test set.
    train_subset, val_subset = random_split(
        full_trainset, [45000, 5000], generator=torch.Generator().manual_seed(42))

    if subset_size is not None:
        train_subset = maybe_subset(train_subset, subset_size)
        val_subset = maybe_subset(val_subset, subset_size)

    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    return trainloader, valloader


def train_and_validate(trial, model, optimizer, epochs, trainloader, valloader, device):
    criterion = nn.CrossEntropyLoss()
    val_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_accuracy = 100 * val_correct / val_total

        trial.report(val_accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_accuracy


def objective(trial, optimizer_name, epochs, batch_size, subset_size):
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNet5Layer(num_classes=100).to(device)
    trainloader, valloader = _cifar100_train_val_loaders(batch_size, subset_size)

    if optimizer_name == "adam":
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        # DeltaGradWindowed's real Sec. 2 kwargs -- no `gamma` (removed upstream) and
        # no `beta` (that was only ever part of the legacy formula).
        lr = trial.suggest_float("lr", 1e-4, 0.5, log=True)
        alpha = trial.suggest_float("alpha", 0.1, 0.9)
        sigma = trial.suggest_float("sigma", 0.5, 0.99)
        k_val = trial.suggest_int("K", 2, 8)
        optimizer = DeltaGradWindowed(model.parameters(), lr=lr, alpha=alpha, sigma=sigma, K=k_val)

    return train_and_validate(trial, model, optimizer, epochs, trainloader, valloader, device)


def main():
    parser = argparse.ArgumentParser(
        description="Optuna tuning for DeltaGradWindowed vs Adam on CIFAR-100/ConvNet5Layer "
                    "(deltagradpaperplan.pdf Sec 4.1's LR-stress-test base LR).")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    n_trials = 2 if args.smoke else args.n_trials
    epochs = 1 if args.smoke else args.epochs
    batch_size = 16 if args.smoke else args.batch_size
    subset_size = 64 if args.smoke else None

    best_params_dir = "best_params/windowed"
    study_dir = "optuna_studies"
    os.makedirs(best_params_dir, exist_ok=True)
    os.makedirs(study_dir, exist_ok=True)

    for optimizer_name in ["adam", "windowed"]:
        print(f"Starting {optimizer_name} tuning...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, optimizer_name, epochs, batch_size, subset_size),
                       n_trials=n_trials)

        joblib.dump(study.best_params,
                    os.path.join(best_params_dir, f"best_params_{optimizer_name}_b{batch_size}_epochs{epochs}.pkl"))
        joblib.dump(study, os.path.join(study_dir, f"study_{optimizer_name}_b{batch_size}_epochs{epochs}.pkl"))
        print(f"{optimizer_name}: best value={study.best_value:.3f} best_params={study.best_params}")


if __name__ == "__main__":
    main()
