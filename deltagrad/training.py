import time

import torch
import torch.nn as nn

from .models import vae_loss as _default_vae_loss


def get_grad_variance(model, criterion, inputs, labels, num_samples=8):
    """Variance of gradients across different random sub-samples of a batch --
    an empirical proxy for how noisy the current batch's gradient signal is."""
    grads = []
    model.train()

    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.track_running_stats = False

    sample_size = max(1, inputs.size(0) // 4)
    for _ in range(num_samples):
        indices = torch.randperm(inputs.size(0))[:sample_size]
        model.zero_grad()
        outputs = model(inputs[indices])
        loss = criterion(outputs, labels[indices])
        loss.backward()
        all_grads = torch.cat([p.grad.detach().view(-1) for p in model.parameters() if p.grad is not None])
        grads.append(all_grads)

    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.track_running_stats = True

    return torch.var(torch.stack(grads), dim=0).mean().item()


def _measure_grad_variance_and_R(model, optimizer, criterion, inputs, labels):
    """Backs up in-flight grads, measures variance + the optimizer's own R (if it
    tracks one), then restores the original grads so optimizer.step() is unaffected."""
    params_with_grad = [p for p in model.parameters() if p.grad is not None]
    original_grads = [p.grad.clone() for p in params_with_grad]

    variance = get_grad_variance(model, criterion, inputs, labels)

    avg_R, n_with_R = 0.0, 0
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state.get(p, {})
            if 'R' in state:
                avg_R += state['R'].mean().item()
                n_with_R += 1
    r_value = avg_R / n_with_R if n_with_R > 0 else None

    for p, g in zip(params_with_grad, original_grads):
        p.grad = g

    return variance, r_value


def train_classifier(model, optimizer, optimizer_name, train_loader, test_loader,
                      epochs, device=torch.device("cpu"), grad_variance_every=10,
                      criterion=None, scheduler=None):
    """Standard classification train/eval loop with grad-variance + R instrumentation.
    Returns a dict: acc_history, loss_history, r_values, variance_values,
    total_net_time, time_stamps, experiment_start_time, device."""
    criterion = criterion or nn.CrossEntropyLoss()
    model.to(device)

    experiment_start_time = time.time()
    acc_history, loss_history = [], []
    r_values, variance_values = [], []
    total_net_time = 0.0
    time_stamps = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            start_batch = time.time()
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            total_net_time += (time.time() - start_batch)

            if i % grad_variance_every == 0 and i != 0:
                variance, r_value = _measure_grad_variance_and_R(model, optimizer, criterion, inputs, labels)
                variance_values.append(variance)
                if r_value is not None:
                    r_values.append(r_value)

            start_step = time.time()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_net_time += (time.time() - start_step)

            running_loss += loss.item()

        start_eval = time.time()
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        total_net_time += (time.time() - start_eval)
        time_stamps.append(total_net_time)

        epoch_test_acc = 100 * test_correct / test_total
        acc_history.append(epoch_test_acc)
        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)

        print(f"[{optimizer_name}] Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.4f} | Test Acc: {epoch_test_acc:.2f}%")

    return {
        "acc_history": acc_history,
        "loss_history": loss_history,
        "r_values": r_values,
        "variance_values": variance_values,
        "total_net_time": total_net_time,
        "time_stamps": time_stamps,
        "experiment_start_time": experiment_start_time,
        "device": str(device),
    }


def train_vae(model, optimizer, optimizer_name, train_loader, test_loader,
              epochs, device=torch.device("cpu"), grad_variance_every=10,
              loss_fn=None, scheduler=None):
    """Same shape/instrumentation as train_classifier, but for VAE reconstruction:
    `acc_history` holds mean test reconstruction loss per epoch (lower is better)
    instead of accuracy."""
    loss_fn = loss_fn or _default_vae_loss

    def criterion(outputs, targets):
        recon, mu, logvar = outputs
        return loss_fn(recon, targets, mu, logvar)

    model.to(device)
    experiment_start_time = time.time()
    acc_history, loss_history = [], []
    r_values, variance_values = [], []
    total_net_time = 0.0
    time_stamps = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, _) in enumerate(train_loader):
            start_batch = time.time()
            inputs = inputs.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            total_net_time += (time.time() - start_batch)

            if i % grad_variance_every == 0 and i != 0:
                variance, r_value = _measure_grad_variance_and_R(model, optimizer, criterion, inputs, inputs)
                variance_values.append(variance)
                if r_value is not None:
                    r_values.append(r_value)

            start_step = time.time()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_net_time += (time.time() - start_step)

            running_loss += loss.item() / inputs.size(0)  # vae_loss sums over the batch

        start_eval = time.time()
        model.eval()
        test_loss_total, test_examples = 0.0, 0
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                test_loss_total += criterion(outputs, inputs).item()
                test_examples += inputs.size(0)
        total_net_time += (time.time() - start_eval)
        time_stamps.append(total_net_time)

        epoch_test_loss = test_loss_total / test_examples
        acc_history.append(epoch_test_loss)
        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)

        print(f"[{optimizer_name}] Epoch {epoch + 1}/{epochs} | Train Loss: {epoch_loss:.4f} | Test Recon Loss: {epoch_test_loss:.4f}")

    return {
        "acc_history": acc_history,
        "loss_history": loss_history,
        "r_values": r_values,
        "variance_values": variance_values,
        "total_net_time": total_net_time,
        "time_stamps": time_stamps,
        "experiment_start_time": experiment_start_time,
        "device": str(device),
    }
