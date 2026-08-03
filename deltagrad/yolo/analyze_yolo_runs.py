import glob
import os

import matplotlib.pyplot as plt
import torch


def analyze_deltagrad_runs(log_dir="deltagrad_analysis"):
    files = sorted(glob.glob(os.path.join(log_dir, "epoch_*.pt")),
                    key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

    epochs = []
    avg_R = []
    weight_l2_norms = []
    grad_norms = []

    for file in files:
        epoch_num = int(os.path.basename(file).split('_')[1].split('.')[0])
        epochs.append(epoch_num)

        data = torch.load(file)

        epoch_R = []
        epoch_l2 = 0.0
        epoch_grad_norm = 0.0

        for name, metrics in data.items():
            if 'R' in metrics and metrics['R'] is not None:
                epoch_R.append(metrics['R'].float().mean().item())

            weights = metrics['w'].float()
            epoch_l2 += torch.norm(weights, p=2).item()

            if metrics['g'] is not None:
                grads = metrics['g'].float()
                epoch_grad_norm += torch.norm(grads, p=2).item()

        avg_R.append(sum(epoch_R) / len(epoch_R) if epoch_R else 1.0)
        weight_l2_norms.append(epoch_l2)
        grad_norms.append(epoch_grad_norm)

    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    axs[0].plot(epochs, avg_R, color='blue', label='Mean $R_t$')
    axs[0].set_title('Global Optimizer Reliability ($R_t$)')
    axs[0].set_ylabel('Reliability Score (0.1 - 1.0)')
    axs[0].grid(True)

    axs[1].plot(epochs, weight_l2_norms, color='green', label='$L_2$ Norm')
    axs[1].set_title('Total Model Weight $L_2$ Norm (Weight Decay Impact)')
    axs[1].set_ylabel('$\\sum ||\\theta||_2$')
    axs[1].grid(True)

    axs[2].plot(epochs, grad_norms, color='red', label='Grad Norm')
    axs[2].set_title('Global Gradient Norm')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('$||\\nabla L||_2$')
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig('deltagrad_custom_metrics.png')
    plt.show()


if __name__ == "__main__":
    analyze_deltagrad_runs()
