import torch

import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt

import numpy as np



def get_grad_variance(model, criterion, inputs, labels, num_samples=5):
    """Calcula a variância real do gradiente amostrando mini-batches."""
    grads = []
    # Usamos um sub-batch pequeno para calcular a variância local
    for _ in range(num_samples):
        # Amostragem aleatória dentro do batch atual para simular ruído
        indices = torch.randperm(inputs.size(0))[:32] 
        model.zero_grad()
        outputs = model(inputs[indices])
        loss = criterion(outputs, labels[indices])
        loss.backward()
        
        # Flatten de todos os gradientes num vetor só
        all_grads = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
        grads.append(all_grads)
    
    # Variância média entre as amostras
    grads_stack = torch.stack(grads)
    variance = torch.var(grads_stack, dim=0).mean().item()
    return variance


def load_and_plot_results(n_runs=5):
    
    all_r = []
    all_variance = []

    for i in range(1, n_runs + 1):
        r_file = f"r_values_run_DeltaGrad_{i}.txt"
        v_file = f"variance_values_run_DeltaGrad_{i}.txt"

        # Check if files exist to prevent runtime errors
        if os.path.exists(r_file) and os.path.exists(v_file):
            # Load R values: strip whitespace and convert to float
            with open(r_file, "r") as f:
                all_r.extend([float(line.strip()) for line in f if line.strip()])
            
            # Load Variance values: strip whitespace and convert to float
            with open(v_file, "r") as f:
                all_variance.extend([float(line.strip()) for line in f if line.strip()])
            
            print(f"Data from Run {i} successfully loaded.")
        else:
            print(f"Warning: Files for Run {i} were not found.")

    # Proceed to plotting if data was successfully collected
    if all_r and all_variance:
        plot_r_vs_variance(all_r, all_variance)
    else:
        print("Error: No data available to generate the plot.")

def plot_r_vs_variance(r_values, variance_values):
    plt.figure(figsize=(10, 6))
    
    # Scatter plot to analyze the correlation between R and Gradient Variance
    # alpha=0.4 is used to handle point overlap in large datasets
    scatter = plt.scatter(r_values, variance_values, alpha=0.4, 
                         c=variance_values, cmap='viridis', edgecolors='none')
    
    plt.title("DeltaGrad: Analysis of R vs Gradient Variance (Combined Runs)", fontsize=14)
    plt.xlabel("Reliability Metric R (Network Average)", fontsize=12)
    plt.ylabel("Real Gradient Variance", fontsize=12)
    
    # Add a colorbar to indicate variance intensity levels
    plt.colorbar(scatter, label='Variance Magnitude')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save in PDF format for high-quality LaTeX/Overleaf integration
    plt.savefig("deltagrad_r_vs_variance.pdf", bbox_inches='tight')
    
    # Save in PNG format for quick local preview
    plt.savefig("deltagrad_r_vs_variance.png", dpi=300, bbox_inches='tight')

    print("Graphs saved as 'deltagrad_r_vs_variance.pdf' and '.png'")

def plot_convergence(history_acc, optimizer_name, run):

    plt.figure(figsize=(10, 6))
    plt.plot(history_acc, marker='o', label=f'{optimizer_name} Accuracy')
    plt.title(f'Convergence of {optimizer_name} on CIFAR-100', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(range(len(history_acc)))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # Save the convergence plot
    plt.savefig(f"{optimizer_name}_convergence_run_{run}.pdf", bbox_inches='tight')
    plt.savefig(f"{optimizer_name}_convergence_run_{run}.png", dpi=300, bbox_inches='tight')

    print(f"Convergence graphs saved as '{optimizer_name}_convergence_run_{run}.pdf' and '.png'")

def load_variance_data(optimizer_name, n_runs=5):
    """
    Loads variance values from text files for a specific optimizer.
    Expects files named: variance_values_{optimizer_name}_run_{i}.txt
    """
    all_runs_data = []
    
    for i in range(1, n_runs + 1):
        # Ensure filenames match your saving convention (e.g., variance_values_Adam_run_1.txt)
        filename = f"variance_values_{optimizer_name}_run_{i}.txt"
        
        if os.path.exists(filename):
            with open(filename, "r") as f:
                data = [float(line.strip()) for line in f if line.strip()]
                all_runs_data.append(data)
        else:
            print(f"Warning: {filename} not found.")

    # Convert to a 2D numpy array (Runs x Iterations)
    # Note: All runs must have the same number of data points
    return np.array(all_runs_data)

def plot_variance_comparison(n_runs=5):
    # Load data for both optimizers
    # Replace strings with your actual naming convention if different
    delta_data = load_variance_data("DeltaGrad", n_runs)
    adam_data = load_variance_data("Adam", n_runs)

    plt.figure(figsize=(12, 6))

    def plot_with_shading(data, label, color):
        # Calculate mean and standard deviation across runs
        mean_vals = np.mean(data, axis=0)
        std_vals = np.std(data, axis=0)
        iterations = np.arange(len(mean_vals))

        # Plot the main average line
        plt.plot(iterations, mean_vals, label=label, color=color, linewidth=1.5)
        # Add shaded area for variance/uncertainty between runs
        plt.fill_between(iterations, mean_vals - std_vals, mean_vals + std_vals, 
                         color=color, alpha=0.2)

    if delta_data.size > 0:
        plot_with_shading(delta_data, "DeltaGrad", "teal")
    
    if adam_data.size > 0:
        plot_with_shading(adam_data, "Adam", "orange")

    plt.title("Gradient Variance Evolution During Training", fontsize=14)
    plt.xlabel("Measurement Interval (Batches)", fontsize=12)
    plt.ylabel("Real Gradient Variance", fontsize=12)
    plt.yscale('log')  # Variance often spans orders of magnitude
    plt.legend()
    plt.grid(True, which="both", linestyle='--', alpha=0.5)

    # Save outputs for documentation
    plt.savefig("variance_comparison_log.pdf", bbox_inches='tight')
    plt.savefig("variance_comparison_log.png", dpi=300, bbox_inches='tight')
    
    print("Variance comparison plots saved successfully.")
