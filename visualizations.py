import torch

import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr



def get_grad_variance(model, criterion, inputs, labels, num_samples=8):
    grads = []
    model.train() 

    
    # Congelamos a atualização das estatísticas do BatchNorm
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.track_running_stats = False 
    # ---------------------------

    sample_size = max(1, inputs.size(0) // 4) 
    for _ in range(num_samples):
        indices = torch.randperm(inputs.size(0))[:sample_size]
        model.zero_grad()
        outputs = model(inputs[indices])
        loss = criterion(outputs, labels[indices])
        loss.backward()
        all_grads = torch.cat([p.grad.detach().view(-1) for p in model.parameters() if p.grad is not None])
        grads.append(all_grads)
    
    # --- RESTAURAR DEPOIS ---
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.track_running_stats = True
    # -------------------------

    variance = torch.var(torch.stack(grads), dim=0).mean().item()
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
        plot_r_vs_variance_statistical(all_r, all_variance)
    else:
        print("Error: No data available to generate the plot.")


def plot_r_vs_variance(r_values, variance_values):
    plt.figure(figsize=(10, 6))
    
    # Calculate Pearson Correlation
    # This measures the linear relationship between R and Variance
    corr, _ = pearsonr(r_values, variance_values)
    
    scatter = plt.scatter(r_values, variance_values, alpha=0.4, 
                         c=variance_values, cmap='viridis', edgecolors='none')
    
    # Add a text box with the correlation value
    plt.text(0.05, 0.95, f'Pearson Correlation: {corr:.3f}', 
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.title(f"DeltaGrad: R vs Gradient Variance (Correlation: {corr:.3f})", fontsize=14)
    plt.xlabel("Reliability Metric R (Network Average)", fontsize=12)
    plt.ylabel("Real Gradient Variance (p)", fontsize=12)
    
    plt.colorbar(scatter, label='Variance Magnitude')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("deltagrad_r_vs_variance.pdf", bbox_inches='tight')
    plt.savefig("deltagrad_r_vs_variance.png", dpi=300, bbox_inches='tight')

    print("Graphs saved as 'deltagrad_r_vs_variance.pdf' and '.png'")



def plot_r_vs_variance_statistical(r_values, variance_values):
    plt.figure(figsize=(10, 6))
    
    # 1. Calculate Correlation Coefficients
    pearson_val, _ = pearsonr(r_values, variance_values)
    spearman_val, _ = spearmanr(r_values, variance_values)
    
    # 2. Plot using Seaborn
    # lowess=True creates a locally weighted line that follows the data's curve
    sns.regplot(x=r_values, y=variance_values, lowess=True, 
                scatter_kws={'alpha': 0.4, 'color': 'teal', 'edgecolors': 'none'}, 
                line_kws={'color': 'red', 'linewidth': 2, 'label': 'Lowess Trend'})
    
    # 3. Add Statistical Annotation box
    stats_text = (f'Pearson (Linear): {pearson_val:.3f}\n'
                  f'Spearman (Rank): {spearman_val:.3f}')
    
    plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                   fontsize=11, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Formatting
    plt.title("DeltaGrad: Reliability R vs. Gradient Variance Analysis", fontsize=14)
    plt.xlabel("Reliability Metric R (Network Average)", fontsize=12)
    plt.ylabel("Real Gradient Variance (p)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    

    # Save both formats
    plt.savefig("deltagrad_statistical_correlation.png", dpi=300, bbox_inches='tight')
    plt.savefig("deltagrad_statistical_correlation.pdf", bbox_inches='tight')

    print(f"Plot saved. Pearson: {pearson_val:.3f}, Spearman: {spearman_val:.3f}")

def plot_learning_curves(adam_histories, delta_histories):
    """
    Plots the learning curves for multiple runs of Adam and DeltaGrad.
    Shows individual runs with transparency and the mean as a solid line.
    """
    plt.figure(figsize=(10, 6))
    
    # Define epochs based on history length
    epochs = range(1, len(adam_histories[0]) + 1)

    # Plot all Adam curves (Orange)
    for i, history in enumerate(adam_histories):
        # Set label only for the first line to avoid duplicate legend entries
        label = "Adam (Individual)" if i == 0 else None 
        plt.plot(epochs, history, color='orange', alpha=0.3, label=label)
    
    # Plot all DeltaGrad curves (Teal)
    for i, history in enumerate(delta_histories):
        label = "DeltaGrad (Individual)" if i == 0 else None
        plt.plot(epochs, history, color='teal', alpha=0.3, label=label)

    # Calculate and plot the MEAN for each optimizer with a thicker line
    adam_mean = np.mean(adam_histories, axis=0)
    delta_mean = np.mean(delta_histories, axis=0)
    
    plt.plot(epochs, adam_mean, color='darkorange', linewidth=2.5, label="Adam (Mean)")
    plt.plot(epochs, delta_mean, color='darkslategrey', linewidth=2.5, label="DeltaGrad (Mean)")

    # Chart formatting
    plt.title("Learning Curves Comparison: Adam vs DeltaGrad", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save for Overleaf/LaTeX (PDF) and local view (PNG)
    plt.savefig("learning_curves_comparison.pdf", bbox_inches='tight')
    plt.savefig("learning_curves_comparison.png", dpi=300, bbox_inches='tight')

    print("Learning curves plot saved as 'learning_curves_comparison.pdf'")

def load_variance_data(optimizer_name, n_runs=5):
    """
    Loads variance values from text files for a specific optimizer.
    Expects files named: variance_values_{optimizer_name}_run_{i}.txt
    """
    all_runs_data = []
    
    for i in range(1, n_runs + 1):
        # Ensure filenames match your saving convention (e.g., variance_values_Adam_run_1.txt)
        filename = f"variance_values_run_{optimizer_name}_{i}.txt"
        
        if os.path.exists(filename):
            with open(filename, "r") as f:
                data = [float(line.strip()) for line in f if line.strip() and float(line.strip()) > 1e-10]
                all_runs_data.append(data)

        else:
            print(f"Warning: {filename} not found.")

    min_len = min(len(r) for r in all_runs_data)
    all_runs_data = [r[:min_len] for r in all_runs_data]

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


def plot_accuracy_comparison(adam_accs, delta_accs):
    """
    Creates a bar chart comparing the mean accuracy of Adam vs DeltaGrad
    with error bars representing the standard deviation.
    """
    # Calculate Mean and Standard Deviation
    means = [np.mean(adam_accs), np.mean(delta_accs)]
    stds = [np.std(adam_accs), np.std(delta_accs)]
    labels = ['Adam', 'DeltaGrad']
    
    plt.figure(figsize=(8, 6))
    x_pos = np.arange(len(labels))
    
    # Create bars with colors
    bars = plt.bar(x_pos, means, yerr=stds, align='center', 
                   alpha=0.7, color=['orange', 'teal'], 
                   capsize=10, edgecolor='black')

    # Add labels and styling
    plt.ylabel('Final Accuracy (%)', fontsize=12)
    plt.xticks(x_pos, labels, fontsize=12)
    plt.title('Final Accuracy Comparison: Adam vs DeltaGrad', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add text labels on top of the bars for exact values
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')

    # Adjust Y-axis to focus on the relevant accuracy range
    # (e.g., if accuracies are 80-90%, don't start at 0 to show difference better)
    min_acc = min(means) - max(stds) - 5
    plt.ylim(max(0, min_acc), 100)

    # Save outputs
    plt.savefig("accuracy_comparison_bars.pdf", bbox_inches='tight')
    plt.savefig("accuracy_comparison_bars.png", dpi=300, bbox_inches='tight')
     
    print(f"Comparison plot saved. Adam: {means[0]:.2f}% | DeltaGrad: {means[1]:.2f}%")

def calculate_save_metrics(results, optimizer_name):

    mean_acc = np.mean(results)
    std_dev = np.std(results)
    
    std_error = std_dev / np.sqrt(len(results))
    
    print(f"--- Metrics for {optimizer_name} ---")
    print(f"Mean Accuracy: {mean_acc:.2f}%")
    print(f"Standard Deviation: {std_dev:.2f}")
    print(f"95% Confidence Interval: [{mean_acc - 1.96*std_error:.2f}, {mean_acc + 1.96*std_error:.2f}]")

    filename = f"results_{optimizer_name.lower()}.txt"
    
    with open(filename, "w") as f:
        f.write(f"Optimizer: {optimizer_name}\n")
        f.write(f"Final Mean Accuracy: {mean_acc:.2f}%\n")
        f.write(f"Standard Deviation: {std_dev:.4f}\n")
        f.write("-" * 30 + "\n")
        f.write("Individual Runs (Seeds):\n")
        for i, res in enumerate(results):
            f.write(f"Run {i+1}: {res:.2f}%\n")
            
    print(f"Results for {optimizer_name} saved to {filename}")




    