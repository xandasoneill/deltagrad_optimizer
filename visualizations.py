import torch

import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

import joblib

from matplotlib.ticker import MaxNLocator 



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


def load_and_plot_results(results_deltagrad, results_adam):

    
    v_history = results_deltagrad["variance_history"] # Lista de listas (5 x iterações)
    r_history = results_deltagrad["r_history"]        # Lista de listas (5 x iterações)

    if len(v_history) == len(r_history):
        # 1. Gera os gráficos individuais de cada run
        for i in range(len(v_history)):
            plot_individual_run(r_history[i], v_history[i], run_id=i+1)
        
        # 2. Gera o "Killer Plot" com todas as runs juntas
        plot_all_runs_combined(r_history, v_history)
    else:
        print("Variance history is not the same size as R history!")

        

    plot_accuracy_evolution(results_deltagrad, results_adam)
    plot_variance_comparison(results_deltagrad, results_adam)
    plot_combined_loss(results_adam, results_deltagrad, adam_label="Adam", dg_label="DeltaGrad")



def plot_all_runs_combined(r_history_list, v_history_list):

    plt.figure(figsize=(10, 6))
    
    # Paleta profissional expandida
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    r_flat = [item for sublist in r_history_list for item in sublist]
    v_flat = [item for sublist in v_history_list for item in sublist]
    
    # Cálculo da Correlação e P-Value Global
    # O p-value testa a hipótese nula de que não há correlação
    global_r, global_p = pearsonr(r_flat, v_flat)
    
    all_pearsons = []
    for i, (r_vals, v_vals) in enumerate(zip(r_history_list, v_history_list)):
        corr, _ = pearsonr(r_vals, v_vals)
        all_pearsons.append(corr)
        
        current_color = colors[i % len(colors)]
        plt.scatter(r_vals, v_vals, alpha=0.12, color=current_color, edgecolors='none', 
                    label=f'Run {i+1} (r={corr:.2f})')
        
        sns.regplot(x=np.array(r_vals), y=np.array(v_vals), scatter=False, lowess=True, 
                    line_kws={'color': current_color, 'linewidth': 1.2, 'alpha': 0.7})

    # Tendência Global
    sns.regplot(x=np.array(r_flat), y=np.array(v_flat), scatter=False, lowess=True, 
                line_kws={'color': 'black', 'linewidth': 3, 'ls': '--', 'label': 'Global Trend'})

    # Formatação do P-value para notação científica se for muito pequeno
    p_text = f"{global_p:.2e}" if global_p < 0.001 else f"{global_p:.4f}"

    mean_corr = np.mean(all_pearsons)

    plt.xlabel("Reliability Metric R (Network Average)", fontsize=25)
    plt.ylabel("Real Gradient Variance (p)", fontsize=25)
    plt.legend(loc='upper right', 
           fontsize=25, 
           ncol=2 if len(r_history_list) <= 5 else 3,
           columnspacing=0.5,  
           handletextpad=0.2,   
           borderaxespad=0.2)   
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(fontsize=25) # Aumenta os números do eixo X
    plt.yticks(fontsize=25) # Aumenta os números do eixo Y

    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=4))
    
    plt.tight_layout()
    # Gravação obrigatória em PDF para o paper
    plt.savefig("deltagrad_combined_correlation.pdf", bbox_inches='tight')
    plt.savefig("deltagrad_combined_correlation.png", dpi=300, bbox_inches='tight')

    print(f"Gráfico gerado: r={global_r:.3f}, p={p_text}")



def plot_individual_run(r_values, variance_values, run_id):
    plt.figure(figsize=(10, 6))
    
    # Calcular Pearson e p-value
    pearson_val, p_val = pearsonr(r_values, variance_values)
    
    # Formatação do p-value (notação científica se for muito pequeno)
    p_text = f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.4f}"
    
    # Plot com linha Lowess para capturar a tendência não linear
    sns.regplot(x=r_values, y=variance_values, lowess=True, 
                scatter_kws={'alpha': 0.4, 'color': 'teal', 'edgecolors': 'none'}, 
                line_kws={'color': 'red', 'linewidth': 2})
    
    # Título com r e p
  
    plt.xlabel("Reliability Metric R", fontsize=10)
    plt.ylabel("Gradient Variance", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Guardar em ambos os formatos
    plt.savefig(f"deltagrad_run_{run_id}_stats.pdf", bbox_inches='tight')
    plt.savefig(f"deltagrad_run_{run_id}_stats.png", dpi=150, bbox_inches='tight')
    
    plt.close() # Libertar memória do GPU/RAM
    print(f"Individual run {run_id} saved (p={p_text})")



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
    plt.xlabel("Epochs", fontsize=25)
    plt.ylabel("Accuracy (%)", fontsize=25)
    plt.legend(loc='lower right', fontsize = 22)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=25) # Aumenta os números do eixo X
    plt.yticks(fontsize=25)
    
    # Save for Overleaf/LaTeX (PDF) and local view (PNG)
    plt.savefig("learning_curves_comparison.pdf", bbox_inches='tight')
    plt.savefig("learning_curves_comparison.png", dpi=300, bbox_inches='tight')

    print("Learning curves plot saved as 'learning_curves_comparison.pdf'")

def plot_accuracy_comparison(adam_accs, delta_accs):
    """
    Creates a bar chart comparing the mean accuracy of Adam vs DeltaGrad
    with error bars representing the standard deviation.
    """
    # Calculate Mean and Standard Deviation
    means = [np.mean(adam_accs), np.mean(delta_accs)]
    stds = [np.std(adam_accs), np.std(delta_accs)]
    labels = ['Adam', 'DeltaGrad']
    
    plt.figure(figsize=(10, 6))
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




def plot_accuracy_evolution(results_dg, results_adam):

    plt.figure(figsize=(10, 6))
    
    def get_stats(history):
        matrix = np.array(history)
        mean = np.mean(matrix, axis=0)
        std = np.std(matrix, axis=0)
        return mean, std

    # Calcular estatísticas
    dg_mean, dg_std = get_stats(results_dg["acc_history"])
    adam_mean, adam_std = get_stats(results_adam["acc_history"])
    epochs = np.arange(1, len(dg_mean) + 1)

    # Plot DeltaGrad
    plt.plot(epochs, dg_mean, label='DeltaGrad (Mean)', color='teal', linewidth=2)
    plt.fill_between(epochs, dg_mean - dg_std, dg_mean + dg_std, color='teal', alpha=0.2)

    # Plot Adam
    plt.plot(epochs, adam_mean, label='Adam (Mean)', color='orange', linewidth=2)
    plt.fill_between(epochs, adam_mean - adam_std, adam_mean + adam_std, color='orange', alpha=0.2)

    # Adicionar média dos desvios como texto
    avg_std_dg = np.mean(dg_std)
    avg_std_adam = np.mean(adam_std)
    stats_text = (f'Avg Std Dev:\nDG: {avg_std_dg:.2f}%\nAdam: {avg_std_adam:.2f}%')
    plt.gca().text(0.02, 0.05, stats_text, transform=plt.gca().transAxes, fontsize=25,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))


    plt.xlabel("Epoch", fontsize=25)
    plt.ylabel("Validation Accuracy (%)", fontsize=25)
    plt.legend(loc='lower right', fontsize = 23)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(fontsize=25) # Aumenta os números do eixo X
    plt.yticks(fontsize=25)
    
    plt.savefig("accuracy_stability_comparison.pdf", bbox_inches='tight')
    plt.savefig("accuracy_stability_comparison.png", dpi=300, bbox_inches='tight')


def plot_variance_comparison(results_dg, results_adam):
        
    plt.figure(figsize=(10, 6))
    
    # Cores para o gráfico
    dg_color, adam_color = 'teal', 'orange'
    
    def process_and_plot(history_list, label, color):
        # Converter para matriz e filtrar valores < 10^-10
        matrix = np.array(history_list)
        matrix = np.where(matrix < 1e-10, 1e-10, matrix)
        
        # Plot das 5 runs individuais (suaves)
        for i in range(matrix.shape[0]):
            plt.plot(matrix[i], color=color, alpha=0.15, linewidth=1)
        
        # Plot da média (grossa)
        mean_vals = np.mean(matrix, axis=0)
        plt.plot(mean_vals, color=color, linewidth=2.5, label=f'{label} (Mean)')
        
        return np.mean(matrix) # Retorna a média global de todas as runs/épocas

    # Gerar os plots e calcular médias globais
    avg_var_dg = process_and_plot(results_dg["variance_history"], "DeltaGrad", dg_color)
    avg_var_adam = process_and_plot(results_adam["variance_history"], "Adam", adam_color)

    # Configurações do gráfico
    plt.yscale('log') # Escala logarítmica é vital para variância
    plt.xlabel("Batches", fontsize=25)
    plt.ylabel("Real Gradient Variance", fontsize=22)
    plt.xticks(fontsize=25) # Aumenta os números do eixo X
    plt.yticks(fontsize=25)
    
    # Adicionar as médias globais na legenda ou texto
    stats_text = (f'Global Mean Variance:\n'
                    f'DeltaGrad: {avg_var_dg:.2e}\n'
                    f'Adam: {avg_var_adam:.2e}')
    plt.gca().text(0.02, 0.05, stats_text, transform=plt.gca().transAxes, 
                    fontsize=20,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.legend(loc='lower right', fontsize=20)
    plt.grid(True, which="both", linestyle='--', alpha=0.4)
    
    # Guardar em PDF e PNG
    plt.savefig("variance_comparison_stress_test.pdf", bbox_inches='tight')
    plt.savefig("variance_comparison_stress_test.png", dpi=300, bbox_inches='tight')
    print(f"Variance plot saved. DG Avg: {avg_var_dg:.2e}, Adam Avg: {avg_var_adam:.2e}")



def plot_mean_time_per_epoch(adam_runs_stamps, dg_runs_stamps, bin_size=5):

    def get_all_durations(runs_stamps):
        all_runs_durations = []
        for stamps in runs_stamps:
            # Calcula durações individuais por época
            durations = [stamps[0]] + [stamps[i] - stamps[i-1] for i in range(1, len(stamps))]
            all_runs_durations.append(durations)
        return np.array(all_runs_durations)

    # 1. Obter matrizes originais (Runs x Epochs)
    adam_matrix = get_all_durations(adam_runs_stamps)
    dg_matrix = get_all_durations(dg_runs_stamps)
    
    # Calcular médias globais para o texto de overhead (baseado em todos os dados)
    global_avg_adam = np.mean(adam_matrix)
    global_avg_dg = np.mean(dg_matrix)

    # 2. Agrupar dados em janelas de 'bin_size'
    num_epochs = adam_matrix.shape[1]
    num_bins = num_epochs // bin_size
    
    # Redimensionar e calcular a média por bloco de 5 épocas
    adam_binned = adam_matrix[:, :num_bins*bin_size].reshape(adam_matrix.shape[0], num_bins, bin_size).mean(axis=2)
    dg_binned = dg_matrix[:, :num_bins*bin_size].reshape(dg_matrix.shape[0], num_bins, bin_size).mean(axis=2)
    
    # Médias e desvios por bin
    adam_mean = np.mean(adam_binned, axis=0)
    adam_std = np.std(adam_binned, axis=0)
    dg_mean = np.mean(dg_binned, axis=0)
    dg_std = np.std(dg_binned, axis=0)

    # Definir os centros das barras (ex: épocas 5, 10, 15...)
    bin_centers = np.arange(bin_size, (num_bins + 1) * bin_size, bin_size)

    plt.figure(figsize=(12, 7)) 
    sns.set_style("whitegrid")
    
    bar_width = bin_size * 0.35 # Largura proporcional ao intervalo

    # 3. Plot das barras agrupadas
    plt.bar(bin_centers - bar_width/2, adam_mean, bar_width, yerr=adam_std,
            label='Adam', color='#ff7f0e', alpha=0.8, capsize=5)
    
    plt.bar(bin_centers + bar_width/2, dg_mean, bar_width, yerr=dg_std,
            label='DeltaGrad', color='#008080', alpha=0.8, capsize=5)

    # Linhas de média global
    plt.axhline(y=global_avg_adam, color='#ff7f0e', linestyle='--', alpha=0.6, linewidth=2)
    plt.axhline(y=global_avg_dg, color='#008080', linestyle='--', alpha=0.6, linewidth=2)

    plt.xlabel('Epoch Intervals', fontsize=22, fontweight='bold')
    plt.ylabel('Avg Time per Epoch (s)', fontsize=22, fontweight='bold')
    
    plt.xticks(bin_centers, [f"{i-bin_size+1}-{i}" for i in bin_centers], fontsize=22)
    plt.yticks(fontsize=22)
   
    stats_text = (f'Adam Global Avg: {global_avg_adam:.2f}s\n'
                  f'DeltaGrad Global Avg: {global_avg_dg:.2f}s')

    plt.gca().text(0.98, 0.35, stats_text, 
                transform=plt.gca().transAxes, fontsize=25, 
                horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.9))
    
    overhead = (global_avg_dg / global_avg_adam - 1) * 100
    plt.text(0.02, 0.05, f"Avg Overhead: {overhead:.2f}%", 
             transform=plt.gca().transAxes, fontsize=25, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='#008080'))

    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.legend(fontsize=20, loc='lower right')
    plt.tight_layout()
    
    plt.savefig('time_per_epoch.pdf', bbox_inches='tight')

def plot_combined_loss(adam_results, dg_results, adam_label="Adam", dg_label="DeltaGrad"):
    """
    Plots training loss for both Adam and DeltaGrad on a single chart.
    Now receives result dictionaries directly.
    """
    
    # 1. Extract and convert to NumPy arrays to enable .shape and math operations
    # adam_results["loss_history"] is likely a list of lists, np.array() fixes it.
    adam_data = np.array(adam_results["loss_history"])
    dg_data = np.array(dg_results["loss_history"])
    
    # 2. Get dimensions (Number of epochs is the second dimension)
    num_epochs = adam_data.shape[1]
    epochs = np.arange(1, num_epochs + 1)
    
    # Colors for consistency
    colors = {'Adam': '#ff7f0e', 'DeltaGrad': '#008080'}
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # 3. Plotting logic for both optimizers
    for data, label, color in [(adam_data, adam_label, colors['Adam']), 
                               (dg_data, dg_label, colors['DeltaGrad'])]:
        
        # Calculate statistics across runs (axis=0)
        mean_loss = np.mean(data, axis=0)
        std_loss = np.std(data, axis=0)
        
        # Plot individual runs (faint lines)
        for run in data:
            plt.plot(epochs, run, alpha=0.08, color=color, linewidth=1)
            
        # Plot Global Mean
        plt.plot(epochs, mean_loss, color=color, linewidth=3.5, label=f'{label} (Mean)')
        
        # Shaded area for standard deviation (Stability)
        plt.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, 
                         color=color, alpha=0.2, label=f'{label} $\pm$ Std Dev')

    # 4. Scientific Formatting
    plt.xlabel('Epoch', fontsize=25, fontweight='bold')
    plt.ylabel('Training Loss', fontsize=25, fontweight='bold')
 
    # Ensure epoch ticks are integers and visible
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    
    # Legend
    plt.legend(fontsize=25, loc='upper right', frameon=True, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    # 5. Save outputs
    plt.savefig("loss_comparison_combined.png", dpi=300, bbox_inches='tight')
    plt.savefig("loss_comparison_combined.pdf", bbox_inches='tight')