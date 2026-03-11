import torch

import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

import joblib



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


def load_and_plot_results(results_file_deltagrad, results_file_adam):


    if os.path.exists(results_file_deltagrad):

        results_deltagrad = joblib.load(results_file_deltagrad)
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

    else:

        print("Results file path does not exist!")

    if os.path.exists(results_file_adam):

        results_adam = joblib.load(results_file_adam)

    plot_accuracy_evolution(results_deltagrad, results_adam)
    plot_variance_comparison(results_file_adam, results_file_deltagrad)



def plot_all_runs_combined(r_history_list, v_history_list):

    plt.figure(figsize=(12, 7))
    
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
    plt.title(f"DeltaGrad Multi-Run Analysis: Reliability R vs Variance\n"
              f"Global r: {global_r:.3f} | p-value: {p_text}", fontsize=14)
    
    plt.xlabel("Reliability Metric R (Network Average)", fontsize=12)
    plt.ylabel("Real Gradient Variance (p)", fontsize=12)
    plt.legend(loc='upper right', fontsize=9, ncol=2 if len(r_history_list) <= 5 else 3)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Gravação obrigatória em PDF para o paper
    plt.savefig("deltagrad_combined_correlation.pdf", bbox_inches='tight')
    plt.savefig("deltagrad_combined_correlation.png", dpi=300, bbox_inches='tight')

    print(f"Gráfico gerado: r={global_r:.3f}, p={p_text}")



def plot_individual_run(r_values, variance_values, run_id):
    plt.figure(figsize=(8, 5))
    
    # Calcular Pearson e p-value
    pearson_val, p_val = pearsonr(r_values, variance_values)
    
    # Formatação do p-value (notação científica se for muito pequeno)
    p_text = f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.4f}"
    
    # Plot com linha Lowess para capturar a tendência não linear
    sns.regplot(x=r_values, y=variance_values, lowess=True, 
                scatter_kws={'alpha': 0.4, 'color': 'teal', 'edgecolors': 'none'}, 
                line_kws={'color': 'red', 'linewidth': 2})
    
    # Título com r e p
    plt.title(f"Run {run_id}: R vs. Variance\n(r={pearson_val:.3f}, p={p_text})", fontsize=12)
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
    plt.title("Learning Curves Comparison: Adam vs DeltaGrad", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
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
    plt.gca().text(0.02, 0.8, stats_text, transform=plt.gca().transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.title("Training Stability: Accuracy Evolution (5 Runs)", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation Accuracy (%)", fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig("accuracy_stability_comparison.pdf", bbox_inches='tight')
    plt.savefig("accuracy_stability_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_variance_comparison(results_file1, results_file2):
        
        # Lógica de atribuição correta
        if results_file1["optimizer"] == "DeltaGrad":
            results_dg, results_adam = results_file1, results_file2
        else:
            results_dg, results_adam = results_file2, results_file1

        plt.figure(figsize=(12, 7))
        
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
        plt.title("Gradient Variance Evolution: DeltaGrad vs Adam", fontsize=14)
        plt.xlabel("Measurement Interval (Batches)", fontsize=12)
        plt.ylabel("Real Gradient Variance (log scale)", fontsize=12)
        
        # Adicionar as médias globais na legenda ou texto
        stats_text = (f'Global Mean Variance:\n'
                      f'DeltaGrad: {avg_var_dg:.2e}\n'
                      f'Adam: {avg_var_adam:.2e}')
        plt.gca().text(0.02, 0.05, stats_text, transform=plt.gca().transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.legend(loc='upper left')
        plt.grid(True, which="both", linestyle='--', alpha=0.4)
        
        # Guardar em PDF e PNG
        plt.savefig("variance_comparison_stress_test.pdf", bbox_inches='tight')
        plt.savefig("variance_comparison_stress_test.png", dpi=300, bbox_inches='tight')
        print(f"Variance plot saved. DG Avg: {avg_var_dg:.2e}, Adam Avg: {avg_var_adam:.2e}")



def plot_time_per_epoch_comparison(adam_cumulative_stamps, dg_cumulative_stamps):
    """
    Transforma timestamps acumulados em tempo por época e plota a comparação.
    """
    def get_durations(stamps):
        # Calcula a diferença entre stamps consecutivos: [s0, s1-s0, s2-s1, ...]
        return [stamps[0]] + [stamps[i] - stamps[i-1] for i in range(1, len(stamps))]

    adam_durations = get_durations(adam_cumulative_stamps)
    dg_durations = get_durations(dg_cumulative_stamps)
    epochs = np.arange(1, len(adam_durations) + 1)

    plt.figure(figsize=(10, 6))
    
    # Plot de barras lado a lado
    bar_width = 0.35
    plt.bar(epochs - bar_width/2, adam_durations, bar_width, label='Adam', color='orange', alpha=0.8)
    plt.bar(epochs + bar_width/2, dg_durations, bar_width, label='DeltaGrad', color='teal', alpha=0.8)

    # Linhas de média para referência visual rápida
    avg_adam = np.mean(adam_durations)
    avg_dg = np.mean(dg_durations)
    plt.axhline(y=avg_adam, color='darkorange', linestyle='--', alpha=0.6, label=f'Avg Adam ({avg_adam:.1f}s)')
    plt.axhline(y=avg_dg, color='darkslategrey', linestyle='--', alpha=0.6, label=f'Avg DG ({avg_dg:.1f}s)')

    # Configurações do gráfico
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Time per Epoch (seconds)', fontsize=12)
    plt.title('Computational Efficiency: Time per Epoch Comparison', fontsize=14)
    plt.xticks(epochs)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) # Legenda fora para não tapar as barras
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # Cálculo do Overhead para o texto
    overhead_pct = (avg_dg / avg_adam - 1) * 100
    stats_text = f"Avg Overhead: {overhead_pct:.1f}%"
    plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, 
             fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('time_per_epoch_comparison.png', dpi=300)
    plt.savefig('time_per_epoch_comparison.pdf', bbox_inches='tight')