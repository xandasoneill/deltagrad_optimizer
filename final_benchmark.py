import joblib
from DeltaGrad import DeltaGrad
import torch

from model import ConvNet

from visualizations import load_and_plot_results, plot_accuracy_comparison, plot_variance_comparison, plot_learning_curves, calculate_save_metrics
from engine import train_model

best_params_deltagrad = joblib.load("best_params_DeltaGrad_fixed_b16.pkl")
best_params_adam = joblib.load("best_params_Adam_fixed_b16.pkl")



def run_benchmark(n_runs=5, optimizer_name="DeltaGrad"):

    all_accuracies = []
    all_histories = []
    
    for i in range(n_runs):

        model = ConvNet().to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        
        if optimizer_name == "DeltaGrad":
            best_params = best_params_deltagrad
            best_params_to_pass = best_params_deltagrad.copy()
            
            # Triple learning rate here
            best_params_to_pass["lr"] = best_params_to_pass["lr"] * 3
            
            if "batch_size" in best_params_to_pass:
                best_params_to_pass.pop("batch_size")

            optimizer = DeltaGrad(model.parameters(), **best_params_to_pass)
        else:
            best_params = best_params_adam
            best_params_to_pass = best_params_adam.copy()

            # Triple learning rate here
            best_params_to_pass["lr"] = best_params_to_pass["lr"] * 3
            
            if "batch_size" in best_params_to_pass:
                best_params_to_pass.pop("batch_size")

            optimizer = DeltaGrad(model.parameters(), **best_params_to_pass)
        
        histacc, r_values, variance_values = train_model(model, optimizer, optimizer_name, best_params=best_params, batch=16)

        # Save R and variance values for plotting
        with open(f"r_values_run_{optimizer_name}_{i+1}.txt", "w") as f:
            for r in r_values:
                f.write(f"{r}\n")
        with open(f"variance_values_run_{optimizer_name}_{i+1}.txt", "w") as f:
            for v in variance_values:
                f.write(f"{v}\n")
        
        if optimizer_name == "DeltaGrad":

            print(f"Run {i+1} - Final R values: {r_values[-1] if r_values else 'No R values collected'}")
            print(f"Run {i+1} - Final Gradient Variance: {variance_values[-1] if variance_values else 'No variance values collected'}")

        all_histories.append(histacc)

        all_accuracies.append(histacc[-1])  # Append the last accuracy value from history
        print(f"Run {i+1}: Accuracy = {histacc[-1]:.4f}")
    
    return all_accuracies, all_histories

if __name__ == "__main__":


    print("Starting benchmark for Adam...")
    adam_accuracies, adam_histories = run_benchmark(n_runs=5, optimizer_name="Adam")
    print("Starting benchmark for DeltaGrad...")
    deltagrad_accuracies, deltagrad_histories = run_benchmark(n_runs=5, optimizer_name="DeltaGrad")

    load_and_plot_results()  # Load data from all runs and generate the R vs Variance plot
    plot_variance_comparison(n_runs=5)  # Generate variance comparison plots for both optimizers
    plot_accuracy_comparison(adam_accuracies, deltagrad_accuracies)  # Generate accuracy comparison plot
    plot_learning_curves(adam_histories, deltagrad_histories)  # Generate learning curves comparison plot  

    calculate_save_metrics(adam_accuracies, "DeltaGrad")
    calculate_save_metrics(deltagrad_accuracies, "Adam") 

