import joblib
from DeltaGrad import DeltaGrad
import torch

from model import ConvNet

from visualizations import load_and_plot_results, get_grad_variance, plot_convergence, plot_variance_comparison
from engine import train_model

best_params_deltagrad = joblib.load("best_params_DeltaGrad.pkl")
best_params_adam = joblib.load("best_params_Adam.pkl")



def run_benchmark(n_runs=5, optimizer_name="DeltaGrad"):

    all_accuracies = []
    
    for i in range(n_runs):

        model = ConvNet().to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        
        if optimizer_name == "DeltaGrad":
            best_params = best_params_deltagrad
            optimizer = DeltaGrad(model.parameters(), **best_params)
        else:
            best_params = best_params_adam
            optimizer = torch.optim.Adam(model.parameters(), **best_params)
        
        histacc, r_values, variance_values = train_model(model, optimizer, optimizer_name, best_params=best_params)

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

        plot_convergence(histacc, optimizer_name, i+1)



        all_accuracies.append(histacc[-1])  # Append the last accuracy value from history
        print(f"Run {i+1}: Accuracy = {histacc[-1]:.4f}")
    
    return all_accuracies

if __name__ == "__main__":


    print("Starting benchmark for Adam...")
    adam_accuracies = run_benchmark(n_runs=5, optimizer_name="Adam")
    print("Starting benchmark for DeltaGrad...")
    deltagrad_accuracies = run_benchmark(n_runs=5, optimizer_name="DeltaGrad")

    load_and_plot_results()  # Load data from all runs and generate the R vs Variance plot
    plot_variance_comparison(n_runs=5)  # Generate variance comparison plots for both optimizers