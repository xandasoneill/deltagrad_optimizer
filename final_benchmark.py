import joblib
from DeltaGrad import DeltaGrad
import torch
import torch.optim as optim

from model import ConvNet

from visualizations import load_and_plot_results, plot_accuracy_comparison, plot_variance_comparison, plot_learning_curves, calculate_save_metrics
from engine import train_model

best_params_deltagrad = joblib.load("best_params_DeltaGrad_fixed_b16.pkl")
best_params_adam = joblib.load("best_params_Adam_fixed_b16.pkl")



def run_benchmark(n_runs=5, optimizer_name="DeltaGrad"):

    end_accuracies = []
    acc_history = []
    r_history = []
    variance_history = []

    
    for i in range(n_runs):

        model = ConvNet().to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        
        if optimizer_name == "DeltaGrad":
            best_params = best_params_deltagrad
            best_params_to_pass = best_params_deltagrad.copy()
            
            # Triple learning rate here
            #best_params_to_pass["lr"] = best_params_to_pass["lr"] * 3
            
            if "batch_size" in best_params_to_pass:
                best_params_to_pass.pop("batch_size")

            optimizer = DeltaGrad(model.parameters(), **best_params_to_pass)
        else:
            best_params = best_params_adam
            best_params_to_pass = best_params_adam.copy()

            # Triple learning rate here
            #best_params_to_pass["lr"] = best_params_to_pass["lr"] * 3
            
            if "batch_size" in best_params_to_pass:
                best_params_to_pass.pop("batch_size")

            optimizer = optim.Adam(model.parameters(), **best_params_to_pass)
        
        batch_size = best_params["batch_size"]
        histacc, r_values, variance_values = train_model(model, optimizer, optimizer_name, best_params=best_params, batch=batch_size)

        # Save R and variance values for plotting
        r_history.append(r_values)

        variance_history.append(variance_values)
        
        if optimizer_name == "DeltaGrad":

            print(f"Run {i+1} - Final R values: {r_values[-1] if r_values else 'No R values collected'}")
            print(f"Run {i+1} - Final Gradient Variance: {variance_values[-1] if variance_values else 'No variance values collected'}")

        acc_history.append(histacc)

        end_accuracies.append(histacc[-1])  # Append the last accuracy value from history
        print(f"Run {i+1}: Accuracy = {histacc[-1]:.4f}")

    results = {
        "opimizer" : optimizer_name,
        "acc_history": acc_history,
        "r_history": r_history,
        "variance_history": variance_history,


    }

    results_file = f"{optimizer_name}_results_batch{batch_size}_lr{best_params_to_pass["lr"]}.pkl"
    joblib.dump(results, results_file)

    load_and_plot_results()  # Load data from all runs and generate the R vs Variance plot
    plot_variance_comparison(n_runs)  # Generate variance comparison plots for both optimizers

    calculate_save_metrics(adam_accuracies, "Adam")
    calculate_save_metrics(deltagrad_accuracies, "Deltagrad") 
    
    
    return end_accuracies, acc_history, r_history, variance_history

if __name__ == "__main__":


    print("Starting benchmark for Adam...")
    adam_final_accuracies, adam_acc_history = run_benchmark(n_runs=5, optimizer_name="Adam")
    print("Starting benchmark for DeltaGrad...")
    deltagrad_final_accuracies, deltagrad_acc_history = run_benchmark(n_runs=5, optimizer_name="DeltaGrad")

    plot_accuracy_comparison(adam_final_accuracies, deltagrad_final_accuracies)  # Generate accuracy comparison plot
    plot_learning_curves(adam_acc_history, deltagrad_acc_history)  # Generate learning curves comparison plot  


