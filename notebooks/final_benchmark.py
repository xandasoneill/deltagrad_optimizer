import joblib
from DeltaGrad import DeltaGrad
import torch
import torch.optim as optim
from model import ConvNet
from engine import train_model
import time
import numpy as np
import os

# Load pre-tuned best parameters from Optuna trials
best_params_deltagrad = joblib.load("../best_params/best_params_DeltaGrad_fixed_b16_epochs15.pkl")
best_params_adam = joblib.load("../best_params/best_params_Adam_fixed_b16_epochs15.pkl")

def run_benchmark(n_runs=5, optimizer_name="DeltaGrad"):
    end_accuracies = []
    acc_history = []
    r_history = []
    variance_history = []
    total_net_time_history = []
    time_stamps_history = []
    experiment_start_time_history = []
    loss_history = []
    seeds_used = []

    for i in range(n_runs):
        current_seed = torch.seed()
        seeds_used.append(current_seed)
        
        torch.manual_seed(current_seed)
        np.random.seed(current_seed % (2**32))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = ConvNet().to(device)
        print(f"Model passed to device: {device}")
        
        if optimizer_name == "DeltaGrad":
            best_params = best_params_deltagrad
            best_params_to_pass = best_params_deltagrad.copy()
            best_params_to_pass["lr"] = best_params_to_pass["lr"] * best_params_to_pass["gamma"]
            best_params_to_pass.pop("gamma", None)
            
            if "batch_size" in best_params_to_pass:
                best_params_to_pass.pop("batch_size")
            print(f"Using DeltaGrad with params: {best_params_to_pass}")
            optimizer = DeltaGrad(model.parameters(), **best_params_to_pass)
            
        else:
            best_params = best_params_adam
            best_params_to_pass = best_params_adam.copy()
            best_params_to_pass["lr"] = best_params_to_pass["lr"]
            
            if "batch_size" in best_params_to_pass:
                best_params_to_pass.pop("batch_size")
            print(f"Using Adam with params: {best_params_to_pass}")
            optimizer = optim.Adam(model.parameters(), **best_params_to_pass)
        
        batch_size = 16
        n_epochs = 50
        
        histacc, r_values, variance_values, total_net_time, time_stamps, experiment_start_time, device, loss_list = train_model(
            model, 
            optimizer, 
            optimizer_name, 
            epochs=n_epochs, 
            batch=batch_size
        )
        
        experiment_start_time = time.ctime(experiment_start_time)
        print(experiment_start_time)

        r_history.append(r_values)
        variance_history.append(variance_values)
        total_net_time_history.append(total_net_time)
        time_stamps_history.append(time_stamps)
        experiment_start_time_history.append(experiment_start_time)
        loss_history.append(loss_list)

        if optimizer_name == "DeltaGrad":
            print(f"Run {i+1} - Final R values: {r_values[-1] if r_values else 'No R values collected'}")
            print(f"Run {i+1} - Final Gradient Variance: {variance_values[-1] if variance_values else 'No variance values collected'}")

        acc_history.append(histacc)
        end_accuracies.append(histacc[-1]) 
        print(f"Run {i+1}: Accuracy = {histacc[-1]:.4f}")

    results = {
        "optimizer": optimizer_name,
        "epochs": n_epochs,
        "batch_size": batch_size,
        "number_runs": n_runs,
        "dataset": "CIFAR-100", 
        "model_name": "ConvNet",
        "acc_history": acc_history,
        "loss_history": loss_history, 
        "r_history": r_history,
        "variance_history": variance_history,
        "all_timestamps": time_stamps_history, 
        "optimizer_hyperparameters": best_params_to_pass,
        "all_total_times": total_net_time_history,
        "seeds": seeds_used, 
        "device": str(device),
        "start_time": experiment_start_time_history
    }

    # Guarda localmente e depois envia uma cópia de segurança para o Google Drive
    results_file = f"/content/{optimizer_name}_results_batch{batch_size}_lr{best_params_to_pass['lr']}.pkl"
    joblib.dump(results, results_file)
    print(f"Results saved to {results_file}")
    
    # Faz backup para não perderes o ficheiro se o Colab desligar
    os.system(f"cp {results_file} /content/drive/MyDrive/DeltaGrad_domain/notebooks/results")

if __name__ == "__main__":
    
    print("Starting benchmark for Adam...")
    run_benchmark(n_runs=1, optimizer_name="Adam")
    
    print("Starting benchmark for DeltaGrad...")
    run_benchmark(n_runs=1, optimizer_name="DeltaGrad")
