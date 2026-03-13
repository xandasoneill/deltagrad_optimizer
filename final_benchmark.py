import joblib
from DeltaGrad import DeltaGrad
import torch
import torch.optim as optim

from model import ConvNet

from engine import train_model

import time
import numpy as np
# import torch_directml

best_params_deltagrad = joblib.load("best_params_DeltaGrad_fixed_b16_epochs15.pkl")
best_params_adam = joblib.load("best_params_Adam_fixed_b16_epochs15.pkl")



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

        # if torch_directml.is_available():
        #     device = torch_directml.device()
        # elif torch.cuda.is_available():
        #     device = torch.device("cuda")
        # else:
        #     device = torch.device("cpu")

        device = torch.device("cpu")

        model = ConvNet().to(device)
        print(f"Model passed to device:{device}")
        
        if optimizer_name == "DeltaGrad":
            best_params = best_params_deltagrad
            best_params_to_pass = best_params_deltagrad.copy()
            
            # X learning rate here
            best_params_to_pass["lr"] = best_params_to_pass["lr"] *10
            
            if "batch_size" in best_params_to_pass:
                best_params_to_pass.pop("batch_size")
            print(f"Using DeltaGrad with params: {best_params_to_pass}")
            optimizer = DeltaGrad(model.parameters(), **best_params_to_pass)
            
        else:
            best_params = best_params_adam
            best_params_to_pass = best_params_adam.copy()

            # X learning rate here
            best_params_to_pass["lr"] = best_params_to_pass["lr"] *10
            
            if "batch_size" in best_params_to_pass:
                best_params_to_pass.pop("batch_size")
            print(f"Using Adam with params: {best_params_to_pass}")
            optimizer = optim.Adam(model.parameters(), **best_params_to_pass)
        
        batch_size = 16
        histacc, r_values, variance_values, total_net_time, time_stamps, experiment_start_time, device , loss_list = train_model(model, optimizer, optimizer_name, batch=batch_size)
        experiment_start_time = time.ctime(experiment_start_time)
        print(experiment_start_time)
        # Save R and variance values for plotting
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

        end_accuracies.append(histacc[-1])  # Append the last accuracy value from history
        print(f"Run {i+1}: Accuracy = {histacc[-1]:.4f}")

    results = {
        "optimizer": optimizer_name,
        "epochs": 15,
        "batch_size": batch_size,
        "number_runs": n_runs,
        "dataset": "CIFAR-100", 
        "model_name": "ConvNet",
        
        # Históricos (Listas de Listas)
        "acc_history": acc_history,
        "loss_history": loss_history, 
        "r_history": r_history,
        "variance_history": variance_history,
        "all_timestamps": time_stamps_history, 
        
        # Metadados de Execução
        "optimizer_hyperparameters": best_params_to_pass,
        "all_total_times": total_net_time_history,
        "seeds": seeds_used, # Adicionado (boa prática)
        "device": str(device),
        "start_time": experiment_start_time_history
    }

    results_file = f"{optimizer_name}_results_batch{batch_size}_lr{best_params_to_pass['lr']}.pkl"
    joblib.dump(results, results_file)

if __name__ == "__main__":


    #print("Starting benchmark for Adam...")
    #run_benchmark(n_runs=5, optimizer_name="Adam")
    print("Starting benchmark for DeltaGrad...")
    run_benchmark(n_runs=5, optimizer_name="DeltaGrad")

    
