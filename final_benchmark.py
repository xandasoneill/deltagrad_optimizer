# Copyright 2026 Alexandre de Abreu O'Neill Mendes

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import joblib
from DeltaGrad import DeltaGrad
import torch
import torch.optim as optim
from model import ConvNet
from engine import train_model
import time
import numpy as np

# Load pre-tuned best parameters from Optuna trials
best_params_deltagrad = joblib.load("best_params_DeltaGrad_fixed_b512_epochs50.pkl")
best_params_adam = joblib.load("best_params_Adam_fixed_b512_epochs50.pkl")

def run_benchmark(n_runs=5, optimizer_name="DeltaGrad"):
    """
    Runs a series of training sessions to evaluate the stability and performance 
    of a specific optimizer over multiple random seeds.
    """
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
        # Generate and track seed for reproducibility
        current_seed = torch.seed()
        seeds_used.append(current_seed)
        
        torch.manual_seed(current_seed)
        np.random.seed(current_seed % (2**32))

        # Device configuration (Standardized to CPU as per user request)
        device = torch.device("cpu")

        model = ConvNet().to(device)
        print(f"Model passed to device: {device}")
        
        # Select optimizer and load corresponding hyperparameters
        if optimizer_name == "DeltaGrad":
            best_params = best_params_deltagrad
            best_params_to_pass = best_params_deltagrad.copy()
            
            # Learning rate scaling can be adjusted here if needed
            best_params_to_pass["lr"] = best_params_to_pass["lr"] 
            
            # Clean dictionary for the class constructor
            if "batch_size" in best_params_to_pass:
                best_params_to_pass.pop("batch_size")
            print(f"Using DeltaGrad with params: {best_params_to_pass}")
            optimizer = DeltaGrad(model.parameters(), **best_params_to_pass)
            
        else:
            best_params = best_params_adam
            best_params_to_pass = best_params_adam.copy()

            # Learning rate scaling can be adjusted here if needed
            best_params_to_pass["lr"] = best_params_to_pass["lr"] 
            
            if "batch_size" in best_params_to_pass:
                best_params_to_pass.pop("batch_size")
            print(f"Using Adam with params: {best_params_to_pass}")
            optimizer = optim.Adam(model.parameters(), **best_params_to_pass)
        
        # Experiment settings
        batch_size = 512
        n_epochs = 50
        
        # Execute training loop via engine.py
        histacc, r_values, variance_values, total_net_time, time_stamps, experiment_start_time, device, loss_list = train_model(
            model, 
            optimizer, 
            optimizer_name, 
            epochs=n_epochs, 
            batch=batch_size
        )
        
        # Format start time for logging
        experiment_start_time = time.ctime(experiment_start_time)
        print(experiment_start_time)

        # Collect histories for analysis and plotting
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

    # Consolidate all data into a single results dictionary
    results = {
        "optimizer": optimizer_name,
        "epochs": n_epochs,
        "batch_size": batch_size,
        "number_runs": n_runs,
        "dataset": "CIFAR-100", 
        "model_name": "ConvNet",
        
        # Historical Data (List of Lists)
        "acc_history": acc_history,
        "loss_history": loss_history, 
        "r_history": r_history,
        "variance_history": variance_history,
        "all_timestamps": time_stamps_history, 
        
        # Execution Metadata
        "optimizer_hyperparameters": best_params_to_pass,
        "all_total_times": total_net_time_history,
        "seeds": seeds_used, 
        "device": str(device),
        "start_time": experiment_start_time_history
    }

    # Save results to a pkl file for visualization.py
    results_file = f"{optimizer_name}_results_batch{batch_size}_lr{best_params_to_pass['lr']}.pkl"
    joblib.dump(results, results_file)
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    # Execute full benchmark suite
    print("Starting benchmark for Adam...")
    run_benchmark(n_runs=5, optimizer_name="Adam")
    
    print("Starting benchmark for DeltaGrad...")
    run_benchmark(n_runs=5, optimizer_name="DeltaGrad")