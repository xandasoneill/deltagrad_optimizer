import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import optuna
import joblib
from model import ConvNet  
from DeltaGrad import DeltaGrad
import os
import torch_directml

def train_model(trial, model, optimizer, epochs=15):

    if torch_directml.is_available():
        device = torch_directml.device()
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    device = torch.device("cpu")
    
    criterion = nn.CrossEntropyLoss()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 1. Load and Split Data (Preventing Data Leakage)
    # Using CIFAR-100 for training.
    full_trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    
    # Reserve 5000 images for validation (Optuna only evaluates these to decide the trial score) 
    train_subset, val_subset = random_split(
        full_trainset, [45000, 5000], 
        generator=torch.Generator().manual_seed(42)
    )

    # Fixed batch size for reproducibility in the study 
    batch_size = trial.suggest_int("batch_size", 16, 16)
    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        # TRAINING PHASE
        model.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # VALIDATION PHASE (Prevents overfitting during tuning) 
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * val_correct / val_total
        
        # Report validation accuracy to Optuna for pruning decisions 
        trial.report(val_accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return val_accuracy

def objective(trial, optimizer_name):
    # Set seed for total reproducibility between trials 
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ConvNet().to(device)

    if optimizer_name == "Adam":
        # Search space for standard Adam 
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        # Hyperparameters for DeltaGrad 
        lr = trial.suggest_float("lr", 1e-4, 0.5, log=True)   
        gamma = trial.suggest_float("gamma", 0.5, 1.0)
        alpha = trial.suggest_float("alpha", 0.1, 0.9)
        beta = trial.suggest_float("beta", 0.5, 0.99)
        k_val = trial.suggest_int("K", 2, 8) # Window size for stability 
        smooth = trial.suggest_float("smoothing", 0.0001, 0.1)
        
        optimizer = DeltaGrad(
            model.parameters(), 
            lr=lr, 
            gamma=gamma, 
            alpha=alpha, 
            beta=beta, 
            K=k_val, 
            smoothing=smooth
        )

    accuracy = train_model(trial, model, optimizer, epochs=15)
    return accuracy

if __name__ == "__main__":
    # Path configuration
    output_dir = "optuna_studies"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Tuning for Adam
    print("Starting Adam tuning...")
    study_Adam = optuna.create_study(direction="maximize")
    study_Adam.optimize(lambda trial: objective(trial, "Adam"), n_trials=20)
    joblib.dump(study_Adam.best_params, "best_params_Adam_fixed_b16_epochs15.pkl")

    # 2. Tuning for DeltaGrad
    print("\nStarting DeltaGrad tuning...")
    study_DeltaGrad = optuna.create_study(direction="maximize")
    study_DeltaGrad.optimize(lambda trial: objective(trial, "DeltaGrad"), n_trials=20)
    joblib.dump(study_DeltaGrad.best_params, "best_params_DeltaGrad_fixed_b16_epochs15.pkl")
    
    print("\nTuning complete. Best hyperparameters saved.")

    # Save full Optuna study objects for audit and visualization 
    study_path_adam = os.path.join(output_dir, "study_adam_b16_fixed_epochs15.pkl")
    joblib.dump(study_Adam, study_path_adam)

    study_path_dg = os.path.join(output_dir, "study_deltagrad_b16_fixed_epochs15.pkl")
    joblib.dump(study_DeltaGrad, study_path_dg)

    print(f"Study objects stored in: {output_dir}")