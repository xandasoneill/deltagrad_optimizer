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

def train_model(trial, model, optimizer, epochs=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 1. Carregar e Dividir os Dados (Fim do Data Leakage)
    full_trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    
    # Reservamos 5000 imagens para validação (o Optuna só vê estas para decidir o score)
    train_subset, val_subset = random_split(
        full_trainset, [45000, 5000], 
        generator=torch.Generator().manual_seed(42)
    )

    batch_size = trial.suggest_int("batch_size", 16, 16)
    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        # FASE DE TREINO
        model.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # Opcional: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # FASE DE VALIDAÇÃO (O que evita o erro do Carl McBride Ellis)
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
        
        # Reportar ao Optuna baseado nos dados de VALIDAÇÃO
        trial.report(val_accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return val_accuracy

def objective(trial, optimizer_name):
    # Fixar seed para reprodutibilidade total entre trials
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ConvNet().to(device)

    if optimizer_name == "Adam":
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        # Hiperparâmetros do DeltaGrad
        lr = trial.suggest_float("lr", 1e-4, 0.5, log=True)   
        gamma = trial.suggest_float("gamma", 0.5, 1.0)
        alpha = trial.suggest_float("alpha", 0.1, 0.9)
        beta = trial.suggest_float("beta", 0.5, 0.99)
        k_val = trial.suggest_int("K", 2, 8) # Janela de 2 a 8 para maior estabilidade
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
    # Otimização para Adam

    output_dir = "optuna_studies"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    print("Iniciando tuning do Adam...")
    study_Adam = optuna.create_study(direction="maximize")
    study_Adam.optimize(lambda trial: objective(trial, "Adam"), n_trials=20)
    joblib.dump(study_Adam.best_params, "best_params_Adam_fixed_b16.pkl")

    # Otimização para DeltaGrad
    print("\nIniciando tuning do DeltaGrad...")
    study_DeltaGrad = optuna.create_study(direction="maximize")
    study_DeltaGrad.optimize(lambda trial: objective(trial, "DeltaGrad"), n_trials=20)
    joblib.dump(study_DeltaGrad.best_params, "best_params_DeltaGrad_fixed_b16.pkl")
    
    print("\nTuning done. Hiperparameters saved.")

    study_path_adam = os.path.join(output_dir, "study_adam_b16_fixed.pkl")
    joblib.dump(study_Adam, study_path_adam)

    study_path_dg = os.path.join(output_dir, "study_deltagrad_b16_fixed.pkl")
    joblib.dump(study_DeltaGrad, study_path_dg)

    print(f"Studies saved in: {output_dir}")