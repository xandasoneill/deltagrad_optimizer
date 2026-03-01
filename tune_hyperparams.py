import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
from DeltaGrad import DeltaGrad
from visualizations import get_grad_variance
from scipy import stats
import optuna
from model import ConvNet
import joblib

def train_model(trial, model, optimizer, epochs=5):

    criterion = nn.CrossEntropyLoss()
    #Compose: composes several transforms, so the images can be fed to the NN
    transform = transforms.Compose([
        transforms.ToTensor(), #Turns intergers between [0,255], to floats between [0,1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
        #Normalizes the pixel values so that they range between [-1, 1],
        #it does that by making the difference with 0.5, and dividing by 0.5, 
        #in this case
    ])


    #Loading dataset
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=trial.suggest_int("batch_size", 16, 256), shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
 
    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        
        trial.report(accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return accuracy

def objective(trial, optimizer):

    if optimizer == "Adam":
        optimizer_name = "Adam"

    elif optimizer == "DeltaGrad":
        optimizer_name = "DeltaGrad"
    
    model = ConvNet().to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=trial.suggest_float("lr", 1e-5, 1e-2, log=True))
    else:
       

        lr = trial.suggest_float("lr", 1e-3, 0.5, log=True)   
        gamma = trial.suggest_float("gamma", 0.5, 1.0)        # Calibração DeltaGrad
        alpha = trial.suggest_float("alpha", 0.1, 0.9)        # Peso da história no R
        beta = trial.suggest_float("beta", 0.5, 0.99)         # Inércia dos gradientes
        k_val = trial.suggest_int("K", 1, 8)                  
        smooth = trial.suggest_float("smoothing", 0.0001, 0.1)
        

        optimizer = DeltaGrad(
        model.parameters(), 
        lr=lr, 
        gamma=gamma, 
        alpha=alpha, 
        beta=beta, 
        K=k_val, 
        smoothing=smooth,
        )

    accuracy = train_model(trial, model, optimizer)
    return accuracy



if __name__ == "__main__":
    
    study_Adam = optuna.create_study(direction="maximize")
    study_Adam.optimize(lambda trial: objective(trial, "Adam"), n_trials=10)

    joblib.dump(study_Adam.best_params, "best_params_Adam.pkl")

    study_DeltaGrad = optuna.create_study(direction="maximize")
    study_DeltaGrad.optimize(lambda trial: objective(trial, "DeltaGrad"), n_trials=10)

    joblib.dump(study_DeltaGrad.best_params, "best_params_DeltaGrad.pkl")
    



