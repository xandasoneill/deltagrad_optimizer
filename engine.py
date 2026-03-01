import joblib
from DeltaGrad import DeltaGrad
import torch

from model import ConvNet

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

from visualizations import get_grad_variance



def train_model(model, optimizer, optimizer_name, epochs=5, best_params = None):

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

    if optimizer_name == "DeltaGrad":
        batch_size = best_params["batch_size"]
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    else:
        batch_size = best_params["batch_size"]
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Listas para o Método 3
    history_acc = []
    r_values = []
    variance_values = []

    correct = 0
    total = 0   

    for epoch in range(epochs):

        running_loss = 0.0
        model.train()

        for i, (inputs, labels) in enumerate(trainloader):

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

                # ---Getting Data to check relation between R and gradient variance ---
            if i % batch_size == 0:

                var_real = get_grad_variance(model, criterion, inputs, labels)
                
                if optimizer_name == "DeltaGrad":

                    avg_R = 0.0
                    n_params = 0
                    found_r = False
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            state = optimizer.state[p]
                            if 'R' in state:
                                avg_R += state['R'].mean().item()
                                n_params += 1
                                found_r = True
                
                if found_r:
                    r_values.append(avg_R / n_params)
                    variance_values.append(var_real)
                else:
                    print(f"Warning: Batch {i} - Optimizer didn't find 'R' in state!")


            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            #Data for convergence plot
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_acc = 100 * correct / total
        history_acc.append(epoch_acc) # Agora history_acc terá tamanho = epochs (10)
        print(f"Epoch {epoch+1} finished.")
            

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100 * correct / total
        

        print(f"[{optimizer_name}] Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")


    #Graph for R vs Gradient Variance

    return history_acc, r_values, variance_values
