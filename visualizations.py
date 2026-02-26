import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
from DeltaGrad import DeltaGrad


def get_grad_variance(model, criterion, inputs, labels, num_samples=5):
    """Calcula a variância real do gradiente amostrando mini-batches."""
    grads = []
    # Usamos um sub-batch pequeno para calcular a variância local
    for _ in range(num_samples):
        # Amostragem aleatória dentro do batch atual para simular ruído
        indices = torch.randperm(inputs.size(0))[:32] 
        model.zero_grad()
        outputs = model(inputs[indices])
        loss = criterion(outputs, labels[indices])
        loss.backward()
        
        # Flatten de todos os gradientes num vetor só
        all_grads = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
        grads.append(all_grads)
    
    # Variância média entre as amostras
    grads_stack = torch.stack(grads)
    variance = torch.var(grads_stack, dim=0).mean().item()
    return variance


