import joblib
from DeltaGrad import DeltaGrad
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import ConvNet
from visualizations import get_grad_variance

def train_model(model, optimizer, optimizer_name, epochs=15, batch=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Loading Datasets: Training and Independent Testing (To avoid data leakage)
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    #batch_size = best_params["batch_size"]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch, shuffle=False)

    print(f"Training on: {device}")

    # Data collection lists for analysis
    history_acc = []
    r_values = []
    variance_values = []

    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # --- DATA COLLECTION (R vs Gradient Variance) ---
            # Collect data every 5 batches to reduce computational overhead
            if i % 10 == 0 and i != 0:
                # Save current training gradients to prevent the measurement from clearing them
                original_grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]
                
                # Measure real gradient variance using sub-sampling
                var_real = get_grad_variance(model, criterion, inputs, labels)
                variance_values.append(var_real)
                
                if optimizer_name == "DeltaGrad":
                    avg_R = 0.0
                    n_params = 0
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            state = optimizer.state[p]
                            if 'R' in state:
                                avg_R += state['R'].mean().item()
                                n_params += 1
                    
                    if n_params > 0:
                        r_values.append(avg_R / n_params)
                
                # Restore original gradients so the optimizer.step() works correctly
                for p, g in zip([p for p in model.parameters() if p.grad is not None], original_grads):
                    p.grad = g

            # Optimization step
            optimizer.step()
            running_loss += loss.item()

        # --- TESTING PHASE (Generalization Accuracy) ---
        # Evaluation mode: Disables dropout and BN updates for stable inference
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        # Calculate epoch metrics
        epoch_test_acc = 100 * test_correct / test_total
        history_acc.append(epoch_test_acc)
        epoch_loss = running_loss / len(trainloader)
        
        print(f"[{optimizer_name}] Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Test Acc: {epoch_test_acc:.2f}%")

    return history_acc, r_values, variance_values
