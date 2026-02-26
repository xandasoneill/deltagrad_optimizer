import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
from DeltaGrad import DeltaGrad
from visualizations import get_grad_variance

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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

#CNN

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Bloco 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) # Estabilidade extra
        self.pool = nn.MaxPool2d(2, 2)
        # Bloco 2
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Bloco 3
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        # Classificação
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 100)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(optimizer_name, learning_rate, epochs=10):
    print(f"\n🥊 A iniciar treino com {optimizer_name} (LR={learning_rate})...")
    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == "DeltaGrad":
        optimizer = DeltaGrad(model.parameters(), lr=learning_rate, gamma=0.8, K=4, alpha=0.8, smoothing=0.95)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Listas para o Método 3
    r_values = []
    variance_values = []
    avg_R = 0.0
    n_params = 0
    running_loss = 0.0
    correct = 0
    total = 0   
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

                # --- MÉTODO 3: DEBUG ---
            if optimizer_name == "DeltaGrad" and i % 50 == 0:
                var_real = get_grad_variance(model, criterion, inputs, labels)
                
                found_r = False
                for group in optimizer.param_groups:
                    for p in group['params']:
                        state = optimizer.state[p]
                        # PRINT DE DEBUG:
                        # print(f"Keys no state: {state.keys()}") 
                        if 'R' in state:
                            avg_R += state['R'].mean().item()
                            n_params += 1
                            found_r = True
            
                if found_r:
                    r_values.append(avg_R / n_params)
                    variance_values.append(var_real)
                else:
                    print(f"Aviso: Batch {i} - Otimizador não encontrou a chave 'R' no state!")


            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100 * correct / total
        

        print(f"[{optimizer_name}] Época {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

    if optimizer_name == "DeltaGrad" and len(r_values) > 0:
        plt.figure(figsize=(10, 6))
        plt.scatter(r_values, variance_values, alpha=0.6, c=variance_values, cmap='viridis', edgecolors='k')
        
        plt.title("DeltaGrad: R vs Gradient Variance", fontsize=14)
        plt.xlabel("Metric R (Network Average)", fontsize=12)
        plt.ylabel("Real Gradient Variance", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # 1. Guardar para o Overleaf (PDF é o melhor para LaTeX)
        plt.savefig("metodo3_deltagrad.pdf", bbox_inches='tight')
        
        # 2. Guardar em PNG para veres rápido no PC
        plt.savefig("metodo3_deltagrad.png", dpi=300, bbox_inches='tight')
        
        print("✅ Gráficos guardados como 'metodo3_deltagrad.pdf' e '.png'")
        plt.show()

    return r_values, variance_values




# Executar
r_data, var_data = train_model("DeltaGrad", 0.25, epochs=10)