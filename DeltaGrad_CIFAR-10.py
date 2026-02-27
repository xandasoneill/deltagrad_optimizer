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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=75, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

#CNN

class ConvNet(nn.Module):
    def __init__(self, num_classes=100):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)
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

    # Listas para o Método 3
    history_acc = []
    r_values = []
    variance_values = []
    avg_R = 0.0
    n_params = 0
    correct = 0
    total = 0   
    start_time = time.time()

    print(f"\n Initializing training with {optimizer_name} (LR={learning_rate})...")
    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == "DeltaGrad":
        optimizer = DeltaGrad(model.parameters(), lr=learning_rate, gamma=0.849976873438669, K=5, alpha=0.2112904776285402, smoothing=0.033098309469824665, beta=0.5809374913747086)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    
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
            if optimizer_name == "DeltaGrad" and i % 50 == 0:
                var_real = get_grad_variance(model, criterion, inputs, labels)
                
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
                    print(f"Aviso: Batch {i} - Optimizer didn't find 'R' in state!")


            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            #Data for convergence plot
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_acc = 100 * correct / total
        history_acc.append(epoch_acc) # Agora history_acc terá tamanho = epochs (10)
        print(f"Época {epoch+1} concluída.")
            

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100 * correct / total
        

        print(f"[{optimizer_name}] Época {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")


    #Graph for R vs Gradient Variance
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

    return history_acc, r_values, variance_values

#Train DeltaGrad (Proposed)
dg_acc, r_values, v_values = train_model("DeltaGrad", learning_rate=0.08953341539706354, epochs=10)

#Calculate Pearson correlation and R²between R and Gradient Variance
r_pearson, p_value = stats.pearsonr(r_values, v_values)
r_squared = r_pearson**2

print(f"r: {r_pearson:.4f}")
print(f"R² : {r_squared:.4f}")
print(f"P-value: {p_value:.4e}")


# Train Adam (Baseline)
adam_acc, _, v_values = train_model("Adam", learning_rate=0.001, epochs=10)


# 3. Generate Convergence Plot Comparing DeltaGrad and Adam
plt.figure(figsize=(10, 6))

epochs_range = range(1, 11)
plt.plot(epochs_range, dg_acc, 'o-', label='DeltaGrad (Proposed)', linewidth=2, markersize=6)
plt.plot(epochs_range, adam_acc, 's--', label='Adam (Baseline)', linewidth=2, markersize=6)

plt.title('Convergence Rate on CIFAR-10', fontsize=14, fontweight='bold')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Test Accuracy (%)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Guardar em PDF para o Overleaf
plt.savefig("convergence_comparison.pdf", bbox_inches='tight')
plt.savefig("convergence_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

print("✅ Comparative graph saved as 'convergence_comparison.pdf'")