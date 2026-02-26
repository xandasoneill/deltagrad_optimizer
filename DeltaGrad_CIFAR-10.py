import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
from DeltaGrad import DeltaGrad


#Compose: composes several transforms, so the images can be fed to the NN
transform = transforms.Compose([
    transforms.ToTensor(), #Turns intergers between [0,255], to floats between [0,1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    #Normalizes the pixel values so that they range between [-1, 1],
    #it does that by making the difference with 0.5, and dividing by 0.5, 
    #in this case
])

#Loading dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
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
        self.fc2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Função de Treino Reutilizável
def train_model(optimizer_name, learning_rate, epochs=10):
    
    print(f"\n🥊 A iniciar treino com {optimizer_name} (LR={learning_rate})...")
    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == "DeltaGrad":
        # HIperparâmetros "Seguros" para CNN
        optimizer = DeltaGrad(model.parameters(), lr=learning_rate, gamma=0.8, K=4, alpha=0.8, smoothing =0.95)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Adam default geralmente é 0.001
        
    history_loss = []
    history_acc = []
    smooth_decay = 0.1
    start_time = time.time()
    
    for epoch in range(epochs):
        #current_smooth = optimizer.param_groups[0]['smoothing']
        #new_smooth = max(0.0, current_smooth * smooth_decay)
        running_loss = 0.0
        correct = 0
        total = 0
        
        model.train()
        '''
        if epoch == 10: # A meio do caminho
            print("--- A MUDAR DE MUDANÇA (LR 0.25 -> 0.025) ---")
            optimizer.param_groups[0]['lr'] = 0.025
        '''

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Clip global para justiça
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100 * correct / total
        history_loss.append(epoch_loss)
        history_acc.append(epoch_acc)
        #optimizer.param_groups[0]['smoothing'] = new_smooth
        #print(f"Época {epoch+1} | Smoothing atual: {new_smooth:.4f}")
        
        print(f"[{optimizer_name}] Época {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
        
    duration = time.time() - start_time
    print(f"⏱️ Tempo total: {duration:.1f}s")


train_model("Adam", 0.25, epochs=10)