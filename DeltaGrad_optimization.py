import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import optuna
from DeltaGrad import DeltaGrad # Importa a tua classe

# 1. DEFINIR O DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. DEFINIR A REDE (ConvNet)
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

# 3. DEFINIR A FUNÇÃO DE TREINO (Versão simplificada para o Optuna)
def train_for_optuna(trial, model, optimizer, epochs=5):
    criterion = nn.CrossEntropyLoss()
    # Carregar dados (CIFAR-100)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=trial.suggest_int("batch_size", 16, 256), shuffle=True)

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
        # Opcional: Reportar progresso ao Optuna para pruning
        trial.report(accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return accuracy

# 4. A FUNÇÃO OBJECTIVO
def objective(trial):
    # 1. Sugestões de Hiperparâmetros (O "Cérebro" do Optuna)
    lr = trial.suggest_float("lr", 1e-3, 0.5, log=True)    # Taxa de aprendizagem
    gamma = trial.suggest_float("gamma", 0.5, 1.0)        # Calibração DeltaGrad
    alpha = trial.suggest_float("alpha", 0.1, 0.9)        # Peso da história no R
    beta = trial.suggest_float("beta", 0.5, 0.99)         # Inércia dos gradientes
    k_val = trial.suggest_int("K", 1, 8)                  # Tamanho da memória (janela)
    smooth = trial.suggest_float("smoothing", 0.0001, 0.1)  # Suavização do gradiente

    # 2. Inicializar Modelo e Optimizador com as sugestões
    model = ConvNet(num_classes=100).to(device)
    optimizer = DeltaGrad(
        model.parameters(), 
        lr=lr, 
        gamma=gamma, 
        alpha=alpha, 
        beta=beta, 
        K=k_val, 
        smoothing=smooth,
    )
    
    # 3. Executar o treino (3 a 5 épocas é suficiente para o Optuna decidir)
    return train_for_optuna(trial, model, optimizer, epochs=5)

# 5. EXECUTAR
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print(f"Melhores parâmetros: {study.best_params}")