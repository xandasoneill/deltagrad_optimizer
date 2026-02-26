import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ==========================================
# 1. IMPORTAR O TEU OTIMIZADOR
# ==========================================
# (Cola aqui a classe DeltaGrad corrigida ou importa-a)
# Se estiver no mesmo ficheiro, deixa estar. Se não:
from DeltaGrad import DeltaGrad 

# ==========================================
# 2. PREPARAR DADOS (MNIST)
# ==========================================
# Transformar imagens em Tensores e Normalizar (ajuda a convergência)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("A descarregar MNIST (pode demorar 1 min)...")
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                         transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, 
                                        transform=transform)

# Carregadores de dados (Batches de 64 imagens)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# ==========================================
# 3. O MODELO (Rede "Perceptrão" Clássica)
# ==========================================
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.flatten = nn.Flatten() # Transforma imagem 28x28 em linha de 784
        self.fc1 = nn.Linear(784, 128) 
        self.relu = nn.ReLU() # Aqui ReLU funciona bem (MNIST é fácil)
        self.fc2 = nn.Linear(128, 10)  # 10 saídas (Dígitos 0 a 9)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = NeuralNet()

# Mover para GPU se disponível (acelera muito)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(f"A treinar em: {device}")

# ==========================================
# 4. CONFIGURAÇÃO (O TEU MOMENTO!)
# ==========================================
criterion = nn.CrossEntropyLoss()


# Tenta o teu DeltaGrad.
# Sugestão: Começa com lr=0.1. Se for estável, tenta subir para 1.0 ou mais.
optimizer = DeltaGrad(model.parameters(), lr=0.1, gamma=0.5, K=4, alpha=0.5, smoothing=0.9)

# Para comparar depois, descomenta isto e vê a diferença:
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 5. LOOP DE TREINO
# ==========================================
num_epochs = 3 # MNIST aprende rápido, 3 épocas chega para ver

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # 1. Forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 2. Backward
        optimizer.zero_grad()
        loss.backward()
        
        # 3. Step (DeltaGrad em ação!)
        # Opcional: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if (i+1) % 300 == 0:
            print(f'Época [{epoch+1}/{num_epochs}], Passo [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# ==========================================
# 6. TESTE DE PRECISÃO
# ==========================================
print("\n--- A Avaliar Precisão ---")
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f'Precisão do DeltaGrad no MNIST: {acc:.2f}%')