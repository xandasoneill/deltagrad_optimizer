import torch
import torch.nn as nn
from DeltaGrad import DeltaGrad

# ---------------------------------------------------------
# 1. OS DADOS (A Lógica que ela tem de aprender)
# ---------------------------------------------------------
# Entradas (Input): 4 combinações possíveis de 0 e 1
X = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)

# Respostas Corretas (Target): A lógica XOR
# 0, 1, 1, 0
Y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# ---------------------------------------------------------
# 2. O MODELO (A "Caixa com Botões")
# ---------------------------------------------------------
# Uma rede simples com:
# - 2 entradas (os dois interruptores)
# - 5 neurónios no meio (para "pensar")
# - 1 saída (a luz acesa ou apagada)
model = nn.Sequential(
    nn.Linear(2, 5),  # Camada de entrada -> Escondida
    nn.Tanh(),        # Função de ativação (dá não-linearidade)
    nn.Linear(5, 1),  # Camada Escondida -> Saída
    nn.Sigmoid()      # Espreme o resultado entre 0 e 1 (probabilidade)
)

print("Rede Neural criada!")
print(f"Parâmetros para treinar: {sum(p.numel() for p in model.parameters())}")
# (Devem ser uns 20 e tal pesos/botões para o teu otimizador rodar)

# ---------------------------------------------------------
# 3. O TREINO (Usando o teu DeltaGrad!)
# ---------------------------------------------------------
# Instanciamos o TEU otimizador
# lr=0.1 costuma ser bom para XOR simples
optimizer = DeltaGrad(model.parameters(), lr=6.0, gamma=0.9, K=4, alpha=0.9)

loss_fn = nn.MSELoss() # Erro Quadrático Médio (distância da resposta certa)

print("\n--- A Iniciar Treino ---")

print("\n--- A Iniciar Debug ---")

for epoch in range(5000):
    optimizer.zero_grad()
    predictions = model(X)
    loss = loss_fn(predictions, Y)
    loss.backward()
    
    # --- DEBUGGING ---
    # Vamos ver se existem gradientes ANTES do otimizador mexer
    if epoch % 500 == 0:
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.norm().item())
        
        print(f"Época {epoch}: Erro = {loss.item():.5f}")
        print(f"   -> Normas dos Gradientes: {grads}")
        # Se as normas forem 0.0, o problema é do Modelo (ReLU morta).
        # Se as normas existirem (> 0) mas o erro não baixar, o problema é do Otimizador.

    optimizer.step()
    
    if loss.item() < 0.001:
        print(f"\n✅ CONVERGIU na época {epoch}!")
        break
# ---------------------------------------------------------
# 4. O TESTE FINAL
# ---------------------------------------------------------
print("\n--- Vamos ver o que ela aprendeu ---")
with torch.no_grad():
    test_output = model(X)
    predicted = (test_output > 0.5).float() # Arredondar para 0 ou 1
    
    for i in range(4):
        input_vals = X[i].tolist()
        pred_val = test_output[i].item()
        real_logic = Y[i].item()
        print(f"Entrada {input_vals} | Previsão: {pred_val:.4f} | Resposta Real: {real_logic}")