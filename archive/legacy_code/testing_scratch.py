# import numpy as np
# from scipy.optimize import curve_fit
# from sklearn.metrics import r2_score
# import os

# def inverse_model(x, a, b, c):
#     return a / (x + b) + c

# all_r = []
# all_variance = []

# for i in range(1, 5 + 1):
#     r_file = f"r_values_run_DeltaGrad_{i}.txt"
#     v_file = f"variance_values_run_DeltaGrad_{i}.txt"

#     # Check if files exist to prevent runtime errors
#     if os.path.exists(r_file) and os.path.exists(v_file):
#         # Load R values: strip whitespace and convert to float
#         with open(r_file, "r") as f:
#             all_r.extend([float(line.strip()) for line in f if line.strip()])
        
#         # Load Variance values: strip whitespace and convert to float
#         with open(v_file, "r") as f:
#             all_variance.extend([float(line.strip()) for line in f if line.strip()])
        
#         print(f"Data from Run {i} successfully loaded.")
#     else:
#         print(f"Warning: Files for Run {i} were not found.")

# # Proceed to plotting if data was successfully collected
# if all_r and all_variance:
#     # 2. Preparar os dados (removendo o primeiro ponto se ainda não o fizeste)
#     # --- CORREÇÃO DOS EIXOS ---
#     # X deve ser a Variância, Y deve ser o R
#     x_data = np.array(all_variance)
#     y_data = np.array(all_r)

#     # 3. Ajustar a curva aos dados com mais iterações e melhores chutes iniciais
#     try:
#         # Aumentamos maxfev para 10.000 para dar tempo ao algoritmo
#         # Ajustamos p0: 
#         # a: escala pequena pois a var é pequena
#         # b: pequeno ajuste de offset
#         # c: 0.7 (onde o teu R costuma estabilizar)
#         popt, _ = curve_fit(inverse_model, x_data, y_data, 
#                             p0=[1e-6, 1e-4, 0.7], 
#                             maxfev=10000)

#         # 4. Calcular os valores previstos e o R²
#         y_pred = inverse_model(x_data, *popt)
#         r2 = r2_score(y_data, y_pred)

#         print(f"Sucesso!")
#         print(f"Coeficiente de Determinação (R²): {r2:.4f}")
#         print(f"Equação: R = {popt[0]:.2e} / (Var + {popt[1]:.2e}) + {popt[2]:.4f}")

#     except RuntimeError as e:
#         print(f"Erro persistente: {e}")
#         print("Dica: Tenta normalizar os valores de Variância (ex: Var * 1000) para facilitar o cálculo.")



# import joblib 

# results =joblib.load("results_batchtest/DeltaGrad_results_batch64_lr0.03194028510565753.pkl")

# print(results["device"])

# import torch

# # 1. Criamos a nossa "história" de gradientes (Dummy Tensors)
# # Index 0: Oldest, Index 2: Newest
# history = torch.tensor([10.0, 20.0, 30.0]) 

# # 2. Criamos a nossa base de pesos alpha
# # Resulta em: [1.0, 0.1, 0.01] (alpha^0, alpha^1, alpha^2)
# alpha = 0.1
# alpha_weights = torch.tensor([alpha**i for i in range(3)])

# print(f"History Tensors: {history}")
# print(f"Base Alpha Weights: {alpha_weights}\n")


# # Como estavas a fazer:
# current_alpha_old = alpha_weights[:3]

# # Simulação da multiplicação que acontece no R
# result_old = history * current_alpha_old

# print("--- LOGICA ANTIGA ---")
# print(f"Pesos aplicados: {current_alpha_old}")
# print(f"Cálculo: {history[0]}*1.0, {history[1]}*0.1, {history[2]}*0.01")
# print(f"Resultado: {result_old}")
# # O resultado é dominado pelo 10.0 (o mais velho)


# # Como deve ser para reagir ao ruído atual:
# current_alpha_new = alpha_weights[:3].flip(0)

# # Simulação da multiplicação
# result_new = history * current_alpha_new

# print("--- LOGICA NOVA (CORRIGIDA) ---")
# print(f"Pesos aplicados: {current_alpha_new}") # Agora é [0.01, 0.1, 1.0]
# print(f"Cálculo: {history[0]}*0.01, {history[1]}*0.1, {history[2]}*1.0")
# print(f"Resultado: {result_new}")
# # O resultado é dominado pelo 30.0 (o mais recente!)

import joblib
import pprint

# 1. Coloca aqui o caminho exato do teu ficheiro
caminho_do_ficheiro = "best_params/best_params_DeltaGrad_fixed_b16_epochs15.pkl"

# 2. Carrega os dados
meus_parametros = joblib.load(caminho_do_ficheiro)

# 3. Imprime o conteúdo de forma legível
print("--- TIPO DE DADOS ---")
print(type(meus_parametros))
print("\n--- CONTEÚDO ---")
pprint.pprint(meus_parametros)