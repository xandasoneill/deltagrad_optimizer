import torch
from torch.optim import Optimizer

class DeltaGrad(Optimizer):

    
    def __init__(self, 
                params, 
                lr=0.01,
                gamma=0.9, 
                K=4, 
                alpha=0.1,
                beta=0.9, 
                smoothing = 0.9,
                epsilon=1e-8):
        
        defaults = dict(lr=lr, gamma=gamma, alpha=alpha, 
                        beta=beta, K=K, smoothing=smoothing, epsilon=epsilon)
        
        super(DeltaGrad, self).__init__(params, defaults)
        
    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            
            
            eta = group['lr']
            gamma = group['gamma']
            alpha = group['alpha']
            beta = group['beta']
            K = group['K']
            smooth_factor = group['smoothing']
            eps = group['epsilon']

            for p in group['params']:
                    
                if p.grad is None:
                    continue
                
                # 1. Obter o gradiente atual (Ruidoso)
                # No PyTorch, o gradiente está em: p.grad.data
                grad = p.grad.data

                # 2. Aceder ao "Estado" (Memória) deste peso específico
                state = self.state[p]

                # --- INICIALIZAÇÃO (Só corre na 1ª vez) ---
                if len(state) == 0:
                    state['step'] = 0
                    state['smooth_grad'] = torch.zeros_like(p.data)
                    state['smooth_grad'].copy_(grad)

                    state['grad_history'] = []
                    
                state['step'] += 1
                
                history = state['grad_history']
                smooth = state['smooth_grad']
                
                R_sum = 0
                grad_inertia_num = 0 
                grad_inertia_den = 0
                cur_k = len(history)

                if state['step'] > 1:
                    smooth.mul_(smooth_factor).add_(grad, alpha=(1-smooth_factor))

                if cur_k > 0:

                    for j in range(1, cur_k+1):

                        g_prev = history[-j]
                        diff = smooth - g_prev
                        sum = smooth.abs() + g_prev.abs() + eps
                        term = (alpha**(j-1)) * diff.abs() / sum
                        R_sum = R_sum + term

                        grad_inertia_num += (beta**j)*g_prev
                        grad_inertia_den += beta**j

                    R = (cur_k - R_sum) / cur_k
                    R = torch.clamp(R, min=0.1, max=1.0)

                    grad_inertia = grad_inertia_num / (grad_inertia_den+eps)   

                else:

                    R = torch.ones_like(smooth)
                    grad_inertia = smooth.clone()


                state['R'] = R.detach()
                history.append(smooth.clone().detach())

                
                if len(history) > K:
                    history.pop(0)

                # --- O ESPIÃO (DEBUG) ---
                # Vamos espreitar o que está a acontecer dentro do motor
                # Só imprimimos para o primeiro grupo e primeiro parâmetro para não spammar
                if state['step'] % 100 == 0 and len(p.shape) > 1 and p.shape[0] == 32: 
                    # p.shape[0] == 32 é um truque para apanhar a primeira camada conv (32 filtros)
                    
                    avg_R = R.mean().item()
                    avg_grad = grad.abs().mean().item()
                    avg_smooth = smooth.abs().mean().item()
                    
                    print(f"\n--- DIAGNÓSTICO (Passo {state['step']}) ---")
                    print(f"R Médio: {avg_R:.5f} (Min: {R.min().item():.5f}, Max: {R.max().item():.5f})")
                    print(f"Gradiente Médio: {avg_grad:.5f}")
                    print(f"Alpha atual: {alpha}")
                    
                    if avg_R <= 0.0011:
                         print("⚠️ ALERTA: O R está colado no fundo (Clamp Min)!")
                    if avg_R >= 0.99:
                         print("⚠️ ALERTA: O R está colado no teto (Clamp Max)!")

                p.data.addcmul_(grad_inertia, R, value=(-eta*gamma))


        return loss

        