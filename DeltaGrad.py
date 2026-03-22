
# Copyright 2026 Alexandre de Abreu O'Neill Mendes

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
                smoothing=0.9,
                weight_decay=0,
                epsilon=1e-8):
        
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta parameter: {beta}")
        
        defaults = dict(lr=lr, gamma=gamma, alpha=alpha, 
                        beta=beta, K=K, smoothing=smoothing, 
                        weight_decay=weight_decay, epsilon=epsilon)
        
        super(DeltaGrad, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Hyperparameters
            eta = group['lr']
            gamma = group['gamma']
            alpha = group['alpha']
            beta = group['beta']
            K = group['K']
            smooth_factor = group['smoothing']
            weight_decay = group['weight_decay']
            eps = group['epsilon']

            # Pre-calculate exponential weights for vectorized history
            alpha_weights = torch.tensor([alpha**i for i in range(K)], device=group['params'][0].device)
            beta_weights = torch.tensor([beta**(i+1) for i in range(K)], device=group['params'][0].device)

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                # Initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['smooth_grad'] = grad.clone()
                    state['grad_history'] = [] # Stores up to K past smoothed gradients
                    
                state['step'] += 1
                smooth = state['smooth_grad']
                history = state['grad_history']

                # Apply Weight Decay (L2 Regularization)
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Update smoothed gradient (EMA)
                if state['step'] > 1:
                    smooth.mul_(smooth_factor).add_(grad, alpha=(1 - smooth_factor))

                cur_k = len(history)
                if cur_k > 0:
                    # Vectorized computation of R and Inertia
                    # 1. Stack history into a single tensor [K, parameters_shape]
                    history_tensor = torch.stack(history) 
                    
                    # 2. Vectorized R Calculation (Reliability)
                    diff = (smooth - history_tensor).abs()
                    sum_val = smooth.abs() + history_tensor.abs() + eps
                    
                    # Broadcast alpha weights across the history dimension
                    # We only use weights up to the current history length
                    current_alpha = alpha_weights[:cur_k].view(-1, *([1] * smooth.dim()))
                    terms = current_alpha * (diff / sum_val)
                    R_sum = terms.sum(dim=0)
                    
                    R = (cur_k - R_sum) / cur_k
                    R = torch.clamp(R, min=0.1, max=1.0)

                    # 3. Vectorized Inertia Calculation
                    current_beta = beta_weights[:cur_k].view(-1, *([1] * smooth.dim()))
                    grad_inertia_num = (current_beta * history_tensor).sum(dim=0)
                    grad_inertia_den = beta_weights[:cur_k].sum()
                    
                    # Bias correction for early steps
                    grad_inertia = grad_inertia_num / (grad_inertia_den + eps)
                else:
                    R = torch.ones_like(smooth)
                    grad_inertia = smooth.clone()

                # Store R for diagnostics
                state['R'] = R

                # Update history buffer (Circular)
                history.append(smooth.clone())
                if len(history) > K:
                    history.pop(0)

                # Final Weight Update
                # p = p - (learning_rate * adaptive_gamma * R * inertia)
                p.addcmul_(grad_inertia, R, value=(-eta * gamma))

        return loss
