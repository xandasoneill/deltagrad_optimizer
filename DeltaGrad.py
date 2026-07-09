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
        
        defaults = dict(lr=lr, alpha=alpha, 
                        beta=beta, K=K, smoothing=smoothing, 
                        weight_decay=weight_decay, epsilon=epsilon)
        
        super(DeltaGrad, self).__init__(params, defaults)

    @torch.no_grad()
    @torch.compile  # JIT compile for C++/CUDA fusion
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            eta = group['lr']
            alpha = group['alpha']
            beta = group['beta']
            K = group['K']
            smooth_factor = group['smoothing']
            weight_decay = group['weight_decay']
            eps = group['epsilon']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                # Initialize state and pre-allocate circular buffer
                if len(state) == 0:
                    state['step'] = 0
                    state['smooth_grad'] = grad.clone()
                    state['history_buffer'] = torch.zeros((K,) + p.shape, dtype=p.dtype, device=p.device)
                    state['history_count'] = 0
                    
                state['step'] += 1
                smooth = state['smooth_grad']
                history_buffer = state['history_buffer']

                # In-place weight decay
                if weight_decay != 0:
                    grad.add_(p, alpha=weight_decay)

                # In-place EMA update
                if state['step'] > 1:
                    smooth.mul_(smooth_factor).add_(grad, alpha=(1 - smooth_factor))
                else:
                    smooth.copy_(grad)

                cur_k = state['history_count']
                
                if cur_k > 0:
                    valid_history = history_buffer[:cur_k]
                    
                    # Vectorized decay weights on GPU
                    powers = torch.arange(cur_k - 1, -1, -1, device=p.device, dtype=p.dtype)
                    alpha_w = (alpha ** powers).view(-1, *([1] * p.dim()))
                    beta_w = (beta ** (powers + 1)).view(-1, *([1] * p.dim()))

                    # Vectorized Reliability (R)
                    diff = (smooth - valid_history).abs()
                    sum_val = smooth.abs() + valid_history.abs() + eps
                    
                    terms = alpha_w * (diff / sum_val)
                    R_sum = terms.sum(dim=0)
                    
                    R = (cur_k - R_sum) / cur_k
                    R.clamp_(min=0.1, max=1.0) # In-place clamp

                    # Vectorized Inertia
                    grad_inertia_num = (beta_w * valid_history).sum(dim=0)
                    grad_inertia_den = beta_w.sum()
                    grad_inertia = grad_inertia_num / (grad_inertia_den + eps)
                else:
                    R = torch.ones_like(smooth)
                    grad_inertia = smooth.clone()

                state['R'] = R

                # Update circular buffer in-place
                idx = (state['step'] - 1) % K
                history_buffer[idx].copy_(smooth) 
                
                if state['history_count'] < K:
                    state['history_count'] += 1

                # Final in-place weight update
                p.addcmul_(grad_inertia, R, value=(-eta))

        return loss