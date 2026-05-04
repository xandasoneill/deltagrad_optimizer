import torch
from torch.optim import Optimizer

class DeltaGrad(Optimizer):
    def __init__(self, 
                params, 
                lr=0.01,
                K=4, 
                alpha=0.1,
                smoothing=0.9, # High factor for momentum and exploration
                weight_decay=0,
                epsilon=1e-8):
        
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(lr=lr, alpha=alpha, 
                        K=K, smoothing=smoothing, 
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
            alpha = group['alpha']
            K = group['K']
            smooth_factor = group['smoothing']
            weight_decay = group['weight_decay']
            eps = group['epsilon']

            # Pre-calculate exponential weights for vectorized history
            device = group['params'][0].device
            alpha_weights = torch.tensor([alpha**i for i in range(K)], device=device)

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                # Initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['smooth_grad'] = grad.clone() # EMA for momentum
                    state['raw_history'] = []          # History of raw gradients for R
                    
                state['step'] += 1
                smooth = state['smooth_grad']
                raw_history = state['raw_history']

                # Apply Weight Decay (L2 Regularization)
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Update the Smoothed Gradient (Inertia/Momentum)
                # This provides the 'mass' to explore the loss landscape
                if state['step'] > 1:
                    smooth.mul_(smooth_factor).add_(grad, alpha=(1 - smooth_factor))

                cur_k = len(raw_history)
                if cur_k > 0:
                    # 1. Vectorized R Calculation (Reliability)
                    # We use RAW gradients here to detect batch noise and wrong labels instantly
                    raw_history_tensor = torch.stack(raw_history) 
                    
                    diff = (grad - raw_history_tensor).abs()
                    sum_val = grad.abs() + raw_history_tensor.abs() + eps
                    
                    current_alpha = alpha_weights[:cur_k].flip(0).view(-1, *([1] * grad.dim()))
                    terms = current_alpha * (diff / sum_val)
                    R_sum = terms.sum(dim=0)
                    
                    R = (cur_k - R_sum) / cur_k
                    R = torch.clamp(R, min=0.1, max=1.0)
                else:
                    R = torch.ones_like(grad)

                # Store R for diagnostics
                state['R'] = R

                # Update raw history buffer (Circular) with the current RAW gradient
                raw_history.append(grad.clone())
                if len(raw_history) > K:
                    raw_history.pop(0)

                # 2. Final Weight Update
                # Inertia = 'smooth' (The EMA gradient)
                # Step = learning_rate * global_scaling * R * smooth
                p.addcmul_(smooth, R, value=(-eta))

        return loss