import torch
from torch.optim import Optimizer


class DeltaGradWindowed(Optimizer):
    """Non-infinite horizon DeltaGrad (finite window K), per deltagradpaperplan.pdf Sec. 2.

        g_s^(t)  = sigma * g_s^(t-1) + (1-sigma) * g^(t)
        Phi_j    = |g_s^(t) - g_s^(t-j)| / (|g_s^(t)| + |g_s^(t-j)| + eps)   j = 1..K-1
        R_t      = clamp((K - sum_j alpha^j * Phi_j) / K, A, B)
        theta_t+1 = theta_t - lr * (R_t * g_s^(t))

    Phi_0 (comparing g_s^(t) to itself) is always 0, so it is omitted from the sum
    rather than computed. R_t divides by the fixed window K (not however many past
    steps are actually available yet) so R_t is only == 1 before any history exists.
    """

    def __init__(self,
                 params,
                 lr=0.01,
                 K=4,
                 alpha=0.1,
                 sigma=0.9,
                 A=0.1,
                 B=1.0,
                 weight_decay=0,
                 epsilon=1e-8):

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if K < 1:
            raise ValueError(f"Invalid window size K: {K}")
        if not 0.0 <= sigma < 1.0:
            raise ValueError(f"Invalid sigma parameter: {sigma}")
        if A > B:
            raise ValueError(f"Invalid clamp bounds: A={A} > B={B}")

        defaults = dict(lr=lr, K=K, alpha=alpha, sigma=sigma,
                         A=A, B=B, weight_decay=weight_decay, epsilon=epsilon)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, K, alpha, sigma = group['lr'], group['K'], group['alpha'], group['sigma']
            A, B, eps, weight_decay = group['A'], group['B'], group['epsilon'], group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['g_s'] = grad.clone()
                    state['history'] = torch.zeros((K,) + p.shape, dtype=p.dtype, device=p.device)
                    state['history_count'] = 0
                state['step'] += 1

                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                g_s = state['g_s']
                if state['step'] > 1:
                    g_s.mul_(sigma).add_(grad, alpha=(1 - sigma))
                else:
                    g_s.copy_(grad)

                history = state['history']
                cur_k = state['history_count']

                if cur_k > 0:
                    valid_history = history[:cur_k]  # index 0 = g_s^(t-1), ..., index cur_k-1 = g_s^(t-cur_k)
                    powers = torch.arange(1, cur_k + 1, device=p.device, dtype=p.dtype)
                    alpha_w = (alpha ** powers).view(-1, *([1] * p.dim()))

                    diff = (g_s - valid_history).abs()
                    denom = g_s.abs() + valid_history.abs() + eps
                    R_sum = (alpha_w * (diff / denom)).sum(dim=0)
                else:
                    R_sum = torch.zeros_like(g_s)

                R = ((K - R_sum) / K).clamp_(min=A, max=B)
                state['R'] = R

                # Roll-then-overwrite keeps ages correctly aligned across wraparound
                # (a fixed-slice-index scheme misaligns ages once step > K).
                history.copy_(torch.roll(history, shifts=1, dims=0))
                history[0].copy_(g_s)
                if cur_k < K:
                    state['history_count'] += 1

                p.addcmul_(g_s, R, value=-lr)

        return loss


class DeltaGradWindowedLegacy(Optimizer):
    """Original (pre-paper-spec) windowed DeltaGrad, preserved verbatim for
    reproducing old tuned hyperparameters (best_params/*.pkl) and past results.

    Deviates from deltagradpaperplan.pdf Sec. 2 in two ways, both intentionally
    kept as-is here rather than fixed: it re-averages the history buffer with a
    second `beta`-weighted decay ("grad_inertia") instead of using g_s directly
    as momentum, and it divides R's numerator/denominator by however many steps
    of history are available (`cur_k`) rather than the fixed window `K`. It also
    reads the history buffer via a fixed slice index that silently misaligns
    "age" once `step > K`. Not a spec to extend -- see DeltaGradWindowed instead.
    """

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

        super().__init__(params, defaults)

    @torch.no_grad()
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

                if len(state) == 0:
                    state['step'] = 0
                    state['smooth_grad'] = grad.clone()
                    state['history_buffer'] = torch.zeros((K,) + p.shape, dtype=p.dtype, device=p.device)
                    state['history_count'] = 0

                state['step'] += 1
                smooth = state['smooth_grad']
                history_buffer = state['history_buffer']

                if weight_decay != 0:
                    grad.add_(p, alpha=weight_decay)

                if state['step'] > 1:
                    smooth.mul_(smooth_factor).add_(grad, alpha=(1 - smooth_factor))
                else:
                    smooth.copy_(grad)

                cur_k = state['history_count']

                if cur_k > 0:
                    valid_history = history_buffer[:cur_k]

                    powers = torch.arange(cur_k - 1, -1, -1, device=p.device, dtype=p.dtype)
                    alpha_w = (alpha ** powers).view(-1, *([1] * p.dim()))
                    beta_w = (beta ** (powers + 1)).view(-1, *([1] * p.dim()))

                    diff = (smooth - valid_history).abs()
                    sum_val = smooth.abs() + valid_history.abs() + eps

                    terms = alpha_w * (diff / sum_val)
                    R_sum = terms.sum(dim=0)

                    R = (cur_k - R_sum) / cur_k
                    R.clamp_(min=0.1, max=1.0)

                    grad_inertia_num = (beta_w * valid_history).sum(dim=0)
                    grad_inertia_den = beta_w.sum()
                    grad_inertia = grad_inertia_num / (grad_inertia_den + eps)
                else:
                    R = torch.ones_like(smooth)
                    grad_inertia = smooth.clone()

                state['R'] = R

                idx = (state['step'] - 1) % K
                history_buffer[idx].copy_(smooth)

                if state['history_count'] < K:
                    state['history_count'] += 1

                p.addcmul_(grad_inertia, R, value=(-eta))

        return loss
