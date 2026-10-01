import torch


class GaussianPrior:
    """Isotropic Gaussian prior N(0, std^2 I) at t=1, with energy E(x) = ||x||^2 / (2 std^2)."""

    def __init__(self, std: float = 1.0, max_grad_E_norm: float = 100.0):
        self.std = std
        self.max_grad_E_norm = max_grad_E_norm

    def sample(self, batch_size, data_dim, device):
        return self.std * torch.randn(batch_size, *data_dim, device=device)

    def clip(self, grad_E):
        # norm over the last axis, as in the original ASBS implementation
        norm = torch.linalg.vector_norm(grad_E, dim=-1).detach()
        clip_coefficient = torch.clamp(self.max_grad_E_norm / (norm + 1e-6), max=1).unsqueeze(-1)
        return grad_E * clip_coefficient

    def grad_E(self, x):
        """grad E(x) = - grad log p(x) = x / std^2"""
        original_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        grad_E = (x - 0.0) / self.std**2
        return self.clip(grad_E.reshape(original_shape))


class TerminalCost:
    """Terminal adjoint a_1 = grad E(x1) + h(x1), where h is the corrector network evaluated at t=1."""

    def __init__(self, prior: GaussianPrior, corrector: torch.nn.Module):
        self.prior = prior
        self.corrector = corrector

    def __call__(self, x1, is_init_stage=False):
        grad_E = self.prior.grad_E(x1)
        if is_init_stage:  # first adjoint-matching stage uses a zero corrector
            return grad_E
        t1 = torch.ones(x1.shape[0], 1).to(x1)
        with torch.no_grad():
            corrector = self.corrector(t1, x1)
        return grad_E + corrector
