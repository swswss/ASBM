"""Reference SDEs, the controlled SDE, and the SDE/ODE integrator used by ASBM.

All reference processes are written as  dX_t = f(t, X_t) dt + g(t) dW_t,  t in [0, 1],
where t=0 is the data side and t=1 is the prior (noise) side.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch


def _bcast(v: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Reshape a per-sample tensor of shape (B,) / (B, 1) so it broadcasts against `ref` (B, ...)."""
    return v.reshape(v.shape[0], *([1] * (ref.ndim - 1)))


class BaseSDE(torch.nn.Module):
    """dX_t = f(t, X_t) dt + g(t) dW_t"""

    fm = False  # True only for the flow-matching (ODE) parametrization of the backward process

    def __init__(self, name=None):
        super().__init__()
        self.name = name

    def randn_like(self, x: torch.Tensor):
        return torch.randn_like(x)

    def propagate(self, x, dx):
        return x + dx

    def _pt_gauss_param(self, t, mu0):
        """Gaussian transition p_{t|0}(x | x0) = N(mu_t, var_t I). Returns (mu_t, var_t)."""
        raise NotImplementedError

    def cond_score(self, x0, t, xt, return_raw=False):
        """grad log p_{t|0}(xt | x0) = (mu_t - xt) / var_t"""
        loc, var = self._pt_gauss_param(t, x0)
        var = torch.as_tensor(var, device=xt.device).clamp_min(1e-12)
        var = var.reshape(var.shape[0], *([1] * (loc.ndim - 1)))
        if return_raw:
            return loc, xt, var
        return (loc - xt) / var

    def sample_posterior(self, t, x0, x1):
        """Sample x_t from the reference bridge p^{base}_{t|0,1}(x | x0, x1)."""
        raise NotImplementedError

    # f
    def drift(self, t: torch.Tensor, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    # g
    def diff(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BrownianMotionSDE(BaseSDE):
    """dX_t = k dW_t"""

    def __init__(self, k, name="brownian"):
        super().__init__(name)
        self.k = k

    def drift(self, t, x, **kwargs):
        return torch.zeros_like(x)

    def diff(self, t):
        return self.k * torch.ones_like(t)

    def _pt_gauss_param(self, t, mu0):
        return mu0, self.k**2 * t

    def sample_posterior(self, t, x0, x1):
        original_shape = x0.shape
        x0 = x0.reshape(x0.shape[0], -1)
        x1 = x1.reshape(x1.shape[0], -1)
        (B, D) = x0.shape
        assert x1.shape == (B, D) and t.shape == (B, 1)

        z = self.randn_like(x0)
        mean = (1 - t) * x0 + t * x1
        coeff = self.k**2 * t * (1 - t)
        coeff[coeff < 0] = 0  # avoid numerical error close to the boundary
        sample = mean + torch.sqrt(coeff) * z
        return sample.reshape(original_shape)


class VESDE(BaseSDE):
    """Variance exploding: dX_t = sigma_min (sigma_max/sigma_min)^t sqrt(2 log(sigma_max/sigma_min)) dW_t"""

    def __init__(self, sigma_min, sigma_max, name="ve"):
        super().__init__(name)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_diff = sigma_max / sigma_min
        self.total_var = sigma_max**2 - sigma_min**2

    def _diffsquare_integral(self, t):
        """int_0^t g^2(s) ds"""
        return (self.sigma_min**2) * ((self.sigma_diff) ** (2 * t) - 1)

    def drift(self, t, x, **kwargs):
        return torch.zeros_like(x)

    def diff(self, t):
        return self.sigma_min * (self.sigma_diff**t) * ((2 * np.log(self.sigma_diff)) ** 0.5)

    def _pt_gauss_param(self, t, mu0):
        return mu0, self._diffsquare_integral(t)

    def sample_posterior(self, t, x0, x1):
        original_shape = x0.shape
        x0 = x0.reshape(x0.shape[0], -1)
        x1 = x1.reshape(x1.shape[0], -1)
        (B, D) = x0.shape
        assert x1.shape == (B, D) and t.shape == (B, 1)

        t_reparam = self._diffsquare_integral(t) / self.total_var
        z = self.randn_like(x0)
        mean = (1 - t_reparam) * x0 + t_reparam * x1
        coeff = self.total_var * t_reparam * (1 - t_reparam)
        coeff[coeff < 0] = 0
        sample = mean + torch.sqrt(coeff) * z
        return sample.reshape(original_shape)


class EDM_VE(BaseSDE):
    """VE process with sigma_t linear in t (or the EDM rho-schedule if `edm_sigma`)."""

    def __init__(self, sigma_min, sigma_max, name="edm_ve", edm_sigma=False):
        super().__init__(name)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.total_var = sigma_max**2 - sigma_min**2
        self.rho = 7
        self.edm_sigma = edm_sigma

    def sigma_t(self, t):
        if self.edm_sigma:
            rho_inv = 1 / self.rho
            return (self.sigma_min**rho_inv + t * (self.sigma_max**rho_inv - self.sigma_min**rho_inv)) ** self.rho
        return (1 - t) * self.sigma_min + t * self.sigma_max

    def sigma_t_differentiate(self, t):
        if self.edm_sigma:
            rho_inv = 1 / self.rho
            base = self.sigma_min**rho_inv + t * (self.sigma_max**rho_inv - self.sigma_min**rho_inv)
            return self.rho * (base ** (self.rho - 1)) * (self.sigma_max**rho_inv - self.sigma_min**rho_inv)
        return self.sigma_max - self.sigma_min

    def _diffsquare_integral(self, t):
        """int_0^t g^2(s) ds"""
        return self.sigma_t(t) ** 2 - self.sigma_min**2

    def drift(self, t, x, **kwargs):
        return torch.zeros_like(x)

    def diff(self, t):
        return torch.sqrt(2 * self.sigma_t(t) * self.sigma_t_differentiate(t))

    def _pt_gauss_param(self, t, mu0):
        return mu0, self._diffsquare_integral(t)

    def sample_posterior(self, t, x0, x1):
        original_shape = x0.shape
        x0 = x0.reshape(x0.shape[0], -1)
        x1 = x1.reshape(x1.shape[0], -1)
        (B, D) = x0.shape
        assert x1.shape == (B, D) and t.shape == (B, 1)

        z = self.randn_like(x0)
        t_reparam = self._diffsquare_integral(t) / self.total_var
        mean = (1 - t_reparam) * x0 + t_reparam * x1
        coeff = self._diffsquare_integral(t) * (1 - t_reparam)
        coeff[coeff < 0] = 0
        sample = mean + torch.sqrt(coeff) * z
        return sample.reshape(original_shape)


class VPSDE(BaseSDE):
    """Variance preserving: dX_t = -beta_t / 2 X_t dt + sigma sqrt(beta_t) dW_t,
    with beta_t linear from beta0 (t=0) to beta1 (t=1), or constant (beta0+beta1)/2 if `const`.
    """

    def __init__(self, beta0=0.1, beta1=4.0, sigma=1.0, name="vp", const=False):
        super().__init__(name)
        self.register_buffer("beta1", torch.tensor(beta1, dtype=torch.float), persistent=False)
        self.register_buffer("beta0", torch.tensor(beta0, dtype=torch.float), persistent=False)
        self.register_buffer("sigma", torch.tensor(sigma, dtype=torch.float), persistent=False)
        self.const = const
        if self.const:
            self.register_buffer("beta_const", torch.tensor((beta0 + beta1) / 2, dtype=torch.float), persistent=False)

    def _beta(self, t):
        if self.const:
            return self.beta_const
        return torch.lerp(self.beta0, self.beta1, t)

    def drift(self, t, x, **kwargs):
        return -0.5 * self._beta(t) * x

    def diff(self, t):
        return self.sigma * torch.sqrt(self._beta(t))

    def coeff1(self, t):
        """log(kappa_bar_t) = -1/2 int_0^t beta(s) ds"""
        if self.const:
            return -0.5 * self.beta_const * t
        return -0.25 * t * (self._beta(t) + self.beta0)

    def coeff2(self, t):
        """log(kappa_t) = -1/2 int_t^1 beta(s) ds"""
        if self.const:
            return -0.5 * self.beta_const * (1 - t)
        return -0.25 * (1 - t) * (self._beta(t) + self.beta1)

    def _pt_gauss_param(self, t, mu0):
        original_shape = mu0.shape
        mu0 = mu0.reshape(mu0.shape[0], -1)
        coeff = self.coeff1(t)
        mu = mu0 * torch.exp(coeff)
        var = self.sigma**2 * (1 - torch.exp(2 * coeff))
        var[var < 0] = 0
        return mu.reshape(original_shape), var

    def sample_posterior(self, t, x0, x1, return_coeffs=False):
        """Bridge sample x_t = a_t x0 + b_t x1 + std_t z. With `return_coeffs`, also returns (a_t, b_t, std_t)."""
        original_shape = x0.shape
        x0 = x0.reshape(x0.shape[0], -1)
        x1 = x1.reshape(x1.shape[0], -1)

        coeff1 = self.coeff1(t)  # log(kappa_bar_t)
        coeff2 = self.coeff2(t)  # log(kappa_t)
        if self.const:
            coeff3 = -0.5 * self.beta_const
        else:
            coeff3 = -0.25 * (self.beta1 + self.beta0)  # log(kappa_bar_1)

        x0_coeff = torch.exp(coeff1) * (1 - torch.exp(2 * coeff2)) / (1 - torch.exp(2 * coeff3))
        x1_coeff = torch.exp(coeff2) * (1 - torch.exp(2 * coeff1)) / (1 - torch.exp(2 * coeff3))
        mu = x0_coeff * x0 + x1_coeff * x1

        var = (1 - torch.exp(2 * coeff1)) * (1 - torch.exp(2 * coeff2)) / (1 - torch.exp(2 * coeff3) + 1e-8)
        var[var < 0] = 0
        std = self.sigma * torch.sqrt(var)

        xt = mu + std * torch.randn_like(mu)
        if return_coeffs:
            return xt.reshape(original_shape), x0_coeff, x1_coeff, std
        return xt.reshape(original_shape)

    def sample_marginal(self, t, x0):
        """x_t ~ p_{t|0}(. | x0) = N(alpha_t x0, sigma_t^2 I). Returns (x_t, eps, sigma_t) with sigma_t of shape (B, 1)."""
        original_shape = x0.shape
        x0 = x0.reshape(x0.shape[0], -1)
        coeff = self.coeff1(t)
        alpha_t = torch.exp(coeff)
        sigma_t = self.sigma * torch.sqrt((1.0 - torch.exp(2.0 * coeff)).clamp_min(0.0))
        z = torch.randn_like(x0)
        xt = alpha_t * x0 + sigma_t * z
        return xt.reshape(original_shape), z.reshape(original_shape), sigma_t


class LinearSDE(BaseSDE):
    """dX_t = -m X_t / (1 - m t) dt + k sqrt((1 + m t) / (1 - m t)) dW_t"""

    def __init__(self, m=1.0, k=1.0, name="linear"):
        super().__init__(name)
        self.register_buffer("m", torch.tensor(m, dtype=torch.float), persistent=False)
        self.register_buffer("k", torch.tensor(k, dtype=torch.float), persistent=False)

    def drift(self, t, x, **kwargs):
        return -self.m * x / (1 - self.m * t)

    def diff(self, t):
        return self.k * torch.sqrt((1 + self.m * t) / (1 - self.m * t))

    def adjoint_weight(self, t):
        return (1 - self.m) / (1 - self.m * t)

    def _pt_gauss_param(self, t, mu0):
        original_shape = mu0.shape
        mu0 = mu0.reshape(mu0.shape[0], -1)
        mu = mu0 * (1 - self.m * t)
        var = self.k**2 * t
        var[var < 0] = 0
        return mu.reshape(original_shape), var

    def sample_posterior(self, t, x0, x1):
        original_shape = x0.shape
        x0 = x0.reshape(x0.shape[0], -1)
        x1 = x1.reshape(x1.shape[0], -1)

        x0_coeff = (1 - t) * (1 - t * self.m**2) / (1 - self.m * t)
        x1_coeff = t * (1 - self.m) / (1 - self.m * t)
        mu = x0_coeff * x0 + x1_coeff * x1

        var = self.k**2 * t * (1 - t) * (1 - t * self.m**2) / (1 - self.m * t) ** 2
        var[var < 0] = 0
        xt = mu + torch.sqrt(var) * torch.randn_like(mu)
        return xt.reshape(original_shape)


class LinearStdSDE(BaseSDE):
    """dX_t = -m X_t / (1 - m t) dt + sqrt(2) k sqrt(t / (1 - m t)) dW_t  (std of p_{t|0} is k t)"""

    def __init__(self, m=1.0, k=1.0, name="linear_std"):
        super().__init__(name)
        self.register_buffer("m", torch.tensor(m, dtype=torch.float), persistent=False)
        self.register_buffer("k", torch.tensor(k, dtype=torch.float), persistent=False)

    def drift(self, t, x, **kwargs):
        return -self.m * x / (1 - self.m * t)

    def diff(self, t):
        return 2**0.5 * self.k * torch.sqrt(t / (1 - self.m * t))

    def adjoint_weight(self, t):
        return (1 - self.m) / (1 - self.m * t)

    def _pt_gauss_param(self, t, mu0):
        original_shape = mu0.shape
        mu0 = mu0.reshape(mu0.shape[0], -1)
        mu = mu0 * (1 - self.m * t)
        var = (self.k**2) * (t**2)
        var[var < 0] = 0
        return mu.reshape(original_shape), var

    def sample_posterior(self, t, x0, x1):
        original_shape = x0.shape
        x0 = x0.reshape(x0.shape[0], -1)
        x1 = x1.reshape(x1.shape[0], -1)

        x0_coeff = (1 - 2 * self.m * t - t**2 + 2 * self.m * (t**2)) / (1 - self.m * t)
        x1_coeff = (t**2) * (1 - self.m) / (1 - self.m * t)
        mu = x0_coeff * x0 + x1_coeff * x1

        var = (self.k**2) * (t**2) * (1 - 2 * self.m * t - t**2 + 2 * self.m * (t**2)) / (1 - self.m * t) ** 2
        var[var < 0] = 0
        xt = mu + torch.sqrt(var) * torch.randn_like(mu)
        return xt.reshape(original_shape)


class FlowMatching(BaseSDE):
    """Deterministic backward parametrization: the network predicts the velocity v_t = x1 - x0."""

    fm = True

    def __init__(self):
        super().__init__("flow_matching")

    def drift(self, t, x, **kwargs):
        return None

    def diff(self, t):
        return 0.0


def build_reference_sde(opt) -> BaseSDE:
    if opt.sde_type in ["vp", "vp_const"]:
        return VPSDE(beta0=opt.beta_t0, beta1=opt.beta_t1, sigma=opt.vp_sigma, name=opt.sde_type,
                     const=opt.sde_type == "vp_const")
    if opt.sde_type == "ve":
        return VESDE(sigma_min=opt.sigma_min, sigma_max=opt.sigma_max)
    if opt.sde_type == "edm_ve":
        return EDM_VE(sigma_min=opt.sigma_min, sigma_max=opt.sigma_max, edm_sigma=opt.edm_sigma)
    if opt.sde_type == "linear":
        return LinearSDE(m=opt.linear_m, k=opt.linear_k)
    if opt.sde_type == "linear_std":
        return LinearStdSDE(m=opt.linear_m, k=opt.linear_k)
    if opt.sde_type == "brownian":
        return BrownianMotionSDE(k=opt.linear_k)
    raise NotImplementedError(f"Unknown sde type: {opt.sde_type}")


class ControlledSDE(BaseSDE):
    """dX_t = ( f(t, X_t) + direction * g(t)^2 u(t, X_t) ) dt + g(t) dW_t

    `u` is the forward control (direction=+1, t: 0 -> 1) or the backward score (direction=-1, t: 1 -> 0).
    With a FlowMatching reference, the drift is the predicted velocity u(t, x) itself.
    """

    def __init__(self, ref_sde: BaseSDE, u: torch.nn.Module):
        super().__init__("controlled_sde")
        self.ref_sde = ref_sde
        self.u = u

    def sample_base_posterior(self, t, x0, x1):
        return self.ref_sde.sample_posterior(t, x0, x1)

    def randn_like(self, x):
        return self.ref_sde.randn_like(x)

    def propagate(self, x, dx):
        return self.ref_sde.propagate(x, dx)

    def diff(self, t):
        return self.ref_sde.diff(t)

    def drift(self, t, x, direction=1.0, **kwargs):
        if self.ref_sde.fm:
            return self.u(t, x)
        return self.ref_sde.drift(t, x) + direction * self.diff(t) ** 2 * self.u(t, x)


@torch.no_grad()
def sdeint(
    sde: BaseSDE,
    state0: torch.Tensor,
    timesteps: torch.Tensor,
    zero_last_step_noise: bool = False,
    only_boundary: bool = False,
    pf_ode: str | None = None,
    backward_sde: ControlledSDE | None = None,
    forward_sde: ControlledSDE | None = None,
) -> List[torch.Tensor]:
    """Integrate along `timesteps` (increasing: forward t 0->1, decreasing: backward t 1->0).

    - Default: Euler-Maruyama on `sde` (or on `backward_sde` if given).
    - `pf_ode='heun'` (backward only): Heun on the probability-flow ODE of the bridge,
          dx/dt = f(t, x) + 1/2 g(t)^2 ( u_fwd(t, x) - s_bwd(t, x) ),
      where u_fwd comes from `forward_sde` (or `sde`) and s_bwd from `backward_sde`.
      Without a forward process (score-matching baseline), u_fwd = 0.
    """
    T = len(timesteps)
    assert T > 1

    direction = 1.0 if timesteps[-1] > timesteps[0] else -1.0
    is_pf_ode = pf_ode is not None and direction < 0
    if is_pf_ode:
        assert pf_ode == "heun", f"Unsupported ODE solver: {pf_ode}"
        if backward_sde is None:
            backward_sde, sde = sde, forward_sde
    elif backward_sde is not None:
        sde = backward_sde

    for m in (sde, backward_sde, forward_sde):
        if m is not None:
            m.train(False)

    def pf_velocity(t, x):
        if backward_sde.ref_sde.fm:
            return backward_sde.u(t, x)
        if sde is None:  # no forward control
            ref = backward_sde.ref_sde
            return ref.drift(t, x) - 0.5 * (ref.diff(t) ** 2) * backward_sde.u(t, x)
        return sde.ref_sde.drift(t, x) + 0.5 * sde.diff(t) ** 2 * (sde.u(t, x) - backward_sde.u(t, x))

    state = state0.clone()
    states = [state0]
    for i in range(T - 1):
        t = timesteps[i]
        dt = timesteps[i + 1] - t

        if is_pf_ode:
            k1 = pf_velocity(t, state)
            x_predictor = state + dt * k1
            if i < T - 2:
                k2 = pf_velocity(t + dt, x_predictor)
                state = state + (dt / 2.0) * (k1 + k2)
            else:  # last step: Euler
                state = x_predictor
        else:
            if isinstance(sde, ControlledSDE):
                drift = sde.drift(t, state, direction) * dt
            else:
                drift = sde.drift(t, state) * dt
            diffusion = sde.diff(t) * dt.abs().sqrt() * sde.randn_like(state)

            zero_noise = zero_last_step_noise and i == (T - 2)
            state = sde.propagate(state, drift if zero_noise else drift + diffusion)

        if not only_boundary:
            states.append(state)

    if only_boundary:
        return states[0], state
    return states
