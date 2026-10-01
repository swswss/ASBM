import torch
from torch.utils.data import DataLoader

from components.buffer import BatchBuffer
from components.prior import TerminalCost
from components.sdes import ControlledSDE, LinearSDE, LinearStdSDE, VPSDE, sdeint


class Matcher:
    def __init__(self, sde: ControlledSDE, buffer: BatchBuffer, resample_size: int, duplicates: int = 1, min_t0: float = 0.0):
        self.sde = sde
        self.buffer = buffer
        self.resample_size = resample_size
        self.duplicates = duplicates
        self.min_t0 = min_t0

    def build_dataloader(self, batch_size, duplicates=None) -> DataLoader:
        dataset = self.buffer.build_dataset(self.duplicates if duplicates is None else duplicates)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    def sample_t(self, batch_size):
        return torch.rand(batch_size, 1) * (1 - self.min_t0) + self.min_t0


class AdjointMatcher(Matcher):
    """Adjoint matching for the forward control u(t, x).

    Simulates (x0, x1) with the current controlled forward SDE, computes the terminal adjoint
    a_1 = grad E(x1) + corrector(x1), and regresses u(t, x_t) onto -a_t with x_t ~ p^{base}_{t|0,1}.
    """

    def __init__(self, terminal_cost: TerminalCost, **kwargs):
        super().__init__(**kwargs)
        self.terminal_cost = terminal_cost

    @torch.no_grad()
    def populate_buffer(self, x0, timesteps, is_init_stage, zero_last_step_noise=False):
        sde = self.sde.ref_sde if is_init_stage else self.sde
        (x0, x1) = sdeint(sde, x0, timesteps, zero_last_step_noise=zero_last_step_noise, only_boundary=True)
        adjoint1 = self.terminal_cost(x1, is_init_stage=is_init_stage).clone()
        assert x1.shape == adjoint1.shape == x0.shape
        self.buffer.add({"x0": x0.to("cpu"), "x1": x1.to("cpu"), "adjoint1": adjoint1.to("cpu")})

    def prepare_target(self, data, device):
        x0 = data["x0"].to(device)
        x1 = data["x1"].to(device)
        adjoint = data["adjoint1"].to(device)  # constant in time up to the reference-SDE scaling below

        t = self.sample_t(x0.shape[0]).to(device)
        xt = self.sde.sample_base_posterior(t, x0, x1)

        ref_sde = self.sde.ref_sde
        if isinstance(ref_sde, VPSDE):
            adjoint = (adjoint.reshape(adjoint.shape[0], -1) * torch.exp(ref_sde.coeff2(t))).reshape(adjoint.shape)
        elif isinstance(ref_sde, (LinearSDE, LinearStdSDE)):
            adjoint = (adjoint.reshape(adjoint.shape[0], -1) * ref_sde.adjoint_weight(t)).reshape(adjoint.shape)

        assert t.shape == (xt.shape[0], 1) and adjoint.shape == xt.shape
        return (t, xt), -adjoint


class CorrectorMatcher(Matcher):
    """Corrector matching: regresses h(1, x1) onto grad log p^{base}_{1|0}(x1 | x0) for (x0, x1) from the forward SDE."""

    @torch.no_grad()
    def populate_buffer(self, x0, timesteps, is_init_stage, zero_last_step_noise=False):
        if is_init_stage:  # first stage: (x0, x1) from the uncontrolled reference transition
            mu, var = self.sde.ref_sde._pt_gauss_param(1.0, x0)
            x1 = mu + var**0.5 * torch.randn_like(mu)
        else:
            (x0, x1) = sdeint(self.sde, x0, timesteps, zero_last_step_noise=zero_last_step_noise, only_boundary=True)
        assert x1.shape == x0.shape
        self.buffer.add({"x0": x0.to("cpu"), "x1": x1.to("cpu")})

    def prepare_target(self, data, device):
        x0 = data["x0"].to(device)
        x1 = data["x1"].to(device)
        t1 = torch.ones(x0.shape[0], 1).to(device)
        score = self.sde.ref_sde.cond_score(x0, t1, x1)
        return (t1, x1), score


class ScoreMatcher(CorrectorMatcher):
    """Backward (score) matching on the bridge p^{base}_{t|0,1} with (x0, x1) from the learned forward SDE.

    Returns ((t, x_t), target, weight); the loss is || weight * s(t, x_t) - target ||^2.
    """

    def __init__(self, flow_matching=False, **kwargs):
        super().__init__(**kwargs)
        self.fm = flow_matching

    def prepare_target(self, data, device):
        x0 = data["x0"].to(device)
        x1 = data["x1"].to(device)
        t = self.sample_t(x0.shape[0]).to(device)

        if self.fm:
            original_shape = x0.shape
            x0 = x0.reshape(x0.shape[0], -1)
            x1 = x1.reshape(x1.shape[0], -1)
            vt = (x1 - x0).reshape(original_shape)
            xt = (t * x1 + (1 - t) * x0).reshape(original_shape)
            return (t, xt), vt, 1.0

        xt = self.sde.sample_base_posterior(t, x0, x1)
        loc, xt, var = self.sde.ref_sde.cond_score(x0, t, xt, return_raw=True)
        std = torch.sqrt(var)
        std = std.reshape(std.shape[0], *([1] * (xt.ndim - 1)))
        score = (loc - xt) / std  # = std * grad log p_{t|0}(x_t | x0)
        return (t, xt), score, std
