"""One-step distillation of a teacher (ASBM, or the score-matching baseline) for CIFAR-10.

The one-step generator G(z) is trained with
  - a distribution-matching loss (DMD-style): the teacher score and a "fake" score (trained online on G's samples)
    are compared on noised generator samples, and
  - a regression loss to (noise, teacher ODE sample) pairs.
For the ASBM teacher, generator samples are noised with the reference bridge between x0 = G(z) and x1 = z;
for the score-matching teacher, with the VP marginal p_{t|0}.
"""
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.aggregation import MeanMetric
from torchvision.utils import save_image
from tqdm.auto import tqdm

import data
import util
from components.sdes import build_reference_sde, sdeint
from runner import build_unet


class OneStepGenerator(nn.Module):
    """x0 = G(z): a score network evaluated at a fixed time `condition_t`, converted to an x0-prediction."""

    def __init__(self, base_model, beta0, beta1, conversion_method="tweedie", condition_t=1.0):
        super().__init__()
        self.base = base_model
        self.register_buffer("beta0", torch.as_tensor(beta0, dtype=torch.float).clone(), persistent=False)
        self.register_buffer("beta1", torch.as_tensor(beta1, dtype=torch.float).clone(), persistent=False)
        self.condition_t = condition_t
        assert conversion_method in ["tweedie", "vp", "scratch"]
        self.conversion_method = conversion_method

    def beta_t(self, t):
        return torch.lerp(self.beta0, self.beta1, t)

    def alpha_sigma(self, t):
        t = t.view(t.shape[0], 1, 1, 1)
        kappa_bar = -0.25 * t * (self.beta_t(t) + self.beta0)
        alpha = torch.exp(kappa_bar)
        sigma = (1 - torch.exp(2 * kappa_bar)).clamp(min=0.0).sqrt()
        return alpha, sigma

    def forward(self, z):
        B = z.shape[0]
        t = torch.full((B, 1), self.condition_t, device=z.device)
        if self.conversion_method == "tweedie":
            score = self.base(t, z)
            alpha, sigma = self.alpha_sigma(t)
            x0_hat = (z.double() + (sigma * sigma) * score.double()) / (alpha + 1e-8)
            return x0_hat.float()
        if self.conversion_method == "vp":
            score = self.base(t, z)
            int_beta_t = (0.5 * t * (self.beta_t(t) + self.beta0)).reshape(B, 1, 1, 1)  # 1/2 int_0^t beta(s) ds
            return z + 0.5 * int_beta_t * z + int_beta_t * score
        return self.base(t, z)


class PairSampler:
    """Iterates over the stored (noise, teacher sample) pairs in random order, reshuffling after each pass."""

    def __init__(self, path, seed=0):
        self.pairs = np.load(path, mmap_mode="r")  # [N, 2, C, H, W]
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(len(self.pairs))
        self.idx = 0

    def sample(self, bs):
        out = []
        while len(out) < bs:
            if self.idx >= len(self.order):
                self.rng.shuffle(self.order)
                self.idx = 0
            take = min(bs - len(out), len(self.order) - self.idx)
            out.extend(self.order[self.idx: self.idx + take].tolist())
            self.idx += take
        arr = torch.from_numpy(np.asarray(self.pairs[out], dtype=np.float32))
        return arr[:, 0], arr[:, 1]


@torch.no_grad()
def teacher_sample(run, z, nfe):
    """Teacher samples with the probability-flow ODE (Heun), t: 1 -> 1/(2 nfe) in `nfe` points, then -> 0."""
    timesteps = torch.linspace(1.0, (1 / nfe) / 2, steps=nfe)
    timesteps = torch.cat([timesteps, torch.tensor([0.0])]).to(run.device)
    _, x0 = sdeint(
        None if run.opt.score_matching else run.sde_f_ema,
        z,
        timesteps,
        zero_last_step_noise=True,
        only_boundary=True,
        pf_ode="heun",
        backward_sde=run.sde_b_ema,
    )
    return x0


@torch.no_grad()
def build_pairs(run, path):
    opt, device = run.opt, run.device
    N, bs = opt.distill_num_pairs, 1024
    tmp = path + ".tmp.npy"
    pairs = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(N, 2, *opt.data_dim))
    for start in tqdm(range(0, N, bs), desc="Teacher pairs"):
        n = min(bs, N - start)
        z = torch.randn(n, *opt.data_dim).to(device)
        x0 = teacher_sample(run, z.clone(), opt.distill_reg_nfe)
        pairs[start: start + n, 0] = z.cpu().numpy()
        pairs[start: start + n, 1] = x0.cpu().numpy()
    pairs.flush()
    del pairs
    os.replace(tmp, path)


def set_requires_grad(net, flag):
    for p in net.parameters():
        p.requires_grad_(flag)


def train_distill(run):
    opt, device = run.opt, run.device
    is_sm = opt.score_matching
    ref = run.ref_sde

    # ---------- models ----------
    teacher = copy.deepcopy(run.ema_b)
    if opt.distill_conv == "scratch":
        base = build_unet(opt.num_res_blocks_sm, opt.data_dim)
    else:
        base = copy.deepcopy(teacher)  # initialize the generator from the teacher score network
    G = OneStepGenerator(base, ref.beta0, ref.beta1, opt.distill_conv, opt.distill_cond_t).to(device)
    fake_score = copy.deepcopy(teacher).to(device)
    teacher.requires_grad_(False)
    teacher.eval()
    G.requires_grad_(True)
    fake_score.requires_grad_(True)

    opt_fake = torch.optim.AdamW(fake_score.parameters(), lr=opt.distill_lr, betas=(0.9, 0.999), weight_decay=0.01)
    opt_G = torch.optim.AdamW([p for p in G.parameters() if p.requires_grad], lr=opt.distill_lr,
                              betas=(0.9, 0.999), weight_decay=0.01)

    with torch.no_grad():
        x = teacher_sample(run, torch.randn(64, *opt.data_dim).to(device), opt.distill_reg_nfe)
        save_image((x.clip(-1, 1) + 1) / 2, os.path.join(opt.ckpt_path, "teacher_samples.png"), nrow=8)

    # ---------- regression pairs ----------
    use_reg = opt.distill_lambda_reg > 0
    if use_reg:
        pair_path = opt.distill_pair_path or os.path.join(opt.ckpt_path, f"pairs_{opt.distill_reg_nfe}nfe.npy")
        if os.path.exists(pair_path):
            run.print_log(f"Use the teacher pairs in {pair_path}")
        else:
            run.print_log(f"Generate {opt.distill_num_pairs} teacher pairs into {pair_path}")
            build_pairs(run, pair_path)
        pair_sampler = PairSampler(pair_path)

    # ---------- helpers ----------
    t_min, t_max = 1e-5, 1.0

    def sample_t(B):
        return t_min + (t_max - t_min) * torch.rand(B, 1, device=device)

    def forward_diffuse(x0, t):
        """VP marginal: x_t = alpha_t x0 + sigma_t eps (score-matching teacher)."""
        kappa_bar = ref.coeff1(t.reshape(t.shape[0], 1, 1, 1))
        alpha = torch.exp(kappa_bar)
        sigma = (1 - torch.exp(2 * kappa_bar)).clamp(min=0.0).sqrt()
        eps = torch.randn_like(x0)
        return alpha * x0 + sigma * eps, alpha, sigma, eps

    def dm_weight(x1_coeff):
        if opt.distill_weight == "one":
            return 1.0
        if opt.distill_weight == "bt":
            w = x1_coeff
        else:  # bt_half
            w = x1_coeff * opt.distill_w_scaler + (1 - opt.distill_w_scaler)
        return w.reshape(w.shape[0], 1, 1, 1)

    # ---------- training ----------
    dataset = data.build_dataset(opt)
    loader = data.build_loader(dataset, opt.distill_bs, shuffle=True)

    G.eval()
    with torch.no_grad():
        x = G(torch.randn(64, *opt.data_dim).to(device))
        save_image((x.clip(-1, 1) + 1) / 2, os.path.join(opt.ckpt_path, "generator_init.png"), nrow=8)
    G.train(True)
    fake_score.train(True)

    step = 0
    for epoch in range(opt.distill_epochs):
        epoch_loss = MeanMetric().to(device)
        pbar = tqdm(loader, desc=f"Distill ep {epoch + 1}", total=len(loader))
        for batch in pbar:
            step += 1
            x_real = batch[0].to(device, non_blocking=True)  # only sets the batch shape
            B = x_real.shape[0]
            gen_step = step % opt.distill_update_ratio == 0

            z = torch.randn_like(x_real)
            if gen_step:
                x_fake = G(z)
            else:
                with torch.no_grad():
                    x_fake = G(z)
            x_fake_det = x_fake.detach()

            # ----- (A) generator update -----
            loss_dm_val, loss_reg_val = 0.0, 0.0
            if gen_step:
                opt_G.zero_grad(set_to_none=True)
                if use_reg:
                    z_reg, img_reg = pair_sampler.sample(opt.distill_bs)
                    img_reg = (img_reg.clip(-1, 1) / 2 + 0.5).to(device)
                    x_fake_reg = G(z_reg.to(device)) / 2 + 0.5

                set_requires_grad(fake_score, False)
                with torch.no_grad():
                    t_dm = sample_t(B)
                    if is_sm:
                        x_t, alpha, sigma, _ = forward_diffuse(x_fake, t_dm)
                    else:
                        x_t, _, x1_coeff, _ = ref.sample_posterior(t_dm, x_fake, z, return_coeffs=True)
                    s_real = teacher(t_dm, x_t)
                    s_fake = fake_score(t_dm, x_t)
                    if is_sm:  # scores -> x0 predictions (Tweedie)
                        x0_real = ((x_t.double() + sigma**2 * s_real.double()) / alpha).float()
                        x0_fake = ((x_t.double() + sigma**2 * s_fake.double()) / alpha).float()
                        s_real, s_fake = x0_real - x_fake, x0_fake - x_fake
                        w = 1 / torch.abs(s_real).mean(dim=[1, 2, 3], keepdim=True)
                    else:
                        w = dm_weight(x1_coeff)
                    v = torch.nan_to_num(w * (s_fake - s_real))

                loss_dm = 0.5 * F.mse_loss(x_fake, (x_fake - v).detach(), reduction="mean")
                loss_G = opt.distill_lambda_dm * loss_dm
                loss_dm_val = float(loss_dm.detach())
                if use_reg:
                    loss_reg = F.mse_loss(img_reg, x_fake_reg)
                    loss_G = loss_G + opt.distill_lambda_reg * loss_reg
                    loss_reg_val = float(loss_reg.detach())
                set_requires_grad(fake_score, True)

                loss_G.backward()
                torch.nn.utils.clip_grad_norm_(G.parameters(), 10.0)
                opt_G.step()
                opt_fake.zero_grad(set_to_none=True)

            # ----- (B) fake-score update (every step), on the generator's current samples -----
            opt_fake.zero_grad(set_to_none=True)
            t_fake = sample_t(B)
            if is_sm:
                x_t_fake, _, weight, eps = forward_diffuse(x_fake_det, t_fake)
                target = -eps
            else:
                x_t_fake = ref.sample_posterior(t_fake, x_fake_det, z)
                loc, x_t_fake, var = ref.cond_score(x_fake_det, t_fake, x_t_fake, return_raw=True)
                weight = torch.sqrt(var).reshape(var.shape[0], *([1] * (x_t_fake.ndim - 1)))
                target = (loc - x_t_fake) / weight
            loss_fake = torch.mean((weight * fake_score(t_fake, x_t_fake) - target) ** 2)
            loss_fake.backward()
            torch.nn.utils.clip_grad_norm_(fake_score.parameters(), 10.0)
            opt_fake.step()

            epoch_loss.update(float(loss_fake.detach()))
            pbar.set_postfix(loss_fake=float(loss_fake.detach()), loss_dm=loss_dm_val, loss_reg=loss_reg_val)

        run.print_log(f"[distill | ep={epoch + 1}] loss_fake={float(epoch_loss.compute().detach().cpu()):.4f}")
        if (epoch + 1) % opt.distill_save_freq == 0 or epoch + 1 == opt.distill_epochs:
            G.eval()
            with torch.no_grad():
                x = G(torch.randn(64, *opt.data_dim).to(device))
                save_image((x.clip(-1, 1) + 1) / 2, os.path.join(opt.ckpt_path, f"distill_{epoch + 1}ep_samples.png"), nrow=8)
            ckpt = {"one_step_generator": G.state_dict(), "fake_score_model": fake_score.state_dict()}
            torch.save(ckpt, os.path.join(opt.ckpt_path, f"distill_{epoch + 1}ep.pt"))
            print(util.green(f"checkpoint saved: distill_{epoch + 1}ep.pt"))
            G.train(True)


def load_generator(opt):
    ref = build_reference_sde(opt)
    base = build_unet(opt.num_res_blocks_sm, opt.data_dim)
    G = OneStepGenerator(base, ref.beta0, ref.beta1, opt.distill_conv, opt.distill_cond_t)
    G.load_state_dict(torch.load(opt.load, map_location="cpu")["one_step_generator"])
    return G.to(opt.device).eval()


@torch.no_grad()
def generate(G, opt, out_dir, num_samples, batch_size):
    os.makedirs(out_dir, exist_ok=True)
    n_written = 0
    pbar = tqdm(total=num_samples, desc="Generating (1 step)")
    while n_written < num_samples:
        z = torch.randn(batch_size, *opt.data_dim, device=opt.device)
        x = (G(z).clip(-1, 1) / 2 + 0.5)[: num_samples - n_written].cpu()
        for i in range(x.shape[0]):
            save_image(x[i], os.path.join(out_dir, f"{n_written + i + 1:06d}.png"))
        n_written += x.shape[0]
        pbar.update(x.shape[0])
    pbar.close()
