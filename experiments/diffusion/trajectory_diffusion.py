"""
Gaussian diffusion over trajectories.
Ported from https://github.com/jannerm/diffuser — self-contained, no diffuser package imports.
"""
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Schedule and helpers
# ---------------------------------------------------------------------------

def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """Cosine noise schedule (Nichol & Dhariwal 2021)."""
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.tensor(np.clip(betas, 0, 0.999), dtype=dtype)


def extract(a, t, x_shape):
    """Gather schedule values at timestep t and reshape for broadcasting."""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def apply_conditioning(x, conditions, action_dim):
    """
    Pin the observation part of the trajectory at given timestep indices.
    conditions : dict  { timestep_index : observation_tensor [batch x obs_dim] }
    """
    for t, val in conditions.items():
        x[:, t, action_dim:] = val.clone()
    return x


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class WeightedLoss(nn.Module):
    def __init__(self, weights, action_dim):
        super().__init__()
        self.register_buffer('weights', weights)
        self.action_dim = action_dim

    def forward(self, pred, targ):
        """pred, targ: [batch x horizon x transition_dim]"""
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        a0_loss = (loss[:, 0, :self.action_dim] / self.weights[0, :self.action_dim]).mean()
        return weighted_loss, {'a0_loss': a0_loss}


class WeightedL1(WeightedLoss):
    def _loss(self, pred, targ):
        return torch.abs(pred - targ)


class WeightedL2(WeightedLoss):
    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')


Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2,
}

Sample = namedtuple('Sample', 'trajectories values chains')


# ---------------------------------------------------------------------------
# Diffusion model
# ---------------------------------------------------------------------------

def _default_sample_fn(model, x, cond, t):
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)
    model_std = torch.exp(0.5 * model_log_variance)
    noise = torch.randn_like(x)
    noise[t == 0] = 0.0
    values = torch.zeros(len(x), device=x.device)
    return model_mean + model_std * noise, values


class GaussianDiffusion(nn.Module):
    """
    Denoising diffusion over fixed-length trajectories.

    Trajectories have shape [batch x horizon x transition_dim] where
    transition_dim = action_dim + observation_dim.

    Conditioning is applied externally via apply_conditioning():
    at each denoising step the observation dimensions at specific
    timestep indices (e.g. t=0 for start, t=horizon-1 for goal)
    are clamped to their given values.
    """

    def __init__(
        self,
        model,
        horizon,
        observation_dim,
        action_dim,
        n_timesteps=20,
        loss_type='l2',
        clip_denoised=False,
        predict_epsilon=False,
        action_weight=10.0,
        loss_discount=1.0,
        loss_weights=None,
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer(
            'posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            'posterior_mean_coef1',
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            'posterior_mean_coef2',
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        loss_weights = self._get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    def _get_loss_weights(self, action_weight, discount, weights_dict):
        self.action_weight = action_weight
        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)
        if weights_dict is None:
            weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def predict_start_from_noise(self, x_t, t, noise):
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(self.posterior_variance, t, x_t.shape)
        log_var = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var

    def p_mean_variance(self, x, cond, t):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))
        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        mean, var, log_var = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return mean, var, log_var

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False,
                      sample_fn=_default_sample_fn, **sample_kwargs):
        device = self.betas.device
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        chain = [x] if return_chain else None

        it = reversed(range(self.n_timesteps))
        if verbose:
            it = tqdm(it, total=self.n_timesteps, desc='diffusion sampling')
        for i in it:
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x, values = sample_fn(self, x, cond, t, **sample_kwargs)
            x = apply_conditioning(x, cond, self.action_dim)
            if return_chain:
                chain.append(x)

        inds = torch.argsort(values, descending=True)
        x = x[inds]
        values = values[inds]
        if return_chain:
            chain = torch.stack(chain, dim=1)
        return Sample(x, values, chain)

    @torch.no_grad()
    def conditional_sample(self, cond, horizon=None, **sample_kwargs):
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)
        return self.p_sample_loop(shape, cond, **sample_kwargs)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, cond, t):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape
        target = noise if self.predict_epsilon else x_start
        loss, info = self.loss_fn(x_recon, target)
        return loss, info

    def loss(self, x, *args):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, *args, t)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond, *args, **kwargs)
