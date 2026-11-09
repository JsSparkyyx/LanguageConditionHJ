"""Actor and critic: plain MLPs with input normalization (residual/PLAN.md, "Network").

The actor outputs the mean of a Gaussian over u (12 numbers, one per arm joint) with a state-independent std,
initialised at ``init_std`` (0.1 of the delta bound: 0.02 rad, Yuxuan 11-09) and logged. The robot executes
delta = delta_bound * clip(u, -1, 1); PPO's ratio uses the log-probability of the unclipped u.

The critic predicts V(s) * (1 - gamma): the return lives in [0, 1 / (1 - gamma)] = [0, 100], so the network's
output stays in [0, 1]. ``value()`` returns V(s) in return units.

Observation normalization: running mean / variance (``RunningNorm``), owned by the learner, shipped to the workers
with the weights, updated from each batch AFTER that batch's PPO update (the update uses the statistics the
behaviour policy used).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


class RunningNorm(nn.Module):
    """Running mean / variance over observations (Chan's parallel update), clipped normalized output."""

    def __init__(self, dim: int, clip: float = 10.0):
        super().__init__()
        self.clip = float(clip)
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("var", torch.ones(dim, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(1e-4, dtype=torch.float64))

    @torch.no_grad()
    def update(self, x: np.ndarray | torch.Tensor) -> None:
        x = torch.as_tensor(np.asarray(x), dtype=torch.float64).reshape(-1, self.mean.shape[0])
        if x.shape[0] == 0:
            return
        b_mean = x.mean(0)
        b_var = x.var(0, unbiased=False)
        b_n = float(x.shape[0])
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean += delta * b_n / tot
        m2 = self.var * self.count + b_var * b_n + delta.pow(2) * self.count * b_n / tot
        self.var = m2 / tot
        self.count = tot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(self.var + 1e-8).to(x.dtype)
        return torch.clamp((x - self.mean.to(x.dtype)) / std, -self.clip, self.clip)


def mlp(n_in: int, hidden, n_out: int, out_gain: float) -> nn.Sequential:
    layers, d = [], n_in
    for h in hidden:
        lin = nn.Linear(d, int(h))
        nn.init.orthogonal_(lin.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.Tanh()]
        d = int(h)
    out = nn.Linear(d, n_out)
    nn.init.orthogonal_(out.weight, gain=out_gain)
    nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256), init_std: float = 0.1,
                 min_std: float = 0.01, gamma: float = 0.99, obs_clip: float = 10.0):
        super().__init__()
        self.obs_dim, self.act_dim = int(obs_dim), int(act_dim)
        self.gamma = float(gamma)
        self.min_log_std = math.log(float(min_std))
        self.norm = RunningNorm(self.obs_dim, clip=obs_clip)
        self.actor = mlp(self.obs_dim, hidden, self.act_dim, out_gain=0.01)   # mean ~ 0 at start: delta ~ 0
        self.log_std = nn.Parameter(torch.full((self.act_dim,), math.log(float(init_std))))
        self.critic = mlp(self.obs_dim, hidden, 1, out_gain=1.0)

    # ------------------------------------------------------------------------
    def std(self) -> torch.Tensor:
        return torch.exp(torch.clamp(self.log_std, min=self.min_log_std))

    def dist(self, obs_n: torch.Tensor) -> torch.distributions.Normal:
        mu = self.actor(obs_n)
        return torch.distributions.Normal(mu, self.std().expand_as(mu))

    def value_scaled(self, obs_n: torch.Tensor) -> torch.Tensor:
        """Critic output: V(s) * (1 - gamma)."""
        return self.critic(obs_n).squeeze(-1)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_scaled(self.norm(obs)) / (1.0 - self.gamma)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, float, float]:
        """One observation -> (u [act_dim], log-prob of u, V(s) in return units)."""
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32))[None]
        xn = self.norm(x)
        d = self.dist(xn)
        u = d.mean if deterministic else d.sample()
        logp = d.log_prob(u).sum(-1)
        v = self.value_scaled(xn) / (1.0 - self.gamma)
        return u[0].numpy().astype(np.float64), float(logp[0]), float(v[0])

    def evaluate(self, obs: torch.Tensor, u: torch.Tensor):
        """(log-prob of u, entropy, V(s) * (1 - gamma)) for a batch."""
        xn = self.norm(obs)
        d = self.dist(xn)
        return d.log_prob(u).sum(-1), d.entropy().sum(-1), self.value_scaled(xn)


def build_model(cfg, obs_dim: int, act_dim: int) -> ActorCritic:
    return ActorCritic(obs_dim, act_dim, hidden=tuple(cfg.hidden), init_std=cfg.init_std, min_std=cfg.min_std,
                       gamma=cfg.gamma, obs_clip=cfg.obs_clip)


def state_for_workers(model: ActorCritic) -> dict:
    """CPU copy of the weights (+ normalizer), small enough to save every update."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
