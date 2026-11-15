"""PPO: advantages over episodes and the clipped update.

An episode arrives as a dict of per-transition arrays (``episode.Episode.to_arrays``):

- ``obs`` [T, D], ``u`` [T, A] (the unclipped Gaussian sample), ``logp`` [T], ``value`` [T] (V(s_t) of the behaviour
  critic, return units);
- ``rew`` [T]: the discounted reward summed inside the transition (one tick, or more when ticks ran without the
  policy), ``disc`` [T]: gamma^(ticks in the transition);
- ``end``: how it ended, one of "violation" (terminal, nothing after), "finished" (terminal, plus the completion
  value ``boot``), "truncated" (bootstrap with ``last_value`` = V(s_T)).

GAE with a per-transition discount: delta_t = r_t + disc_t * V_next - V_t, A_t = delta_t + disc_t * lambda * A_{t+1}.
"""

from __future__ import annotations

import numpy as np
import torch


def episode_advantages(ep: dict, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """(advantages [T], returns [T]) of one episode."""
    rew = np.asarray(ep["rew"], dtype=np.float64)
    disc = np.asarray(ep["disc"], dtype=np.float64)
    val = np.asarray(ep["value"], dtype=np.float64)
    T = rew.shape[0]
    if T == 0:
        return np.zeros(0), np.zeros(0)
    end = ep["end"]
    if end == "violation":
        v_last = 0.0
    elif end == "finished":
        v_last = float(ep["boot"])
    elif end == "truncated":
        v_last = float(ep["last_value"])
    else:
        raise ValueError(f"unknown episode end {end!r}")
    v_next = np.append(val[1:], v_last)
    delta = rew + disc * v_next - val
    adv = np.zeros(T, dtype=np.float64)
    a = 0.0
    for t in range(T - 1, -1, -1):
        a = delta[t] + (disc[t] * lam * a if t < T - 1 else 0.0)
        adv[t] = a
    return adv, adv + val


def make_batch(episodes: list[dict], lam: float) -> dict:
    obs, u, logp, adv, ret, val = [], [], [], [], [], []
    for ep in episodes:
        if len(ep["rew"]) == 0:
            continue
        a, r = episode_advantages(ep, lam)
        obs.append(ep["obs"]); u.append(ep["u"]); logp.append(ep["logp"])
        adv.append(a); ret.append(r); val.append(ep["value"])
    if not obs:
        return {"n": 0}
    return {
        "obs": np.concatenate(obs).astype(np.float32),
        "u": np.concatenate(u).astype(np.float32),
        "logp": np.concatenate(logp).astype(np.float32),
        "adv": np.concatenate(adv).astype(np.float32),
        "ret": np.concatenate(ret).astype(np.float32),
        "value": np.concatenate(val).astype(np.float32),
        "n": int(sum(len(x) for x in adv)),
    }


def ppo_update(model, opt, batch: dict, cfg, rng: np.random.Generator) -> dict:
    """Clipped PPO epochs over one batch; returns diagnostics."""
    n = batch["n"]
    obs = torch.as_tensor(batch["obs"])
    u = torch.as_tensor(batch["u"])
    old_logp = torch.as_tensor(batch["logp"])
    adv = torch.as_tensor(batch["adv"])
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    scale = 1.0 - float(cfg.gamma)
    ret_s = torch.as_tensor(batch["ret"]) * scale        # value target in critic units
    old_v_s = torch.as_tensor(batch["value"]) * scale
    stats = {"loss_pi": [], "loss_v": [], "entropy": [], "kl": [], "clipfrac": []}
    mb = max(1, min(int(cfg.minibatch), n))
    stopped_early = False
    epochs_done = 0
    steps = 0
    per_mb = getattr(cfg, "kl_check", "epoch") == "minibatch"
    for _epoch in range(int(cfg.epochs)):
        perm = rng.permutation(n)
        kls = []
        for s in range(0, n, mb):
            idx = torch.as_tensor(perm[s:s + mb])
            logp, ent, v_s = model.evaluate(obs[idx], u[idx])
            log_ratio = logp - old_logp[idx]
            ratio = torch.exp(log_ratio)
            if per_mb and cfg.target_kl:
                with torch.no_grad():
                    kl_now = float(((ratio - 1) - log_ratio).mean())
                if kl_now > 1.5 * float(cfg.target_kl):
                    stopped_early = True
                    break
            a = adv[idx]
            loss_pi = -torch.min(ratio * a, torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * a).mean()
            # clipped value loss (PPO2 style), in critic units
            v_clip = old_v_s[idx] + torch.clamp(v_s - old_v_s[idx], -cfg.clip_eps * 0.5, cfg.clip_eps * 0.5)
            loss_v = 0.5 * torch.max((v_s - ret_s[idx]) ** 2, (v_clip - ret_s[idx]) ** 2).mean()
            loss = loss_pi + cfg.value_coef * loss_v - cfg.entropy_coef * ent.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            steps += 1
            with torch.no_grad():
                kl = float(((ratio - 1) - log_ratio).mean())
                kls.append(kl)
                stats["loss_pi"].append(float(loss_pi))
                stats["loss_v"].append(float(loss_v))
                stats["entropy"].append(float(ent.mean()))
                stats["kl"].append(kl)
                stats["clipfrac"].append(float(((ratio - 1).abs() > cfg.clip_eps).float().mean()))
        if stopped_early:
            break
        epochs_done += 1
        if not per_mb and cfg.target_kl and np.mean(kls) > cfg.target_kl:
            stopped_early = True
            break
    with torch.no_grad():
        ret = batch["ret"].astype(np.float64)
        val = batch["value"].astype(np.float64)
        ev = 1.0 - np.var(ret - val) / max(np.var(ret), 1e-8)
    out = {k: float(np.mean(v)) if v else float("nan") for k, v in stats.items()}
    out.update({"explained_var": float(ev), "epochs": epochs_done, "kl_stop": float(stopped_early), "steps": steps,
                "adv_raw_std": float(np.std(batch["adv"])), "ret_mean": float(np.mean(ret)), "n": n})
    return out
