"""Uniform plan retiming (the risk knob). Pure numpy."""

from __future__ import annotations

import numpy as np


def retime(
    position: np.ndarray,
    velocity: np.ndarray,
    s: float,
    dt: float = 1 / 250,
    v_limit: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Uniform time scaling. s > 1 is faster.

    New sample times t_k = k * s * dt for k = 0..K-1, with K = floor((T-1)/s),
    plus the exact final row, so T_out = K + 1. position' is per-joint linear
    interpolation of position at t_k / dt; velocity' = s * interpolated
    velocity. If v_limit is given, info["v_ratio"] = max |velocity'| / v_limit
    is reported but the velocity is NOT clipped. info also holds {"s", "T_in",
    "T_out"}.
    """
    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    if position.ndim != 2 or velocity.shape != position.shape:
        raise ValueError("position and velocity must be [T, n] with matching shapes")
    if position.shape[0] < 2:
        raise ValueError("need at least two rows to retime")
    if s <= 0:
        raise ValueError("s must be positive")
    T = position.shape[0]
    K = int(np.floor((T - 1) / s))
    if K < 1:  # degenerate (s >= T-1): keep start + final rows
        K = 1
    tau = np.arange(K) * s  # fractional row indices, tau_k = t_k / dt
    pos_out = np.vstack([_interp_rows(position, tau), position[-1:]])
    vel_out = np.vstack([_interp_rows(velocity, tau), velocity[-1:]]) * s
    info = {"s": float(s), "T_in": int(T), "T_out": int(K + 1)}
    if v_limit is not None:
        v_limit = np.asarray(v_limit, dtype=np.float64).reshape(-1)
        if np.any(v_limit <= 0):
            raise ValueError("v_limit must be positive")
        info["v_ratio"] = float(np.max(np.abs(vel_out) / v_limit[None, :]))
    else:
        info["v_ratio"] = None
    return pos_out, vel_out, info


def _interp_rows(x: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Per-column linear interpolation of [T, n] rows at fractional indices tau."""
    T = x.shape[0]
    i0 = np.clip(np.floor(tau).astype(int), 0, T - 2)
    f = np.clip(tau - i0, 0.0, 1.0)[:, None]
    return x[i0] * (1.0 - f) + x[i0 + 1] * f
