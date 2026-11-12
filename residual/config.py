"""Every knob of the residual policy, with the agreed defaults (residual/PLAN.md section 1, Yuxuan 2026-11-09).

``ResidualConfig.from_dict`` ignores unknown keys and fills missing ones with today's defaults, so a run keeps
loading after fields are added. ``apply_sets(cfg, ["k=v", ...])`` parses CLI overrides as JSON values (strings need
JSON quotes only when they look like numbers).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields

TICK_DT = 0.04          # s: one control tick (25 Hz)
STEPS_PER_TICK = 10     # physics steps per tick (250 Hz physics)
N_ARM = 6               # ur5 arm joints per arm
N_ACT = 2 * N_ARM       # the residual's dimension: both arms' joints, grippers untouched


@dataclass
class ResidualConfig:
    # ---------------------------------------------------------------- task ----
    task: str = "selective_avoid"
    embodiment: str = "ur5-wsg"
    max_physics_steps: int = 12500       # 50 s: an episode past this is a truncation (bootstrap V(s))
    max_inits_per_process: int = 200    # Env builds per process before it restarts (pybind11 TLS leak)

    # ------------------------------------------------------- controller ----
    delta_bound: float = 0.2            # rad per joint: how far from the plan the policy may put the arm
    speed_kappa: float = 1.0            # final clip = URDF velocity limit x tick x kappa (0.13 rad/tick at 3.14 rad/s)
    replan_trigger: str = "deviation"   # "deviation" | "fixed" | "none"
    replan_every: int = 5               # fixed: replan every N ticks (5 = 5 Hz), not when the plan ends within N ticks
    replan_threshold: float = 0.1       # deviation: replan when any joint is this far off its plan row (rad)
    nom_noise_std: float = 0.01         # rad, per joint per tick, added to the EXECUTED a_nom in training (Yuxuan 11-09)
    act_in_gripper_segments: bool = True  # gripper close / open run through the 25 Hz loop (Yuxuan 11-09)

    # ---------------------------------------------------------- reward ----
    gamma: float = 0.99
    w_min: float = 0.0                  # correction weight w ~ U(w_min, w_max) per episode, a policy input
    w_max: float = 2.0
    completion: str = "success"         # "success": 1/(1-gamma) only if check_success(), else 0; "always": paid anyway

    # -------------------------------------------------------- episodes ----
    in_way_frac: float = 0.5            # share of episodes whose named object is in the planner's way
    stay_frac: float = 0.5              # share of episodes whose named rule is "stay" (else "avoid")
    out_way_debt_max: int = 2           # rebuild a scene with no out-of-the-way object once out-of-way episodes lag by this
    settle_ticks: int = 50              # finished = plan done and everything slower than below for this many ticks (2 s)
    settle_cap_ticks: int = 300         # give up waiting after this many ticks (logged; counted as settled)
    settle_v_ee: float = 0.01           # m/s: both hands
    settle_v_item: float = 0.02         # m/s: every item
    settle_tilt_rate: float = 1.0       # deg/s: the named object's tilt

    # ----------------------------------------------------------- state ----
    future_ticks: tuple = (5, 10, 15, 20, 25)
    dist_clip: float = 0.5              # m: link / held distances to the named object are clipped here in the state

    # ----------------------------------------------------------- model ----
    hidden: tuple = (256, 256)
    init_std: float = 0.1               # policy std at start, in units of the delta bound (0.02 rad), logged
    min_std: float = 0.01
    obs_clip: float = 10.0

    # ------------------------------------------------------------- PPO ----
    lr: float = 3e-4
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    epochs: int = 5
    minibatch: int = 4096
    steps_per_update: int = 16384       # transitions per PPO update
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: float = 0.03             # stop the epochs early past this
    max_policy_lag: int = 1             # episodes collected with weights older than this many versions are dropped

    # ------------------------------------------------------------ eval ----
    eval_every: int = 10                # PPO updates between evaluations
    eval_in_way: int = 20
    eval_out_way: int = 20
    eval_w: float = 1.0
    eval_video_scenes: int = 4          # the first N/2 in-the-way and N/2 out-of-the-way evaluation scenes
    seed_base_train: int = 1_000_000
    seed_base_eval: int = 9_000_000

    # --------------------------------------------------------- runtime ----
    workers_per_gpu: int = 8
    wandb_project: str = "cehj-residual"
    extra: dict = field(default_factory=dict)

    # ---------------------------------------------------------- helpers ----
    @property
    def completion_value(self) -> float:
        """Value of surviving forever with no correction: sum of gamma^k = 1 / (1 - gamma) (100 at 0.99)."""
        return 1.0 / (1.0 - float(self.gamma))

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    @classmethod
    def from_dict(cls, d: dict | None) -> "ResidualConfig":
        d = dict(d or {})
        known = {f.name: f for f in fields(cls)}
        kw = {}
        for k, v in d.items():
            if k not in known:
                continue
            if isinstance(getattr(cls, k, None), tuple) and isinstance(v, list):
                v = tuple(v)
            kw[k] = v
        cfg = cls(**kw)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.replan_trigger not in ("deviation", "fixed", "none"):
            raise ValueError(f"replan_trigger must be deviation / fixed / none, got {self.replan_trigger!r}")
        if self.completion not in ("success", "always"):
            raise ValueError(f"completion must be success / always, got {self.completion!r}")
        if self.replan_trigger == "deviation" and not (0.0 < self.replan_threshold < self.delta_bound):
            # PLAN: the threshold must sit below the delta bound, or it never fires
            raise ValueError(f"replan_threshold {self.replan_threshold} must be in (0, delta_bound={self.delta_bound})")
        if not (0.0 <= self.w_min <= self.w_max):
            raise ValueError("need 0 <= w_min <= w_max")


def apply_sets(cfg: ResidualConfig, sets: list[str] | None) -> ResidualConfig:
    """``k=v`` overrides; v is parsed as JSON when it parses, else kept as a string."""
    d = cfg.to_dict()
    for item in sets or []:
        if "=" not in item:
            raise ValueError(f"--set expects k=v, got {item!r}")
        k, v = item.split("=", 1)
        k = k.strip()
        if k not in d:
            raise KeyError(f"unknown config key {k!r}")
        try:
            val = json.loads(v)
        except json.JSONDecodeError:
            val = v
        d[k] = val
    return ResidualConfig.from_dict(d)
