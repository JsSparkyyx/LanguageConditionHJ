#!/usr/bin/env python
"""Collect the key numbers of check 1 (actor_stats) into runs/lang/diag_1030/actor_stats/results.json.

json / numpy only (fine on a login node). Reads the per-run results written by actor_stats.py (vanilla/, qwen/,
vanilla_ckpt_t5full/) and the login-node motion tables (motion_login/), and adds the verdicts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "runs/lang/diag_1030/actor_stats"


def med(d, k):
    v = d.get(k)
    return None if v is None else (v.get("median") if isinstance(v, dict) else v)


def group_row(S: dict) -> dict:
    rk, av, T, tg = S["ranking"], S["actor_vs_Q"], S.get("trace", {}), S["target"]
    return {
        "n_states": S["n_states"], "h_named_median": med(S, "h_named"),
        # (a) actor
        "mu_median": med(S, "mu"), "abs_mu_p90": S["abs_mu"]["p90"],
        "sigma_p10_median_p90": [S["sigma"]["p10"], S["sigma"]["median"], S["sigma"]["p90"]],
        "log_std_raw_median": med(S, "log_std_raw"), "frac_joints_at_sigma_cap": S["frac_joints_at_sigma_cap"],
        "abs_tanh_mu_median_p90_max": [S["abs_tanh_mu_eq_abs_adet_over_dmax"][k] for k in ("median", "p90", "max")],
        "entropy_analytic_mean": S["entropy_analytic_sum"]["mean"], "entropy_mc_mean": S["entropy_mc"]["mean"],
        "norm_adet_rad_median": med(S, "norm_adet_rad"), "norm_aref_rad_median": med(S, "norm_aref_rad"),
        "ratio_norm_adet_over_aref_median": med(S, "ratio_norm_adet_over_aref"),
        "cos_adet_aref_median": med(S, "cos_adet_aref"), "frac_adet_near_zero_max_joint_below_0.05": S["adet_near_zero_frac"],
        # (b) critic
        "Q_stored_median": med(S, "Q_stored"), "frac_Q_stored_below_0": S["frac_Q_stored_below_0"],
        "spread_over_c_abs_median_p90": [S["spread_over_c_abs"]["median"], S["spread_over_c_abs"]["p90"]],
        "spread_16_random_abs_median_p90_max": [S["spread_16_random_abs"][k] for k in ("median", "p90", "max")],
        "spread_16_random_rel_absQ_median": med(S, "spread_16_random_rel_absQ"),
        "sens_rel_logged_way_fp32": S["sens_rel_logged_way_fp32"], "sens_rel_logged_way_tf32": S["sens_rel_logged_way_tf32"],
        "tf32_minus_fp32_absQ_median_p90": [S["tf32_minus_fp32_abs"]["median"], S["tf32_minus_fp32_abs"]["p90"]],
        "gradQ_times_dmax_l2_median_at_aref_zero_adet": [med(S, "gradQ_times_dmax_l2_at_aref"),
                                                         med(S, "gradQ_times_dmax_l2_at_zero"),
                                                         med(av, "gradQ_times_dmax_l2_at_adet")],
        "Q0_minus_Qhalf_median": rk["Q0_minus_Qhalf"]["median"], "Qhalf_minus_Qnom_median": rk["Qhalf_minus_Qnom"]["median"],
        "Q0_minus_Qnom_median": rk["Q0_minus_Qnom"]["median"],
        "frac_Q0_gt_Qhalf": rk["frac_Q0_gt_Qhalf"], "frac_Qhalf_gt_Qnom": rk["frac_Qhalf_gt_Qnom"],
        "frac_order_stop_gt_slow_gt_nom": rk["frac_order_stop_gt_slow_gt_nom"],
        "frac_slow_best_of_three": rk["frac_slow_best_of_three"],
        "frac_abs_Q0_minus_Qhalf_above_tf32_noise_p90": rk["frac_abs_Q0_minus_Qhalf_above_tf32_noise_p90"],
        "argmax_c_hist": rk["argmax_c_hist"],
        "Q_best_candidate_minus_Q_det_median_p90": [av["Q_best_candidate_minus_Q_det"]["median"],
                                                    av["Q_best_candidate_minus_Q_det"]["p90"]],
        "frac_best_candidate_beats_det_by_0.001": av["frac_best_candidate_beats_det_by_0.001"],
        # (c) where the action signal dies (critic twin 1)
        "body_norm_after_injection": med(T, "c1_enc_body_norm_after_injection"),
        "body_norm_after_trunk_blocks": [med(T, f"c1_enc_body_norm_after_block{k}") for k in range(3)],
        "query_norm_after_trunk": med(T, "c1_enc_query_norm_after_block2"),
        "act_proj_out_norm_a0_aref_sat": [med(T, "c1_inj_norm_zero"), med(T, "c1_inj_norm_aref"), med(T, "c1_inj_norm_sat")],
        "inj_delta_abs_aref_sat": [med(T, "c1_aref_inj_delta_abs"), med(T, "c1_sat_inj_delta_abs")],
        "inj_delta_over_body_aref_sat": [med(T, "c1_aref_inj_delta_over_body"), med(T, "c1_sat_inj_delta_over_body")],
        "rel_change_body_after_critic_blocks_sat": [med(T, f"c1_sat_hb{k}_rel") for k in range(3)],
        "rel_change_query_after_critic_blocks_sat": [med(T, f"c1_sat_hq{k}_rel") for k in range(3)],
        "dV_max_sat": med(T, "c1_sat_dV_max"), "dQ_sat": med(T, "c1_sat_dQ"),
        "V_body_min_minus_Q": med(T, "c1_V_body_min_minus_Q"), "V_query_min_minus_Q": med(T, "c1_V_query_min_minus_Q"),
        "softmin_argmin_is_query_frac": T.get("c1_softmin_argmin_is_query_frac"),
        "planner_cmd_dp_mm_mean_over_links": med(T, "eff_dp_norm_mm"),
        # (d) target
        "y_stoch_mean": tg["y_stoch_mean"]["mean"], "y_std_over_8_actor_samples_median": tg["y_stoch_std_over_8_samples"]["median"],
        "abs_ystoch_minus_ydet_median_p90": [tg["abs_ystoch_minus_ydet"]["median"], tg["abs_ystoch_minus_ydet"]["p90"]],
        "abs_ystoch_minus_ynext_median": tg["abs_ystoch_minus_ynext"]["median"],
        "abs_ystoch_minus_yzero_median": tg["abs_ystoch_minus_yzero"]["median"],
        "td_y_minus_Q_median": tg["td_y_stoch_minus_Q_stored"]["median"],
    }


def run_table(path: Path) -> dict:
    R = json.loads(path.read_text())
    out = {"source": str(path.relative_to(REPO)), "run": R["run"], "encoder": R["encoder"], "groups_n": R["groups"],
           "checkpoints": {}}
    for lab, C in R["checkpoints"].items():
        L = C["logged"]
        out["checkpoints"][lab] = {
            "step": C["step"], "round": C["round"], "target_src": C["target_src"],
            "logged_window": L["window_steps"],
            "logged_entropy_mean": (L.get("entropy") or {}).get("mean"),
            "logged_action_sens_rel_mean_min_max": [(L.get("critic/action_sens_rel") or {}).get(k) for k in ("mean", "min", "max")],
            "logged_actor_action_std_mean": (L.get("actor/action_std") or {}).get("mean"),
            "logged_grad_actor_mean": (L.get("grad/actor") or {}).get("mean"),
            "all_states_entropy_analytic_mean": C["all_states_entropy_analytic_mean"],
            "all_states_sigma_median": C["all_states_sigma_median"],
            "all_states_frac_joints_at_sigma_cap": C["all_states_frac_joints_at_sigma_cap"],
            "all_states_abs_tanh_mu_median": C["all_states_abs_tanh_mu_median"],
            "all_states_sens_rel_logged_way_fp32": C.get("all_states_sens_rel_logged_way_fp32"),
            "all_states_sens_rel_logged_way_tf32": C.get("all_states_sens_rel_logged_way_tf32"),
            "groups": {g: group_row(S) for g, S in C["groups"].items()},
        }
    if "motion" in R:
        M = R["motion"]
        fs = M["filter_switches_while_moving"]["first_switch_per_episode"]
        out["motion_t5"] = {
            k: {kk: v[kk] for kk in ("ticks", "abs_cmd_over_dmax_per_joint", "max_link_disp_mm",
                                     "frac_ticks_link_disp_below_0.5mm", "realized_over_cmd_summed", "tray_speed_mps")}
            for k, v in M.items() if isinstance(v, dict) and "ticks" in v}
        out["motion_t5"]["actor_by_round"] = M.get("ring/filter/src1", {}).get("by_round")
        pb = M["planner_brakes"]
        out["planner_brakes_t5"] = {"n": pb["n"], "peak_decel_tipped": pb["peak_decel_tipped"],
                                    "peak_decel_not_tipped": pb["peak_decel_not_tipped"]}
        vp = np.array([x["v_prev"] for x in fs])
        pdc = np.array([x["peak_decel_next4"] for x in fs])
        hm = np.array([x["h_tall_min_next25"] for x in fs])
        out["first_filter_switch_while_tray_moves"] = {
            "n_episodes": len(fs), "tray_speed_before_median_min_max": [float(np.median(vp)), float(vp.min()), float(vp.max())],
            "peak_decel_next4_ticks_median_min_max": [float(np.median(pdc)), float(pdc.min()), float(pdc.max())],
            "n_tall_h_below_0_within_25_ticks": int((hm < 0).sum()),
            "n_tall_tipped_later_in_episode": int(sum(x["tall_tipped_later"] for x in fs)),
            "a_tip_tall_block_example_mps2": 9.81 * 0.032 / 0.1664, "rows": fs}
    return out


def main() -> None:
    res = {"check": "diag_1030 check 1: actor statistics and the critic's action response",
           "code": ["scripts/lang/diag_1030/actor_stats.py", "scripts/lang/diag_1030/actor_stats_digest.py",
                    "scripts/lang/diag_1030/actor_stats_summary.py"],
           "jobs": "23354241 (gpuA100x8-interactive, 1 A100-40GB; smoke + srun steps inside it)",
           "definitions": {
               "h_units": "h and Q in [-1, 1]: +1 comfortably safe, 0 rule just broken, -1 badly broken",
               "dtheta_max": "per-joint box half-width per tick: 0.126 / 0.128 rad (UR5 joints)",
               "mu, sigma": "pre-tanh mean and exp(log_std clamped to [-5, 0]) of each joint's Gaussian; a_det = tanh(mu) dtheta_max",
               "a_ref": "the stored command of the row (planner a_nom on planner ticks, the actor's command on actor ticks)",
               "spread_16_random": "max - min of min(Q1, Q2) over 16 uniform random actions in the box (arm joints)",
               "sens_rel_logged_way": "mean |Q(r1) - Q(r2)| / mean |Q| for two uniform random actions (as _diagnostics logs it)",
               "tf32_noise": "|Q_tf32 - Q_fp32| at the same action (the training step runs in TF32)",
               "groups": "t5_plan_soon: planner ticks of the T5 drag (tray > 2 cm/s), tall block upright, h >= 0 and fails_soon "
                         "(tips within 25 ticks); t5_plan_later: tips later; t5_plan_safe: never tips in the episode; "
                         "t5_actor_pretip / posttip: actor ticks of the ring's T5 filter episodes before / after the tall "
                         "block first tipped; __short: the same states with the short block's rule named; plan_<task> / "
                         "actor_<task>: planner / actor ticks of the other tasks, rule drawn as training draws it; "
                         "train_mix: LangSampler.plan draws (task-balanced, 50 % harmful, half paired)",
           },
           "runs": {}}
    for name in ("vanilla", "vanilla_ckpt_t5full", "qwen"):
        p = OUT / name / "results.json"
        if p.exists():
            res["runs"][name] = run_table(p)
    p = OUT / "motion_login" / "actor_ticks_all_tasks.json"
    if p.exists():
        res["actor_ticks_all_tasks_all_runs"] = json.loads(p.read_text())
    res["tanh_gauss_max_entropy"] = {"sigma_star": 0.8744, "H_per_joint_nats": 0.68364, "H_12_joints": 8.2036,
                                     "H_12_joints_at_mu0_sigma1": 8.0376, "uniform_log2_per_joint": 0.6931}
    vfile = OUT / "verdicts.json"
    if vfile.exists():
        res["verdicts"] = json.loads(vfile.read_text())
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
