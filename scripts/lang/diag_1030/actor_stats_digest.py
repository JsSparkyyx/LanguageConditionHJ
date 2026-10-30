#!/usr/bin/env python
"""Print the key numbers of an actor_stats results.json as plain tables (json only, no torch: fine on a login node).

    python3 scripts/lang/diag_1030/actor_stats_digest.py runs/lang/diag_1030/actor_stats/vanilla/results.json
"""

from __future__ import annotations

import json
import sys


def g(d, *ks, default=float("nan")):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def f(x, p=4):
    if x is None:
        return "-"
    if isinstance(x, float):
        if x != x:
            return "nan"
        if x != 0 and (abs(x) < 10 ** -(p - 1) or abs(x) >= 1e4):
            return f"{x:.{max(p - 2, 1)}e}"
        return f"{x:.{p}f}".rstrip("0").rstrip(".") if "." in f"{x:.{p}f}" else f"{x:.{p}f}"
    return str(x)


def main(path: str) -> None:
    R = json.load(open(path))
    print(f"# {path}\nrun {R['run']} encoder {R['encoder']} gamma {R['gamma']} smoke {R['smoke']}")
    print("groups:", ", ".join(f"{k} {v}" for k, v in R["groups"].items()))
    for lab, C in R["checkpoints"].items():
        L = C["logged"]
        print(f"\n## {lab}: step {C['step']} round {C['round']} target {C['target_src']}")
        print(f"logged (steps {L['window_steps']}, {L['n_log_rows']} rows): entropy {f(g(L, 'entropy', 'mean'))}, "
              f"action_sens_rel {f(g(L, 'critic/action_sens_rel', 'mean'))} "
              f"[{f(g(L, 'critic/action_sens_rel', 'min'))}, {f(g(L, 'critic/action_sens_rel', 'max'))}], "
              f"actor/action_std {f(g(L, 'actor/action_std', 'mean'))}, grad/actor {f(g(L, 'grad/actor', 'mean'))}")
        print(f"all states: H analytic {f(C['all_states_entropy_analytic_mean'])}, H mc {f(C['all_states_entropy_mc_mean'])}, "
              f"frac at sigma cap {f(C['all_states_frac_joints_at_sigma_cap'])}, sigma median {f(C['all_states_sigma_median'])}, "
              f"|tanh mu| median {f(C['all_states_abs_tanh_mu_median'])}, sens_rel fp32 "
              f"{f(C.get('all_states_sens_rel_logged_way_fp32'))} tf32 {f(C.get('all_states_sens_rel_logged_way_tf32'))}")
        print("act_proj weight column norms c1:", C["act_proj_weight_col_norms"]["c1"])
        print("\n(a) actor")
        print("| group | n | mu med | |mu| p90 | sigma p10/med/p90 | log_std raw med | at cap | |tanh mu| med/p90/max | "
              "H an | H mc | ||a_det|| med | ||a_ref|| med | cos med | a_det~0 |")
        print("|" + "---|" * 14)
        for gname, S in C["groups"].items():
            print(f"| {gname} | {S['n_states']} | {f(g(S, 'mu', 'median'))} | {f(g(S, 'abs_mu', 'p90'))} | "
                  f"{f(g(S, 'sigma', 'p10'))}/{f(g(S, 'sigma', 'median'))}/{f(g(S, 'sigma', 'p90'))} | "
                  f"{f(g(S, 'log_std_raw', 'median'))} | {f(S['frac_joints_at_sigma_cap'])} | "
                  f"{f(g(S, 'abs_tanh_mu_eq_abs_adet_over_dmax', 'median'))}/{f(g(S, 'abs_tanh_mu_eq_abs_adet_over_dmax', 'p90'))}/"
                  f"{f(g(S, 'abs_tanh_mu_eq_abs_adet_over_dmax', 'max'))} | {f(g(S, 'entropy_analytic_sum', 'mean'))} | "
                  f"{f(g(S, 'entropy_mc', 'mean'))} | {f(g(S, 'norm_adet_rad', 'median'))} | {f(g(S, 'norm_aref_rad', 'median'))} | "
                  f"{f(g(S, 'cos_adet_aref', 'median'))} | {f(S['adet_near_zero_frac'])} |")
        print("\n(b) critic action response (h units)")
        print("| group | Q stored med | spread c med/p90 | spread 16 rand med/p90/max | rel spread16 med | "
              "grad*dmax l2 med (a_ref/0/a_det) | tf32 noise p90 | Q0-Qhalf med | Qhalf-Qnom med | P(Q0>Qhalf) | "
              "P(Qhalf>Qnom) | P(stop>slow>nom) | argmax c | best cand - Q(a_det) med/p90 |")
        print("|" + "---|" * 14)
        for gname, S in C["groups"].items():
            rk = S["ranking"]
            av = S.get("actor_vs_Q", {})
            print(f"| {gname} | {f(g(S, 'Q_stored', 'median'))} | {f(g(S, 'spread_over_c_abs', 'median'))}/"
                  f"{f(g(S, 'spread_over_c_abs', 'p90'))} | {f(g(S, 'spread_16_random_abs', 'median'))}/"
                  f"{f(g(S, 'spread_16_random_abs', 'p90'))}/{f(g(S, 'spread_16_random_abs', 'max'))} | "
                  f"{f(g(S, 'spread_16_random_rel_absQ', 'median'))} | {f(g(S, 'gradQ_times_dmax_l2_at_aref', 'median'))}/"
                  f"{f(g(S, 'gradQ_times_dmax_l2_at_zero', 'median'))}/{f(g(av, 'gradQ_times_dmax_l2_at_adet', 'median'))} | "
                  f"{f(g(S, 'tf32_minus_fp32_abs', 'p90'))} | {f(g(rk, 'Q0_minus_Qhalf', 'median'))} | "
                  f"{f(g(rk, 'Qhalf_minus_Qnom', 'median'))} | {f(rk['frac_Q0_gt_Qhalf'])} | {f(rk['frac_Qhalf_gt_Qnom'])} | "
                  f"{f(rk['frac_order_stop_gt_slow_gt_nom'])} | {rk['argmax_c_hist']} | "
                  f"{f(g(av, 'Q_best_candidate_minus_Q_det', 'median'))}/{f(g(av, 'Q_best_candidate_minus_Q_det', 'p90'))} |")
        print("\n(c) where the action signal dies (twin 1; means over valid tokens; a = saturated unless noted)")
        print("| group | ||body|| | ||inj(0)|| | ||inj(sat)|| | dinj/body aref | dinj/body sat (mean/max) | "
              "hb0 | hb1 | hb2 rel | hq2 rel | dV max sat | dQ sat | argmin is query |")
        print("|" + "---|" * 13)
        for gname, S in C["groups"].items():
            T = S.get("trace", {})
            print(f"| {gname} | {f(g(T, 'c1_body_norm_mean', 'median'))} | {f(g(T, 'c1_inj_norm_zero', 'median'))} | "
                  f"{f(g(T, 'c1_inj_norm_sat', 'median'))} | {f(g(T, 'c1_aref_inj_delta_over_body', 'median'))} | "
                  f"{f(g(T, 'c1_sat_inj_delta_over_body', 'median'))}/{f(g(T, 'c1_sat_inj_delta_over_body_max', 'median'))} | "
                  f"{f(g(T, 'c1_sat_hb0_rel', 'median'))} | {f(g(T, 'c1_sat_hb1_rel', 'median'))} | "
                  f"{f(g(T, 'c1_sat_hb2_rel', 'median'))} | {f(g(T, 'c1_sat_hq2_rel', 'median'))} | "
                  f"{f(g(T, 'c1_sat_dV_max', 'median'))} | {f(g(T, 'c1_sat_dQ', 'median'))} | "
                  f"{f(T.get('c1_softmin_argmin_is_query_frac'))} |")
        print("\n(d) target composition")
        print("| group | y stoch mean | std over 8 samples med | |ys-ydet| med/p90 | |ys-ynext| med | |ys-y0| med | "
              "|qbar stoch-det| med | td y-Q med |")
        print("|" + "---|" * 8)
        for gname, S in C["groups"].items():
            T = S["target"]
            print(f"| {gname} | {f(g(T, 'y_stoch_mean', 'mean'))} | {f(g(T, 'y_stoch_std_over_8_samples', 'median'))} | "
                  f"{f(g(T, 'abs_ystoch_minus_ydet', 'median'))}/{f(g(T, 'abs_ystoch_minus_ydet', 'p90'))} | "
                  f"{f(g(T, 'abs_ystoch_minus_ynext', 'median'))} | {f(g(T, 'abs_ystoch_minus_yzero', 'median'))} | "
                  f"{f(g(T, 'abs_qbar_stoch_minus_det', 'median'))} | {f(g(T, 'td_y_stoch_minus_Q_stored', 'median'))} |")
    M = R.get("motion")
    if M:
        print("\n## motion (numpy, T5 episodes)")
        for k, v in M.items():
            if isinstance(v, dict) and "ticks" in v:
                print(f"- {k}: ticks {v['ticks']}, |cmd|/dmax per joint med {f(g(v, 'abs_cmd_over_dmax_per_joint', 'median'))} "
                      f"p90 {f(g(v, 'abs_cmd_over_dmax_per_joint', 'p90'))}, max-joint ratio med "
                      f"{f(g(v, 'max_joint_ratio_per_tick', 'median'))}, realized/cmd {f(v['realized_over_cmd_summed'])}, "
                      f"max link disp mm med {f(g(v, 'max_link_disp_mm', 'median'))} p90 {f(g(v, 'max_link_disp_mm', 'p90'))}, "
                      f"<0.5 mm {f(v['frac_ticks_link_disp_below_0.5mm'])}, tray speed med {f(g(v, 'tray_speed_mps', 'median'))}")
                if v.get("by_round"):
                    print("    by round:", json.dumps(v["by_round"]))
        print("- column check:", M.get("qpos_raw_vs_dtheta_column_check"))
        pb = M.get("planner_brakes", {})
        print(f"- planner brakes: n {pb.get('n')}, peak decel tipped {pb.get('peak_decel_tipped')}, not tipped "
              f"{pb.get('peak_decel_not_tipped')}")
        sw = M.get("filter_switches_while_moving", {})
        print(f"- filter switches while moving: n {sw.get('n')}, peak decel next 4 {sw.get('peak_decel_next4_mps2')}, "
              f"v_prev {sw.get('v_prev')}, v_after3 {sw.get('v_after3')}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
