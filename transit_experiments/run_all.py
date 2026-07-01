from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from .evaluation import access_times, fixed_point_evaluate
from .heuristics import alternating_local_search, current_practice, greedy_stop_removal
from .instances import make_route_438_instance, make_synthetic_instance, with_parameters
from .milp import solve_milp
from .plotting import (
    plot_equity_tradeoff,
    plot_route_area_efficiency_equity,
    plot_mu_surfaces,
    plot_route_line_map,
    plot_route_dashboard,
    plot_route_pattern,
    plot_route_scenario_surface,
    plot_runtime_scaling,
    plot_sensitivity_heatmaps,
    plot_sensitivity_surface,
    plot_illustrative_redesign_example,
    plot_small_bars,
    plot_synthetic_layout,
    plot_small_pareto,
)
from .tests import run_smoke_tests


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
TABLES = OUT / "tables"


def _write_latex_table(df: pd.DataFrame, path: Path, columns: list[str], caption: str, label: str) -> None:
    tmp = df.loc[:, columns].copy()
    def _clean_float(x: float) -> float:
        return 0.0 if pd.notna(x) and abs(float(x)) < 0.005 else float(x)

    display_names = {
        "strategy": "Method",
        "stop_budget": r"\(\bar L\)",
        "lambda_eq": r"\(\lambda^{eq}\)",
        "objective": "Obj.",
        "transit_share": "Share (\\%)",
        "avg_access_increase_min": "Access inc. (min)",
        "coverage": "Coverage (\\%)",
        "retained_stops": "Stops",
        "avg_frequency": "Freq.",
        "runtime_sec": "Time (s)",
        "scenario": "Scenario",
        "strategy_short": "Strategy",
        "instance_id": "Instance",
        "periods": "Periods",
        "origins": "Origins",
        "destinations": "Dest.",
        "od_pairs": "OD pairs",
        "total_demand": "Demand",
        "n_stops": "Stops",
        "n_instances": "Inst.",
        "milp_objective": "MILP obj.",
        "local_objective": "Local obj.",
        "milp_share": "MILP share (\\%)",
        "local_share": "Local share (\\%)",
        "improvement_pct": "Improve (\\%)",
        "z_eff": "Eff.",
        "z_eq": "Eq.",
        "mu_car": r"\(\mu_{\mathrm{car}}\)",
        "mu_ebike": r"\(\mu_{\mathrm{e-bike}}\)",
        "ebike_level": "Regime",
    }
    for c in tmp.select_dtypes(include=[float]).columns:
        if c in {"transit_share", "coverage", "mip_gap", "local_share", "milp_share"}:
            tmp[c] = tmp[c].map(lambda x: f"{100 * _clean_float(x):.1f}" if pd.notna(x) else "--")
        else:
            tmp[c] = tmp[c].map(lambda x: f"{_clean_float(x):.2f}" if pd.notna(x) else "--")
    tmp = tmp.rename(columns={c: display_names.get(c, c) for c in tmp.columns})
    column_format = "l" + "r" * (len(tmp.columns) - 1)
    if columns and columns[0] != "strategy":
        column_format = "r" * len(tmp.columns)
    if columns[:2] == ["scenario", "strategy"]:
        column_format = "ll" + "r" * (len(tmp.columns) - 2)
    if columns[:2] == ["scenario", "strategy_short"]:
        column_format = "ll" + "r" * (len(tmp.columns) - 2)
    if columns[:2] == ["ebike_level", "strategy_short"]:
        column_format = "ll" + "r" * (len(tmp.columns) - 2)
    latex = tmp.to_latex(index=False, escape=False, caption=caption, label=label, column_format=column_format, position="htbp")
    latex = latex.replace("\\begin{table}[htbp]", "\\begin{table}[htbp]\n\\centering\n\\small")
    path.write_text(latex, encoding="utf-8")


def _write_route_strategy_table(df: pd.DataFrame, path: Path) -> None:
    strategy_order = ["C", "S", "F", "MILP"]
    scenario_order = ["weekday", "weekend", "holiday"]
    scenario_names = {"weekday": "Weekday", "weekend": "Weekend", "holiday": "Holiday"}
    metric_cols = [
        "objective",
        "z_eff",
        "z_eq",
        "transit_share",
        "avg_access_increase_min",
        "retained_stops",
        "avg_frequency",
        "runtime_sec",
    ]
    avg = df.groupby("strategy_short", as_index=False)[metric_cols].mean()
    avg["scenario"] = "average"
    combined = pd.concat(
        [df[["scenario", "strategy_short", *metric_cols]], avg[["scenario", "strategy_short", *metric_cols]]],
        ignore_index=True,
    )
    combined["strategy_short"] = pd.Categorical(combined["strategy_short"], categories=strategy_order, ordered=True)
    body = []
    for scenario in [*scenario_order, "average"]:
        sub = combined[combined["scenario"].astype(str) == scenario].sort_values("strategy_short")
        label = scenario_names.get(scenario, "Average")
        for ridx, (_, row) in enumerate(sub.iterrows()):
            scenario_cell = rf"\multirow{{4}}{{*}}{{{label}}}" if ridx == 0 else ""
            stops = f"{row['retained_stops']:.1f}" if scenario == "average" else f"{int(round(row['retained_stops']))}"
            body.append(
                " & ".join(
                    [
                        scenario_cell,
                        str(row["strategy_short"]),
                        f"{row['objective']:.2f}",
                        f"{row['z_eff']:.2f}",
                        f"{row['z_eq']:.2f}",
                        f"{100.0 * row['transit_share']:.1f}",
                        f"{row['avg_access_increase_min']:.2f}",
                        stops,
                        f"{row['avg_frequency']:.2f}",
                        rf"{{\color{{blue}}{row['runtime_sec']:.2f}}}",
                    ]
                )
                + r" \\"
            )
        body.append(r"\addlinespace[1.5pt]" if scenario != "average" else r"\bottomrule")
    latex = rf"""\begin{{table}}[htbp]
\centering
\small
\caption{{{{\color{{blue}}Strategy comparison for the Beijing Route 438 case.}}}}
\label{{tab:route438_strategy}}
\begin{{threeparttable}}
\begin{{tabular}}{{llrrrrrrrr}}
\toprule
Scenario & Strategy & Obj. & Eff. & Eq. & Share (\%) & Added walk & Stops & Freq. & {{\color{{blue}}Time (s)}} \\
\midrule
{chr(10).join(body)}
\end{{tabular}}
\begin{{tablenotes}}
\footnotesize
\item {{\color{{blue}}Notes: Eff. is the private-mode generalized-cost component, Eq. is the access-equity component, and Added walk is the demand-weighted increase in access-egress walking time in minutes. Frequencies are vehicles per hour. Time is wall-clock computational time for optimization plus ex-post fixed-point evaluation; for C, it is evaluation time only. Coverage is omitted because all reported designs satisfy the coverage requirement and obtain 100\% covered origins.}}
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
"""
    path.write_text(latex, encoding="utf-8")


def _format_result_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    tmp = df.loc[:, columns].copy()
    integer_like = {"MILP_stops", "retained_stops", "n_stops", "origins", "destinations", "od_pairs", "periods"}
    for c in tmp.select_dtypes(include=[float]).columns:
        if c in integer_like:
            tmp[c] = tmp[c].map(lambda x: f"{int(round(x))}")
        elif c.endswith("share") or c == "transit_share":
            tmp[c] = tmp[c].map(lambda x: f"{100 * x:.1f}")
        elif c in {"mip_gap", "MILP_gap"}:
            tmp[c] = tmp[c].map(lambda x: "--" if pd.isna(x) else f"{100 * x:.2f}")
        else:
            tmp[c] = tmp[c].map(lambda x: f"{x:.2f}")
    names = {
        "instance_id": "Instance",
        "n_stops": "Stops",
        "origins": "Origins",
        "destinations": "Dest.",
        "od_pairs": "OD",
        "periods": "Periods",
        "total_demand": "Demand",
        "C_objective": "C obj.",
        "G_objective": "G obj.",
        "LS_objective": "LS obj.",
        "MILP_objective": "MILP obj.",
        "MILP_share": "MILP share (\\%)",
        "MILP_stops": "MILP stops",
        "MILP_freq": "MILP freq.",
        "MILP_gap": "Gap (\\%)",
    }
    return tmp.rename(columns={c: names.get(c, c) for c in tmp.columns})


def _write_compact_table(df: pd.DataFrame, path: Path, columns: list[str], caption: str, label: str) -> None:
    tmp = _format_result_columns(df, columns)
    column_format = "l" + "r" * (len(tmp.columns) - 1)
    latex = tmp.to_latex(index=False, escape=False, caption=caption, label=label, column_format=column_format, position="htbp")
    latex = latex.replace(
        "\\begin{table}[htbp]",
        "\\begin{table}[htbp]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{3.5pt}",
    )
    path.write_text(latex, encoding="utf-8")


def _write_longtable(df: pd.DataFrame, path: Path, columns: list[str], caption: str, label: str) -> None:
    tmp = _format_result_columns(df, columns)
    column_format = "l" + "r" * (len(tmp.columns) - 1)
    latex = tmp.to_latex(index=False, escape=False, longtable=True, caption=caption, label=label, column_format=column_format)
    latex = latex.replace("\\begin{longtable}", "\\begingroup\n\\scriptsize\n\\setlength{\\tabcolsep}{3.5pt}\n\\begin{longtable}")
    latex = latex.replace("\\end{longtable}", "\\end{longtable}\n\\normalsize\n\\endgroup")
    path.write_text(latex, encoding="utf-8")


def _write_illustrative_method_detail_table(df: pd.DataFrame, path: Path) -> None:
    methods = [
        ("All-stops baseline", "C"),
        ("Greedy removal", "G"),
        ("Local search", "LS"),
        ("MILP approximation", "MILP"),
    ]
    metrics = [
        ("objective", "Obj."),
        ("transit_share", "Share"),
        ("retained_stops", "Stops"),
        ("runtime_sec", "Time"),
    ]
    rows = []
    ordered = df.sort_values(["instance_id", "strategy"])
    for instance_id, sub in ordered.groupby("instance_id", sort=True):
        first = sub.iloc[0]
        row = [
            instance_id,
            f"{int(first['n_stops'])}",
            f"{int(first['od_pairs'])}",
            f"{int(first['periods'])}",
        ]
        for method, _ in methods:
            rec = sub[sub["strategy"] == method].iloc[0]
            for metric, _ in metrics:
                value = float(rec[metric])
                if metric == "transit_share":
                    row.append(f"{100.0 * value:.1f}")
                elif metric == "retained_stops":
                    row.append(f"{int(round(value))}")
                elif metric == "runtime_sec":
                    row.append(f"{value:.2f}")
                else:
                    row.append(f"{value:.2f}")
        rows.append(row)

    body = "\n".join(" & ".join(row) + r" \\" for row in rows)
    method_header = " & ".join([rf"\multicolumn{{4}}{{c}}{{{short}}}" for _, short in methods])
    cmidrules = " ".join(
        rf"\cmidrule(lr){{{5 + 4 * idx}-{8 + 4 * idx}}}" for idx in range(len(methods))
    )
    metric_header = " & ".join([label for _method in methods for _metric, label in metrics])
    latex = rf"""\begin{{landscape}}
\begin{{table}}[p]
\centering
\begingroup
\tiny
\setlength{{\tabcolsep}}{{0.75pt}}
\renewcommand{{\arraystretch}}{{0.64}}
\setlength{{\abovecaptionskip}}{{1pt}}
\setlength{{\belowcaptionskip}}{{1pt}}
\caption{{Per-instance illustrative-experiment results.}}
\label{{tab:illustrative_instance_method_details}}
\resizebox{{0.90\linewidth}}{{!}}{{%
\begin{{tabular}}{{lrrr*{{4}}{{rrrr}}}}
\toprule
\multirow{{2}}{{*}}{{Instance}} & \multirow{{2}}{{*}}{{Stops}} & \multirow{{2}}{{*}}{{OD}} & \multirow{{2}}{{*}}{{Periods}} & {method_header} \\
{cmidrules}
 &  &  &  & {metric_header} \\
\midrule
{body}
\bottomrule
\end{{tabular}}%
}}
\endgroup
\end{{table}}
\end{{landscape}}
"""
    path.write_text(latex, encoding="utf-8")


def _write_parameter_appendix(route_inst) -> None:
    meta_path = ROOT / "data" / "route438_afc_metadata.json"
    afc_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    rows = [
        ("Choice alternatives", "Transit, private car, e-bike, bicycle", "The lower-level MNL has four alternatives: one transit alternative and three private-mode alternatives."),
        ("Private-mode speed", r"\(v_{\mathrm{car}},v_{\mathrm{e-bike}},v_{\mathrm{bike}}=(0.58,0.32,0.22)\) km/min", "Used to construct private car, e-bike, and bicycle baseline travel times, respectively."),
        ("Private non-time cost", r"\(c^k_{ij}=6+0.18d,\ 1.8+0.08d,\ 0.8+0.04d\)", r"Generalized private-mode constants for car, e-bike, and bicycle; \(d\) is OD Euclidean distance in km. The e-bike cost is intentionally low."),
        ("Transit fare", rf"\(C^{{\mathrm{{fare}}}}_{{\mathrm{{tr}}}}={route_inst.fare_tr:.1f}\)", "Generalized non-time transit cost."),
        ("Transit time coefficient", rf"\(\theta^{{\mathrm{{tr}}}}={route_inst.theta_tr:.2f}\)", "Converts total transit travel time into generalized cost; access burden is controlled separately by the equity terms."),
        ("Private time coefficients", rf"\(\theta^k=({', '.join(f'{v:.2f}' for v in route_inst.theta_private)})\)", "Car, e-bike, and bicycle time coefficients."),
        ("Alternative constants", rf"\(\beta^{{\mathrm{{tr}}}}={route_inst.beta_tr:.1f}\); \(\beta^k=({', '.join(f'{v:.2f}' for v in route_inst.beta_private)})\)", r"Mode-specific constants in the utility functions; the transit constant is calibrated to match a 25--30\% current-practice transit share."),
        ("Logit scale", rf"\(\lambda_t={route_inst.logit_lambda[0]:.3f}\ \forall t\)", "Period-specific MNL sensitivity parameter; the same value is used for all aggregation periods."),
        ("MFD equivalent accumulation", r"\(\chi_k\) normalized and absorbed into \(\mu_k\)", "Modal road-space conversion factors are represented through the reduced-form congestion coefficients used in the experiments."),
        ("Congestion coefficients", rf"\(\mu_r={route_inst.mu_r:.3f}\); \(\mu_k=({', '.join(f'{v:.4f}' for v in route_inst.mu_private)})\)", "Transit, car, e-bike, and bicycle congestion-contribution parameters before sensitivity changes."),
        ("Accessibility policy", rf"\(\bar L={route_inst.stop_budget:.2f},\rho_{{min}}={route_inst.rho_min:.2f},\Delta^{{acc}}_{{max}}={route_inst.delta_acc_max_min:.1f}\)", "Stop-removal allowance, minimum covered-origin share, and zone access-increase limit."),
        ("Route 438 periods", f"{route_inst.T} periods: " + ", ".join(route_inst.period_names), "Each period is an operational aggregation interval constructed from AFC boarding times, not a 10-minute slice."),
        ("Route 438 AFC sample", f"{afc_meta.get('records_line_438', 'NA')} line records; {afc_meta.get('paired_trips_studied_direction', 'NA')} paired studied-direction trips", "Route 438 card and QR-code transactions on 2021-12-17 and 2021-12-18 are filtered and paired by card, vehicle, direction, and transaction time."),
        ("Route 438 OD data", f"{route_inst.I} origins, {route_inst.J} destinations, {route_inst.I * route_inst.J} potential OD pairs", f"AFC stop-level trips are mapped to corridor OD centroids; {afc_meta.get('unique_observed_stop_od_pairs', 'NA')} observed stop OD pairs and {afc_meta.get('positive_observed_od_period_cells', 'NA')} positive OD-period cells seed the demand matrix."),
        ("Transit-share expansion", rf"\(\eta^{{\mathrm{{tr}}}}={afc_meta.get('transit_share_assumption', 0.28)}\)", r"Observed Route 438 transit trips are expanded to total multimodal demand using the 25--30\% transit-share assumption for the corridor; private-mode demand is the residual represented through the lower-level mode-choice model."),
        ("Route 438 equity zones", f"{len(set(route_inst.origin_zone.tolist()))} corridor zones", "AFC-derived origins are assigned to consecutive corridor analysis zones according to route-order coordinate; these zones are not administrative or housing-community polygons."),
    ]
    body = "\n".join(f"{group} & {value} & {explanation} \\\\" for group, value, explanation in rows)
    latex = rf"""\scriptsize
\begin{{longtable}}{{p{{0.18\textwidth}}p{{0.30\textwidth}}p{{0.42\textwidth}}}}
\caption{{Parameter settings used in the numerical experiments.\label{{tab:appendix_parameters}}}} \\
\toprule
Parameter group & Value & Explanation \\
\midrule
\endfirsthead
\caption[]{{Parameter settings used in the numerical experiments.}} \\
\toprule
Parameter group & Value & Explanation \\
\midrule
\endhead
\midrule
\multicolumn{{3}}{{r}}{{Continued on next page}} \\
\midrule
\endfoot
\bottomrule
\endlastfoot
{body}
\end{{longtable}}
\normalsize
"""
    (TABLES / "appendix_parameter_settings.tex").write_text(latex, encoding="utf-8")


STRATEGY_SHORT = {
    "Current practice": "C",
    "Stop-only MILP": "S",
    "Frequency-only MILP": "F",
    "Joint MILP": "MILP",
}


def _illustrative_figure_score(inst, baseline_res, milp_res) -> float | None:
    """Choose a large illustrative example with visible OD density and clear mode shift."""
    _, _, o0, _ = access_times(inst, np.ones(inst.S))
    _, _, o1, _ = access_times(inst, milp_res.y)
    changed_origins = int(np.sum(o0 != o1))
    od_pairs = inst.I * inst.J
    if od_pairs < 250 or changed_origins < 2:
        return None
    skip_frac = 1.0 - milp_res.retained_stops / max(1, inst.S)
    share_gain = milp_res.transit_share - baseline_res.transit_share
    private_drop = (baseline_res.private_flow - milp_res.private_flow) / max(1e-9, baseline_res.private_flow)
    if skip_frac <= 0.05 or share_gain <= 0.02 or private_drop <= 0.02:
        return None
    changed_share = changed_origins / max(1, inst.I)
    access_penalty = max(0.0, milp_res.avg_access_increase_min)
    nearest_origin = np.sqrt(
        (inst.origin_x[:, None] - inst.stop_x[None, :]) ** 2
        + (inst.origin_y[:, None] - inst.stop_y[None, :]) ** 2
    ).min(axis=1)
    nearest_dest = np.sqrt(
        (inst.dest_x[:, None] - inst.stop_x[None, :]) ** 2
        + (inst.dest_y[:, None] - inst.stop_y[None, :]) ** 2
    ).min(axis=1)
    od_stop_distance = np.r_[nearest_origin, nearest_dest]
    return (
        0.045 * od_pairs
        + 100.0 * share_gain
        + 35.0 * private_drop
        + 12.0 * skip_frac
        + 1.2 * changed_origins
        - 4.0 * abs(changed_share - 0.10)
        - 1.5 * access_penalty
        - 5.0 * float(np.percentile(od_stop_distance, 90))
        - 2.5 * float(od_stop_distance.max())
    )


def run_illustrative() -> pd.DataFrame:
    configs = []
    scale_stops = [12, 16, 20, 24, 28]
    for idx in range(50):
        n_stops = scale_stops[idx % len(scale_stops)]
        n_o = max(6, n_stops // 2 + 1 + (idx % 3))
        n_d = max(7, n_stops // 2 + 3 + (idx % 4))
        n_periods = [4, 5, 6][idx % 3]
        configs.append((n_stops, n_o, n_d, n_periods, 400 + idx))
    rows = []
    settings = []
    representative = None
    representative_pair = None
    representative_candidates = []
    for n_stops, n_o, n_d, n_periods, seed in configs:
        inst = make_synthetic_instance(n_stops, n_o, n_d, n_periods, seed)
        if representative is None and n_stops == 20:
            representative = inst
        print(f"[illustrative] {inst.name}: {n_o*n_d} OD pairs, {n_periods} periods")
        settings.append(
            {
                "instance": inst.name,
                "instance_id": f"I{len(settings) + 1:02d}",
                "n_stops": n_stops,
                "origins": n_o,
                "destinations": n_d,
                "od_pairs": n_o * n_d,
                "periods": n_periods,
                "total_demand": float(inst.demand.sum()),
            }
        )
        baseline_res = current_practice(inst, strategy="All-stops baseline")
        greedy_res = greedy_stop_removal(inst)
        local_res = alternating_local_search(inst, max_iter=1, strategy="Local search")
        milp_res = solve_milp(inst, strategy="MILP approximation", time_limit=6)
        rows.extend([baseline_res.as_row(), greedy_res.as_row(), local_res.as_row(), milp_res.as_row()])
        score = _illustrative_figure_score(inst, baseline_res, milp_res)
        if score is not None:
            representative_candidates.append((score, inst, baseline_res, milp_res))
        if representative_pair is None and n_stops == 20 and milp_res.retained_stops < inst.S:
            representative_pair = (inst, baseline_res, milp_res)
    df = pd.DataFrame(rows)
    df["n_stops"] = df["instance"].str.extract(r"synthetic_(\d+)_stops").astype(int)
    settings_df = pd.DataFrame(settings)
    df = df.merge(settings_df[["instance", "instance_id", "origins", "destinations", "od_pairs", "periods", "total_demand"]], on="instance", how="left")
    df.to_csv(OUT / "illustrative_results.csv", index=False, encoding="utf-8-sig")
    settings_df.to_csv(OUT / "illustrative_instance_settings.csv", index=False, encoding="utf-8-sig")
    df.to_csv(OUT / "small_scale_results.csv", index=False, encoding="utf-8-sig")
    summary = (
        df.groupby("strategy", as_index=False)
        .agg(
            objective=("objective", "mean"),
            z_eff=("z_eff", "mean"),
            z_eq=("z_eq", "mean"),
            transit_share=("transit_share", "mean"),
            avg_access_increase_min=("avg_access_increase_min", "mean"),
            retained_stops=("retained_stops", "mean"),
            avg_frequency=("avg_frequency", "mean"),
            runtime_sec=("runtime_sec", "mean"),
        )
        .sort_values("objective")
    )
    _write_latex_table(
        summary,
        TABLES / "small_scale_summary.tex",
        ["strategy", "objective", "z_eff", "z_eq", "transit_share", "avg_access_increase_min", "retained_stops", "avg_frequency", "runtime_sec"],
        "Average performance in the illustrative synthetic experiments.",
        "tab:small_scale_summary",
    )
    _write_latex_table(
        summary,
        TABLES / "illustrative_summary.tex",
        ["strategy", "objective", "z_eff", "z_eq", "transit_share", "avg_access_increase_min", "retained_stops", "avg_frequency", "runtime_sec"],
        "Average performance in the illustrative experiments.",
        "tab:illustrative_summary",
    )
    pairs = []
    for instance, sub in df.groupby("instance"):
        local = sub[sub["strategy"] == "Local search"].iloc[0]
        milp = sub[sub["strategy"] == "MILP approximation"].iloc[0]
        pairs.append(
            {
                "instance": instance,
                "n_stops": int(milp["n_stops"]),
                "local_objective": float(local["objective"]),
                "milp_objective": float(milp["objective"]),
                "local_share": float(local["transit_share"]),
                "milp_share": float(milp["transit_share"]),
                "improvement_pct": 100.0 * (float(local["objective"]) - float(milp["objective"])) / max(1e-9, float(local["objective"])),
            }
        )
    pair_df = pd.DataFrame(pairs)
    pair_df = pair_df.merge(settings_df[["instance", "instance_id", "origins", "destinations", "od_pairs", "periods", "total_demand"]], on="instance", how="left")
    pair_df.to_csv(OUT / "small_scale_milp_vs_local.csv", index=False, encoding="utf-8-sig")
    pair_df.to_csv(OUT / "illustrative_milp_vs_local.csv", index=False, encoding="utf-8-sig")
    scale_summary = (
        pair_df.groupby("n_stops", as_index=False)
        .agg(
            n_instances=("instance", "count"),
            local_objective=("local_objective", "mean"),
            milp_objective=("milp_objective", "mean"),
            local_share=("local_share", "mean"),
            milp_share=("milp_share", "mean"),
            improvement_pct=("improvement_pct", "mean"),
        )
        .sort_values("n_stops")
    )
    _write_latex_table(
        scale_summary,
        TABLES / "small_scale_milp_vs_local.tex",
        ["n_stops", "n_instances", "local_objective", "milp_objective", "local_share", "milp_share", "improvement_pct"],
        "MILP and local-search comparison by illustrative instance scale.",
        "tab:small_scale_milp_vs_local",
    )
    _write_latex_table(
        scale_summary,
        TABLES / "illustrative_milp_vs_local.tex",
        ["n_stops", "n_instances", "local_objective", "milp_objective", "local_share", "milp_share", "improvement_pct"],
        "MILP and local-search comparison by illustrative instance scale.",
        "tab:illustrative_milp_vs_local",
    )
    _write_longtable(
        settings_df,
        TABLES / "appendix_illustrative_instance_settings.tex",
        ["instance_id", "n_stops", "origins", "destinations", "od_pairs", "periods", "total_demand"],
        "Detailed settings of the illustrative synthetic instances.",
        "tab:appendix_illustrative_settings",
    )
    result_rows = []
    pivot = df.pivot_table(index=["instance", "instance_id"], columns="strategy", values=["objective", "transit_share", "retained_stops", "avg_frequency", "mip_gap"], aggfunc="first")
    for (instance, instance_id), row in pivot.iterrows():
        result_rows.append(
            {
                "instance": instance,
                "instance_id": instance_id,
                "C_objective": float(row[("objective", "All-stops baseline")]),
                "G_objective": float(row[("objective", "Greedy removal")]),
                "LS_objective": float(row[("objective", "Local search")]),
                "MILP_objective": float(row[("objective", "MILP approximation")]),
                "MILP_share": float(row[("transit_share", "MILP approximation")]),
                "MILP_stops": float(row[("retained_stops", "MILP approximation")]),
                "MILP_freq": float(row[("avg_frequency", "MILP approximation")]),
                "MILP_gap": float(row[("mip_gap", "MILP approximation")]),
            }
        )
    detailed_df = pd.DataFrame(result_rows).sort_values("instance_id")
    detailed_main_df = detailed_df.merge(
        settings_df[["instance", "instance_id", "n_stops", "od_pairs", "periods"]],
        on=["instance", "instance_id"],
        how="left",
    )
    detailed_df.to_csv(OUT / "illustrative_instance_results.csv", index=False, encoding="utf-8-sig")
    _write_compact_table(
        detailed_main_df,
        TABLES / "illustrative_instance_results_main.tex",
        [
            "instance_id",
            "n_stops",
            "od_pairs",
            "periods",
            "C_objective",
            "G_objective",
            "LS_objective",
            "MILP_objective",
            "MILP_share",
            "MILP_stops",
        ],
        "Per-instance performance in the illustrative experiments.",
        "tab:illustrative_instance_results_main",
    )
    _write_illustrative_method_detail_table(
        df,
        TABLES / "illustrative_instance_method_details.tex",
    )
    _write_longtable(
        detailed_df,
        TABLES / "appendix_illustrative_instance_results.tex",
        ["instance_id", "C_objective", "G_objective", "LS_objective", "MILP_objective", "MILP_share", "MILP_stops", "MILP_freq", "MILP_gap"],
        "Detailed results of the illustrative synthetic instances.",
        "tab:appendix_illustrative_results",
    )
    if representative_candidates:
        _, inst, baseline_res, milp_res = max(representative_candidates, key=lambda item: item[0])
        representative_pair = (inst, baseline_res, milp_res)
        representative = inst
    if representative is not None:
        plot_synthetic_layout(representative, FIG)
    if representative_pair is not None:
        plot_illustrative_redesign_example(*representative_pair, FIG)
    return df


def _route_strategy_rows(inst, scenario: str, time_limit: float = 8.0, response_anchors: tuple[float, ...] | None = (0.0, -6.0, -12.0)) -> list[dict]:
    all_y = np.ones(inst.S)
    rows = []
    for res in [
        current_practice(inst, strategy="Current practice"),
        solve_milp(inst, strategy="Stop-only MILP", fixed_freq=inst.baseline_freq, time_limit=time_limit, response_anchors=response_anchors),
        solve_milp(inst, strategy="Frequency-only MILP", fixed_y=all_y, time_limit=max(2.0, 0.75 * time_limit), response_anchors=response_anchors),
        solve_milp(inst, strategy="Joint MILP", time_limit=time_limit, response_anchors=response_anchors),
    ]:
        row = res.as_row()
        row["scenario"] = scenario
        row["strategy_short"] = STRATEGY_SHORT[row["strategy"]]
        rows.append(row)
    return rows


def run_route_438() -> tuple[pd.DataFrame, object]:
    rows = []
    route_inst = None
    for scenario, scale in [("weekday", 1.0), ("weekend", 0.72), ("holiday", 1.18)]:
        inst = make_route_438_instance(demand_scale=scale, scenario=scenario)
        if route_inst is None:
            route_inst = inst
        print(f"[route] {inst.name}: {inst.S} stops, {inst.I*inst.J} OD pairs, {inst.T} periods")
        rows.extend(_route_strategy_rows(inst, scenario))
    df = pd.DataFrame(rows)
    df["scenario"] = pd.Categorical(df["scenario"], categories=["weekday", "weekend", "holiday"], ordered=True)
    df["strategy"] = pd.Categorical(
        df["strategy"],
        categories=["Current practice", "Stop-only MILP", "Frequency-only MILP", "Joint MILP"],
        ordered=True,
    )
    df["strategy_short"] = pd.Categorical(df["strategy_short"], categories=["C", "S", "F", "MILP"], ordered=True)
    df = df.sort_values(["scenario", "strategy"])
    df.to_csv(OUT / "route438_strategy_results.csv", index=False, encoding="utf-8-sig")
    _write_route_strategy_table(df, TABLES / "route438_strategy_table.tex")
    return df, route_inst


def _route_strategy_rows_no_scenario(inst, time_limit: float = 3.0, response_anchors: tuple[float, ...] | None = (0.0,)) -> list[dict]:
    all_y = np.ones(inst.S)
    rows = []
    strategies = [
        current_practice(inst, strategy="Current practice"),
        solve_milp(inst, strategy="Stop-only MILP", fixed_freq=inst.baseline_freq, time_limit=time_limit, response_anchors=response_anchors),
        solve_milp(inst, strategy="Frequency-only MILP", fixed_y=all_y, time_limit=max(2.0, 0.75 * time_limit), response_anchors=response_anchors),
        solve_milp(inst, strategy="Joint MILP", time_limit=time_limit, response_anchors=response_anchors),
    ]
    for res in strategies:
        row = res.as_row()
        row["strategy_short"] = STRATEGY_SHORT[row["strategy"]]
        rows.append(row)
    return rows


def run_mu_sensitivity() -> pd.DataFrame:
    base = make_route_438_instance(scenario="weekday")
    rows = []
    mu_car_grid = np.linspace(0.0035, 0.0140, 10)
    mu_ebike_grid = np.linspace(0.0015, 0.0200, 10)
    bike_mu = float(base.mu_private[2])
    for mu_car in mu_car_grid:
        for mu_ebike in mu_ebike_grid:
            inst = with_parameters(
                base,
                name=f"route438_mu_c{mu_car:.4f}_e{mu_ebike:.4f}",
                mu_private=np.array([mu_car, mu_ebike, bike_mu]),
            )
            print(f"[mu] car={mu_car:.4f}, ebike={mu_ebike:.4f}")
            for row in _route_strategy_rows_no_scenario(inst, time_limit=2.2, response_anchors=(0.0,)):
                row["mu_car"] = mu_car
                row["mu_ebike"] = mu_ebike
                rows.append(row)
    df = pd.DataFrame(rows)
    df["strategy_short"] = pd.Categorical(df["strategy_short"], categories=["C", "S", "F", "MILP"], ordered=True)
    df = df.sort_values(["mu_car", "mu_ebike", "strategy_short"])
    df.to_csv(OUT / "route438_mu_sensitivity_results.csv", index=False, encoding="utf-8-sig")
    high_low = df.copy()
    median_ebike = float(np.median(mu_ebike_grid))
    high_low["ebike_level"] = np.where(high_low["mu_ebike"] > median_ebike, "High e-bike congestion", "Low e-bike congestion")
    summary = (
        high_low.groupby(["ebike_level", "strategy_short"], as_index=False)
        .agg(
            objective=("objective", "mean"),
            z_eff=("z_eff", "mean"),
            z_eq=("z_eq", "mean"),
            transit_share=("transit_share", "mean"),
            avg_access_increase_min=("avg_access_increase_min", "mean"),
        )
        .sort_values(["ebike_level", "strategy_short"])
    )
    _write_latex_table(
        summary,
        TABLES / "route438_mu_sensitivity_summary.tex",
        ["ebike_level", "strategy_short", "objective", "z_eff", "z_eq", "transit_share", "avg_access_increase_min"],
        "Average Route 438 performance under low and high e-bike congestion-contribution regimes.",
        "tab:route438_mu_sensitivity_summary",
    )
    return df


def run_equity_tradeoff() -> pd.DataFrame:
    base = make_route_438_instance(scenario="weekday")
    rows = []
    weights = [0.00, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.40, 1.80, 2.40, 3.00]
    for lambda_eq in weights:
        inst = with_parameters(base, name=f"route438_lambda_{lambda_eq:.2f}", lambda_eq=lambda_eq)
        print(f"[equity] lambda={lambda_eq:.2f}")
        for row in _route_strategy_rows_no_scenario(inst, time_limit=3.5, response_anchors=(0.0, -8.0)):
            row["lambda_eq"] = lambda_eq
            rows.append(row)
    df = pd.DataFrame(rows)
    df["strategy_short"] = pd.Categorical(df["strategy_short"], categories=["C", "S", "F", "MILP"], ordered=True)
    df = df.sort_values(["lambda_eq", "strategy_short"])
    df.to_csv(OUT / "route438_equity_tradeoff_results.csv", index=False, encoding="utf-8-sig")
    milp = df[df["strategy_short"] == "MILP"].copy()
    _write_latex_table(
        milp,
        TABLES / "route438_equity_weight_table.tex",
        ["lambda_eq", "objective", "z_eff", "z_eq", "transit_share", "avg_access_increase_min", "retained_stops", "avg_frequency"],
        "Joint MILP performance under alternative efficiency-equity weights.",
        "tab:route438_equity_weight",
    )
    return df


def run_sensitivity() -> pd.DataFrame:
    base = make_route_438_instance()
    rows = []
    for stop_budget in [0.18, 0.28, 0.38]:
        for lambda_eq in [0.15, 0.40, 0.80]:
            inst = with_parameters(base, stop_budget=stop_budget, lambda_eq=lambda_eq, name=f"route438_L{stop_budget:.2f}_E{lambda_eq:.2f}")
            print(f"[sens] L={stop_budget:.2f}, lambda={lambda_eq:.2f}")
            res = solve_milp(inst, strategy="Joint MILP", time_limit=14)
            row = res.as_row()
            row["stop_budget"] = stop_budget
            row["lambda_eq"] = lambda_eq
            row["rho_min"] = inst.rho_min
            row["demand_scale"] = 1.0
            rows.append(row)
    for rho_min in [0.78, 0.84, 0.90]:
        inst = with_parameters(base, rho_min=rho_min, name=f"route438_rho{rho_min:.2f}")
        res = solve_milp(inst, strategy="Joint MILP", time_limit=12)
        row = res.as_row()
        row["stop_budget"] = inst.stop_budget
        row["lambda_eq"] = inst.lambda_eq
        row["rho_min"] = rho_min
        row["demand_scale"] = 1.0
        rows.append(row)
    for scale in [0.80, 1.00, 1.20]:
        inst = make_route_438_instance(demand_scale=scale)
        inst = with_parameters(inst, name=f"route438_demand{scale:.2f}")
        res = solve_milp(inst, strategy="Joint MILP", time_limit=12)
        row = res.as_row()
        row["stop_budget"] = inst.stop_budget
        row["lambda_eq"] = inst.lambda_eq
        row["rho_min"] = inst.rho_min
        row["demand_scale"] = scale
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "route438_sensitivity_results.csv", index=False, encoding="utf-8-sig")
    heat = df[(df["rho_min"] == base.rho_min) & (df["demand_scale"] == 1.0)]
    _write_latex_table(
        heat.sort_values(["stop_budget", "lambda_eq"]),
        TABLES / "route438_sensitivity_table.tex",
        ["stop_budget", "lambda_eq", "objective", "transit_share", "avg_access_increase_min", "retained_stops", "avg_frequency"],
        "Sensitivity of the joint optimization strategy for Route 438.",
        "tab:route438_sensitivity",
    )
    return df


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    print("[tests] running smoke tests")
    run_smoke_tests()
    small_df = run_illustrative()
    route_df, route_inst = run_route_438()
    mu_df = run_mu_sensitivity()
    trade_df = run_equity_tradeoff()
    _write_parameter_appendix(route_inst)
    print("[figures] rendering")
    plot_small_bars(small_df, FIG)
    plot_small_pareto(small_df, FIG)
    plot_runtime_scaling(small_df, FIG)
    plot_route_line_map(route_inst, FIG)
    holiday_route_inst = make_route_438_instance(scenario="holiday")
    plot_route_area_efficiency_equity(route_df, holiday_route_inst, FIG, scenario="holiday")
    plot_route_pattern(route_df, route_inst.stop_x, FIG)
    plot_route_dashboard(route_df, FIG)
    plot_route_scenario_surface(route_df, FIG)
    plot_mu_surfaces(mu_df, FIG)
    plot_equity_tradeoff(trade_df, FIG)
    print(f"[done] outputs written to {OUT} in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
