from __future__ import annotations

import math
import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from .evaluation import baseline_access, fixed_point_evaluate
from .instances import Instance, pairwise_distance


def _log_breakpoints(demand: float, n: int = 7) -> tuple[list[float], list[float]]:
    eps = max(1e-3, 1e-4 * demand)
    hi = max(demand, eps * 10)
    pts = np.geomspace(eps, hi, n)
    pts[-1] = hi
    return pts.tolist(), np.log(pts).tolist()


def _logit_response_breakpoints(p0: float, scale: float, n: int = 17, span: float = 40.0) -> tuple[list[float], list[float]]:
    """PWL points for transit probability as transit generalized cost changes."""
    p0 = float(np.clip(p0, 1e-4, 1.0 - 1e-4))
    logit0 = math.log(p0 / (1.0 - p0))
    pts = np.linspace(-span, span, n)
    vals = 1.0 / (1.0 + np.exp(-(logit0 - scale * pts)))
    return pts.tolist(), vals.tolist()


def _logit_response_at_delta(p0: float, scale: float, delta: float) -> float:
    p0 = float(np.clip(p0, 1e-4, 1.0 - 1e-4))
    logit0 = math.log(p0 / (1.0 - p0))
    return float(1.0 / (1.0 + math.exp(-(logit0 - scale * delta))))


def solve_kkt_milp(
    instance: Instance,
    strategy: str = "MILP approximation",
    fixed_y: np.ndarray | None = None,
    fixed_freq: np.ndarray | None = None,
    time_limit: float = 30.0,
    output_flag: int = 0,
) -> object:
    t0 = time.perf_counter()
    I, J, S, T, K = instance.I, instance.J, instance.S, instance.T, instance.K
    D = instance.demand
    A0, B0 = baseline_access(instance)
    ostop = pairwise_distance(instance.origin_x, instance.origin_y, instance.stop_x, instance.stop_y) / instance.v_walk_km_min
    dstop = pairwise_distance(instance.dest_x, instance.dest_y, instance.stop_x, instance.stop_y) / instance.v_walk_km_min
    total_demand = float(D.sum())
    m = gp.Model(f"{instance.name}_{strategy}")
    m.Params.OutputFlag = output_flag
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = 0.015
    y = m.addVars(S, vtype=GRB.BINARY, name="y")
    x = m.addVars(I, S, vtype=GRB.BINARY, name="x")
    z = m.addVars(J, S, vtype=GRB.BINARY, name="z")
    u = {}
    for t in range(T):
        vals = instance.freq_values[t]
        u[t] = m.addVars(len(vals), vtype=GRB.BINARY, name=f"u_{t}")
    qtr = m.addVars(I, J, T, lb=0.0, name="qtr")
    qk = m.addVars(I, J, T, K, lb=0.0, name="qk")
    ltr = m.addVars(I, J, T, lb=-30.0, ub=20.0, name="ltr")
    lk = m.addVars(I, J, T, K, lb=-30.0, ub=20.0, name="lk")
    zeta = m.addVars(I, J, lb=0.0, name="zeta")
    rpos = m.addVars(I, J, T, K, lb=0.0, name="rpos")
    rneg = m.addVars(I, J, T, K, lb=0.0, name="rneg")
    cycle = m.addVars(T, lb=0.0, name="cycle")
    W = {}
    for t in range(T):
        W[t] = m.addVars(len(instance.freq_values[t]), lb=0.0, name=f"W_{t}")

    if fixed_y is not None:
        for s, val in enumerate(fixed_y):
            y[s].LB = y[s].UB = int(round(float(val)))
    if fixed_freq is not None:
        for t in range(T):
            vals = instance.freq_values[t]
            chosen = min(range(len(vals)), key=lambda p: abs(vals[p] - fixed_freq[t]))
            for p in range(len(vals)):
                u[t][p].LB = u[t][p].UB = 1 if p == chosen else 0

    m.addConstr(y[0] == 1, name="keep_first")
    m.addConstr(y[S - 1] == 1, name="keep_last")
    m.addConstr(gp.quicksum(y[s] for s in range(S)) >= math.ceil((1 - instance.stop_budget) * S - 1e-9), name="budget")
    for s in range(S - 1):
        m.addConstr(y[s] + y[s + 1] >= 1, name=f"no_consecutive_skip_{s}")
    for i in range(I):
        m.addConstr(gp.quicksum(x[i, s] for s in range(S)) == 1, name=f"assign_o_{i}")
        for s in range(S):
            m.addConstr(x[i, s] <= y[s], name=f"link_o_{i}_{s}")
    for j in range(J):
        m.addConstr(gp.quicksum(z[j, s] for s in range(S)) == 1, name=f"assign_d_{j}")
        for s in range(S):
            m.addConstr(z[j, s] <= y[s], name=f"link_d_{j}_{s}")

    covered_terms = []
    for i in range(I):
        close = [s for s in range(S) if ostop[i, s] * instance.v_walk_km_min <= instance.coverage_radius_km]
        if close:
            covered_terms.append(instance.origin_weights[i] * gp.quicksum(x[i, s] for s in close))
    m.addConstr(gp.quicksum(covered_terms) >= instance.rho_min * float(instance.origin_weights.sum()), name="coverage")
    for zone in sorted(set(instance.origin_zone.tolist())):
        idx = [i for i in range(I) if instance.origin_zone[i] == zone]
        wsum = float(instance.origin_weights[idx].sum())
        lhs = gp.quicksum(
            instance.origin_weights[i] * gp.quicksum(ostop[i, s] * x[i, s] for s in range(S)) for i in idx
        )
        base = float(np.dot(instance.origin_weights[idx], A0[idx]) / wsum)
        m.addConstr(lhs / wsum - base <= instance.delta_acc_max_min, name=f"equity_zone_{zone}")

    for t in range(T):
        vals = instance.freq_values[t]
        m.addConstr(gp.quicksum(u[t][p] for p in range(len(vals))) == 1, name=f"freq_select_{t}")
    gamma = {}
    fexpr = {}
    hexpr = {}
    ivh = {}
    for t in range(T):
        vals = instance.freq_values[t]
        fexpr[t] = gp.quicksum(vals[p] * u[t][p] for p in range(len(vals)))
        hexpr[t] = gp.quicksum((30.0 / vals[p]) * u[t][p] for p in range(len(vals)))
        gamma[t] = gp.quicksum(
            instance.mu_private[k] * gp.quicksum(qk[i, j, t, k] for i in range(I) for j in range(J)) for k in range(K)
        )
        ivh[t] = instance.alpha0 + gp.quicksum(instance.alpha_stop[s] * y[s] for s in range(S)) + instance.mu_r * fexpr[t] + gamma[t]
        m.addConstr(cycle[t] == 2.0 * ivh[t] + instance.turnaround_min, name=f"cycle_{t}")
        cycle_lb = 2.0 * (instance.alpha0 + float(instance.alpha_stop.min()) * math.ceil((1 - instance.stop_budget) * S)) + instance.turnaround_min
        cycle_ub = 2.0 * (instance.alpha0 + float(instance.alpha_stop.sum()) + instance.mu_r * max(vals) + float(D[:, :, t].sum()) * float(instance.mu_private.max())) + instance.turnaround_min
        for p, val in enumerate(vals):
            m.addConstr(W[t][p] <= cycle_ub * u[t][p], name=f"W1_{t}_{p}")
            m.addConstr(W[t][p] >= cycle_lb * u[t][p], name=f"W2_{t}_{p}")
            m.addConstr(W[t][p] <= cycle[t] - cycle_lb * (1 - u[t][p]), name=f"W3_{t}_{p}")
            m.addConstr(W[t][p] >= cycle[t] - cycle_ub * (1 - u[t][p]), name=f"W4_{t}_{p}")
        m.addConstr(gp.quicksum(vals[p] * W[t][p] for p in range(len(vals))) <= instance.fleet[t], name=f"fleet_{t}")

    for i in range(I):
        A_i = gp.quicksum(ostop[i, s] * x[i, s] for s in range(S))
        for j in range(J):
            B_j = gp.quicksum(dstop[j, s] * z[j, s] for s in range(S))
            m.addConstr(zeta[i, j] >= A_i + B_j - A0[i] - B0[j], name=f"zeta_{i}_{j}")
            for t in range(T):
                m.addConstr(qtr[i, j, t] + gp.quicksum(qk[i, j, t, k] for k in range(K)) == D[i, j, t], name=f"demand_{i}_{j}_{t}")
                xpts, ypts = _log_breakpoints(float(D[i, j, t]))
                m.addGenConstrPWL(qtr[i, j, t], ltr[i, j, t], xpts, ypts, name=f"log_tr_{i}_{j}_{t}")
                for k in range(K):
                    m.addGenConstrPWL(qk[i, j, t, k], lk[i, j, t, k], xpts, ypts, name=f"log_k_{i}_{j}_{t}_{k}")
                    cpriv = (
                        instance.theta_private[k] * (instance.private_base_time[i, j, t, k] + instance.mu_r * fexpr[t])
                        + instance.private_extra_cost[i, j, t, k]
                        - instance.beta_private[k]
                    )
                    ctr = (
                        instance.theta_tr
                        * (
                            A_i
                            + hexpr[t]
                            + instance.alpha0
                            + gp.quicksum(instance.alpha_stop[s] * y[s] for s in range(S))
                            + instance.mu_r * fexpr[t]
                            + B_j
                        )
                        + instance.fare_tr
                        - instance.beta_tr
                    )
                    stationarity = (
                        cpriv
                        - ctr
                        + instance.logit_lambda[t] * instance.mu_private[k] * gamma[t]
                        + (1.0 / instance.logit_lambda[t]) * (lk[i, j, t, k] - ltr[i, j, t])
                        + rpos[i, j, t, k]
                        - rneg[i, j, t, k]
                    )
                    m.addConstr(stationarity == 0.0, name=f"kkt_{i}_{j}_{t}_{k}")

    z_eff = gp.quicksum(instance.rho_private[i, j, t, k] * qk[i, j, t, k] for i in range(I) for j in range(J) for t in range(T) for k in range(K)) / total_demand
    z_eq = gp.quicksum(D[i, j, :].sum() * zeta[i, j] for i in range(I) for j in range(J)) / total_demand
    residual = gp.quicksum(rpos[i, j, t, k] + rneg[i, j, t, k] for i in range(I) for j in range(J) for t in range(T) for k in range(K))
    m.setObjective(z_eff + instance.lambda_eq * z_eq + 0.002 * residual, GRB.MINIMIZE)
    m.optimize()
    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        y_out = np.ones(S) if fixed_y is None else np.asarray(fixed_y, dtype=float)
        f_out = instance.baseline_freq if fixed_freq is None else np.asarray(fixed_freq, dtype=float)
        return fixed_point_evaluate(instance, y_out, f_out, strategy=strategy, runtime_sec=time.perf_counter() - t0)
    y_out = np.array([y[s].X for s in range(S)])
    y_out = (y_out >= 0.5).astype(float)
    f_out = np.zeros(T)
    for t in range(T):
        vals = instance.freq_values[t]
        f_out[t] = vals[int(np.argmax([u[t][p].X for p in range(len(vals))]))]
    res = fixed_point_evaluate(
        instance,
        y_out,
        f_out,
        strategy=strategy,
        runtime_sec=time.perf_counter() - t0,
        mip_gap=m.MIPGap if m.SolCount else None,
        kkt_residual=float(residual.getValue()) / max(1.0, I * J * T * K) if m.SolCount else None,
    )
    return res


def _baseline_logit_terms(instance: Instance) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y0 = np.ones(instance.S)
    f0 = instance.baseline_freq
    A0, B0 = baseline_access(instance)
    D = instance.demand
    I, J, T, K = instance.I, instance.J, instance.T, instance.K
    Qpriv = np.zeros((I, J, T, K))
    ivh0 = np.zeros(T)
    wait0 = 30.0 / np.maximum(f0, 1e-6)
    ptr0 = np.zeros((I, J, T))
    damping = 0.60
    for _ in range(80):
        total_private = Qpriv.sum(axis=(0, 1))
        gamma = total_private @ instance.mu_private
        for t in range(T):
            ivh0[t] = instance.alpha0 + np.dot(instance.alpha_stop, y0) + instance.mu_r * f0[t] + gamma[t]
        Qnew = np.zeros_like(Qpriv)
        for t in range(T):
            ctr = (
                instance.theta_tr * (A0[:, None] + wait0[t] + ivh0[t] + B0[None, :])
                + instance.fare_tr
            )
            cpriv = np.zeros((I, J, K))
            for k in range(K):
                cpriv[:, :, k] = (
                    instance.theta_private[k] * (instance.private_base_time[:, :, t, k] + instance.mu_r * f0[t] + gamma[t])
                    + instance.private_extra_cost[:, :, t, k]
                )
            vtr = instance.beta_tr - ctr
            vpriv = instance.beta_private[None, None, :] - cpriv
            util = np.concatenate([instance.logit_lambda[t] * vtr[:, :, None], instance.logit_lambda[t] * vpriv], axis=2)
            util -= util.max(axis=2, keepdims=True)
            prob = np.exp(util)
            prob /= prob.sum(axis=2, keepdims=True)
            ptr0[:, :, t] = prob[:, :, 0]
            Qnew[:, :, t, :] = D[:, :, t, None] * prob[:, :, 1:]
        if np.max(np.abs(Qnew - Qpriv)) <= 1e-6 * max(1.0, float(D.max())):
            Qpriv = Qnew
            break
        Qpriv = damping * Qnew + (1.0 - damping) * Qpriv
    cpriv_min = np.zeros((I, J, T))
    for t in range(T):
        cpriv_min[:, :, t] = np.min(instance.rho_private[:, :, t, :], axis=2)
    return A0, B0, ivh0, ptr0 * (D > 0)


def solve_milp(
    instance: Instance,
    strategy: str = "MILP approximation",
    fixed_y: np.ndarray | None = None,
    fixed_freq: np.ndarray | None = None,
    time_limit: float = 8.0,
    output_flag: int = 0,
    response_anchor: float | None = None,
    response_anchors: tuple[float, ...] | None = (0.0, -6.0, -12.0),
) -> object:
    """Scalable MILP approximation used for the larger experiment batches.

    The binary service-design constraints match the paper model. Traveler
    response is represented by multiple local logit-response approximations
    around the all-stops baseline, and every returned design is then evaluated
    by the nonlinear fixed-point MNL evaluator.
    """
    if response_anchor is None and response_anchors:
        started = time.perf_counter()
        best = None
        gaps = []
        per_anchor_limit = max(2.0, time_limit / max(1, len(response_anchors)))
        for anchor in response_anchors:
            cand = solve_milp(
                instance,
                strategy=strategy,
                fixed_y=fixed_y,
                fixed_freq=fixed_freq,
                time_limit=per_anchor_limit,
                output_flag=output_flag,
                response_anchor=float(anchor),
                response_anchors=None,
            )
            if cand.mip_gap is not None:
                gaps.append(float(cand.mip_gap))
            if best is None or cand.objective < best.objective - 1e-9:
                best = cand
        if best is None:
            y_out = np.ones(instance.S) if fixed_y is None else np.asarray(fixed_y, dtype=float)
            f_out = instance.baseline_freq if fixed_freq is None else np.asarray(fixed_freq, dtype=float)
            return fixed_point_evaluate(instance, y_out, f_out, strategy=strategy, runtime_sec=time.perf_counter() - started)
        best.runtime_sec = time.perf_counter() - started
        best.mip_gap = max(gaps) if gaps else best.mip_gap
        return best

    t0 = time.perf_counter()
    I, J, S, T = instance.I, instance.J, instance.S, instance.T
    D = instance.demand
    A0, B0, ivh0, ptr0 = _baseline_logit_terms(instance)
    ostop = pairwise_distance(instance.origin_x, instance.origin_y, instance.stop_x, instance.stop_y) / instance.v_walk_km_min
    dstop = pairwise_distance(instance.dest_x, instance.dest_y, instance.stop_x, instance.stop_y) / instance.v_walk_km_min
    total_demand = float(D.sum())
    m = gp.Model(f"{instance.name}_{strategy}_linear")
    m.Params.OutputFlag = output_flag
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = 0.01

    y = m.addVars(S, vtype=GRB.BINARY, name="y")
    x = m.addVars(I, S, vtype=GRB.BINARY, name="x")
    z = m.addVars(J, S, vtype=GRB.BINARY, name="z")
    zeta = m.addVars(I, J, lb=0.0, name="zeta")
    phat = m.addVars(I, J, T, lb=0.005, ub=0.985, name="phat")
    cycle = m.addVars(T, lb=0.0, name="cycle")
    W = {}
    u = {}
    for t in range(T):
        vals = instance.freq_values[t]
        u[t] = m.addVars(len(vals), vtype=GRB.BINARY, name=f"u_{t}")
        W[t] = m.addVars(len(vals), lb=0.0, name=f"W_{t}")

    if fixed_y is not None:
        for s, val in enumerate(fixed_y):
            y[s].LB = y[s].UB = int(round(float(val)))
    if fixed_freq is not None:
        for t in range(T):
            vals = instance.freq_values[t]
            chosen = min(range(len(vals)), key=lambda p: abs(vals[p] - fixed_freq[t]))
            for p in range(len(vals)):
                u[t][p].LB = u[t][p].UB = 1 if p == chosen else 0

    m.addConstr(y[0] == 1)
    m.addConstr(y[S - 1] == 1)
    m.addConstr(gp.quicksum(y[s] for s in range(S)) >= math.ceil((1.0 - instance.stop_budget) * S - 1e-9))
    for s in range(S - 1):
        m.addConstr(y[s] + y[s + 1] >= 1)
    for i in range(I):
        m.addConstr(gp.quicksum(x[i, s] for s in range(S)) == 1)
        for s in range(S):
            m.addConstr(x[i, s] <= y[s])
    for j in range(J):
        m.addConstr(gp.quicksum(z[j, s] for s in range(S)) == 1)
        for s in range(S):
            m.addConstr(z[j, s] <= y[s])

    covered_terms = []
    for i in range(I):
        close = [s for s in range(S) if ostop[i, s] * instance.v_walk_km_min <= instance.coverage_radius_km]
        if close:
            covered_terms.append(instance.origin_weights[i] * gp.quicksum(x[i, s] for s in close))
    m.addConstr(gp.quicksum(covered_terms) >= instance.rho_min * float(instance.origin_weights.sum()))
    for zone in sorted(set(instance.origin_zone.tolist())):
        idx = [i for i in range(I) if instance.origin_zone[i] == zone]
        wsum = float(instance.origin_weights[idx].sum())
        lhs = gp.quicksum(instance.origin_weights[i] * gp.quicksum(ostop[i, s] * x[i, s] for s in range(S)) for i in idx)
        base = float(np.dot(instance.origin_weights[idx], A0[idx]) / wsum)
        m.addConstr(lhs / wsum - base <= instance.delta_acc_max_min)

    fexpr, hexpr, ivh = {}, {}, {}
    wait0 = 30.0 / np.maximum(instance.baseline_freq, 1e-6)
    for t in range(T):
        vals = instance.freq_values[t]
        m.addConstr(gp.quicksum(u[t][p] for p in range(len(vals))) == 1)
        fexpr[t] = gp.quicksum(vals[p] * u[t][p] for p in range(len(vals)))
        hexpr[t] = gp.quicksum((30.0 / vals[p]) * u[t][p] for p in range(len(vals)))
        ivh[t] = instance.alpha0 + gp.quicksum(instance.alpha_stop[s] * y[s] for s in range(S)) + instance.mu_r * fexpr[t]
        m.addConstr(cycle[t] == 2.0 * ivh[t] + instance.turnaround_min)
        cycle_lb = 2.0 * (instance.alpha0 + float(instance.alpha_stop.min()) * math.ceil((1.0 - instance.stop_budget) * S)) + instance.turnaround_min
        cycle_ub = 2.0 * (instance.alpha0 + float(instance.alpha_stop.sum()) + instance.mu_r * max(vals)) + instance.turnaround_min
        for p, val in enumerate(vals):
            m.addConstr(W[t][p] <= cycle_ub * u[t][p])
            m.addConstr(W[t][p] >= cycle_lb * u[t][p])
            m.addConstr(W[t][p] <= cycle[t] - cycle_lb * (1 - u[t][p]))
            m.addConstr(W[t][p] >= cycle[t] - cycle_ub * (1 - u[t][p]))
        m.addConstr(gp.quicksum(vals[p] * W[t][p] for p in range(len(vals))) <= instance.fleet[t])

    for i in range(I):
        A_i = gp.quicksum(ostop[i, s] * x[i, s] for s in range(S))
        for j in range(J):
            B_j = gp.quicksum(dstop[j, s] * z[j, s] for s in range(S))
            m.addConstr(zeta[i, j] >= A_i + B_j - A0[i] - B0[j])
            for t in range(T):
                delta_cost = (
                    instance.theta_tr
                    * ((A_i - A0[i]) + (B_j - B0[j]) + (hexpr[t] - wait0[t]) + (ivh[t] - ivh0[t]))
                )
                anchor = 0.0 if response_anchor is None else float(response_anchor)
                p_anchor = _logit_response_at_delta(ptr0[i, j, t], instance.logit_lambda[t], anchor)
                response_slope = instance.logit_lambda[t] * max(0.015, p_anchor * (1.0 - p_anchor))
                m.addConstr(phat[i, j, t] == p_anchor - response_slope * (delta_cost - anchor))

    private_cost = gp.quicksum(
        D[i, j, t] * float(np.mean(instance.rho_private[i, j, t, :])) * (1.0 - phat[i, j, t])
        for i in range(I)
        for j in range(J)
        for t in range(T)
    ) / total_demand
    equity = gp.quicksum(D[i, j, :].sum() * zeta[i, j] for i in range(I) for j in range(J)) / total_demand
    # A small stop-removal regularizer helps the MILP avoid heuristic-like local
    # optima when several coordinated removals are needed.
    retained_penalty = 0.015 * gp.quicksum(y[s] for s in range(S)) / S
    m.setObjective(private_cost + instance.lambda_eq * equity + retained_penalty, GRB.MINIMIZE)
    m.optimize()
    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL) or m.SolCount == 0:
        y_out = np.ones(S) if fixed_y is None else np.asarray(fixed_y, dtype=float)
        f_out = instance.baseline_freq if fixed_freq is None else np.asarray(fixed_freq, dtype=float)
        return fixed_point_evaluate(instance, y_out, f_out, strategy=strategy, runtime_sec=time.perf_counter() - t0)

    y_out = (np.array([y[s].X for s in range(S)]) >= 0.5).astype(float)
    f_out = np.zeros(T)
    for t in range(T):
        vals = instance.freq_values[t]
        f_out[t] = vals[int(np.argmax([u[t][p].X for p in range(len(vals))]))]
    return fixed_point_evaluate(
        instance,
        y_out,
        f_out,
        strategy=strategy,
        runtime_sec=time.perf_counter() - t0,
        mip_gap=m.MIPGap if m.SolCount else None,
    )
