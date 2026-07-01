from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from .instances import Instance, pairwise_distance


@dataclass
class EvaluationResult:
    strategy: str
    instance: str
    y: np.ndarray
    freq: np.ndarray
    objective: float
    z_eff: float
    z_eq: float
    transit_share: float
    transit_ridership: float
    private_flow: float
    avg_access_min: float
    avg_access_increase_min: float
    coverage: float
    max_zone_access_increase: float
    retained_stops: int
    avg_frequency: float
    avg_wait_min: float
    avg_ivh_min: float
    feasible: bool
    runtime_sec: float = 0.0
    mip_gap: float | None = None
    kkt_residual: float | None = None
    fixed_point_iters: int = 0

    def as_row(self) -> dict:
        return {
            "instance": self.instance,
            "strategy": self.strategy,
            "objective": self.objective,
            "z_eff": self.z_eff,
            "z_eq": self.z_eq,
            "transit_share": self.transit_share,
            "transit_ridership": self.transit_ridership,
            "private_flow": self.private_flow,
            "avg_access_min": self.avg_access_min,
            "avg_access_increase_min": self.avg_access_increase_min,
            "coverage": self.coverage,
            "max_zone_access_increase": self.max_zone_access_increase,
            "retained_stops": self.retained_stops,
            "avg_frequency": self.avg_frequency,
            "avg_wait_min": self.avg_wait_min,
            "avg_ivh_min": self.avg_ivh_min,
            "runtime_sec": self.runtime_sec,
            "mip_gap": np.nan if self.mip_gap is None else self.mip_gap,
            "kkt_residual": np.nan if self.kkt_residual is None else self.kkt_residual,
            "fixed_point_iters": self.fixed_point_iters,
            "feasible": self.feasible,
            "y_pattern": "".join("1" if v >= 0.5 else "0" for v in self.y),
            "freq_pattern": ",".join(str(int(round(v))) for v in self.freq),
        }


def baseline_access(instance: Instance) -> tuple[np.ndarray, np.ndarray]:
    y = np.ones(instance.S)
    return access_times(instance, y)[:2]


def access_times(instance: Instance, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    served = np.flatnonzero(y >= 0.5)
    if len(served) == 0:
        inf = np.full(instance.I, np.inf)
        return inf, np.full(instance.J, np.inf), np.full(instance.I, -1), np.full(instance.J, -1)
    odist = pairwise_distance(instance.origin_x, instance.origin_y, instance.stop_x[served], instance.stop_y[served])
    ddist = pairwise_distance(instance.dest_x, instance.dest_y, instance.stop_x[served], instance.stop_y[served])
    o_idx_local = odist.argmin(axis=1)
    d_idx_local = ddist.argmin(axis=1)
    A = odist[np.arange(instance.I), o_idx_local] / instance.v_walk_km_min
    B = ddist[np.arange(instance.J), d_idx_local] / instance.v_walk_km_min
    return A, B, served[o_idx_local], served[d_idx_local]


def feasibility(instance: Instance, y: np.ndarray) -> tuple[bool, dict]:
    y = np.asarray(y, dtype=float)
    A, _, _, _ = access_times(instance, y)
    A0, _ = baseline_access(instance)
    retained = int(np.rint(y).sum())
    min_retained = math.ceil((1.0 - instance.stop_budget) * instance.S - 1e-9)
    no_consec = all(y[s] + y[s + 1] >= 0.5 for s in range(instance.S - 1))
    endpoints = y[0] >= 0.5 and y[-1] >= 0.5
    covered = np.zeros(instance.I)
    served = np.flatnonzero(y >= 0.5)
    if len(served):
        dist = pairwise_distance(instance.origin_x, instance.origin_y, instance.stop_x[served], instance.stop_y[served])
        covered = (dist.min(axis=1) <= instance.coverage_radius_km).astype(float)
    coverage = float(np.dot(instance.origin_weights, covered) / instance.origin_weights.sum())
    zone_inc = []
    for z in sorted(set(instance.origin_zone.tolist())):
        idx = np.where(instance.origin_zone == z)[0]
        weights = instance.origin_weights[idx]
        zone_inc.append(float(np.dot(weights, A[idx] - A0[idx]) / weights.sum()))
    max_zone = max(zone_inc) if zone_inc else 0.0
    ok = (
        retained >= min_retained
        and no_consec
        and endpoints
        and coverage + 1e-9 >= instance.rho_min
        and max_zone <= instance.delta_acc_max_min + 1e-9
    )
    return ok, {
        "retained": retained,
        "min_retained": min_retained,
        "coverage": coverage,
        "max_zone_access_increase": max_zone,
        "no_consecutive_skips": no_consec,
        "endpoints": endpoints,
    }


def fixed_point_evaluate(
    instance: Instance,
    y: np.ndarray,
    freq: np.ndarray,
    strategy: str,
    runtime_sec: float = 0.0,
    mip_gap: float | None = None,
    kkt_residual: float | None = None,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> EvaluationResult:
    t0 = time.perf_counter()
    y = np.asarray(y, dtype=float)
    freq = np.asarray(freq, dtype=float)
    feasible, info = feasibility(instance, y)
    A, B, _, _ = access_times(instance, y)
    A0, B0 = baseline_access(instance)
    D = instance.demand
    I, J, T, K = instance.I, instance.J, instance.T, instance.K
    Qpriv = np.zeros((I, J, T, K))
    Ptr = np.zeros((I, J, T))
    Qtr = np.zeros((I, J, T))
    ivh = np.zeros(T)
    damping = 0.55
    iters = 0
    for it in range(max_iter):
        total_private_mode = Qpriv.sum(axis=(0, 1))
        gamma = total_private_mode @ instance.mu_private
        for t in range(T):
            ivh[t] = instance.alpha0 + np.dot(instance.alpha_stop, y) + instance.mu_r * freq[t] + gamma[t]
        wait = 30.0 / np.maximum(freq, 1e-6)
        Ctr = np.zeros((I, J, T))
        Cpriv = np.zeros((I, J, T, K))
        for t in range(T):
            Ctr[:, :, t] = (
                instance.theta_tr * (A[:, None] + wait[t] + ivh[t] + B[None, :])
                + instance.fare_tr
            )
            for k in range(K):
                private_time = instance.private_base_time[:, :, t, k] + instance.mu_r * freq[t] + gamma[t]
                Cpriv[:, :, t, k] = instance.theta_private[k] * private_time + instance.private_extra_cost[:, :, t, k]
        Vtr = instance.beta_tr - Ctr
        Vpriv = instance.beta_private[None, None, None, :] - Cpriv
        Ppriv = np.zeros_like(Qpriv)
        for t in range(T):
            util = np.concatenate(
                [instance.logit_lambda[t] * Vtr[:, :, t, None], instance.logit_lambda[t] * Vpriv[:, :, t, :]],
                axis=2,
            )
            util -= util.max(axis=2, keepdims=True)
            expu = np.exp(util)
            prob = expu / expu.sum(axis=2, keepdims=True)
            Ptr[:, :, t] = prob[:, :, 0]
            Ppriv[:, :, t, :] = prob[:, :, 1:]
        Qnew = D[:, :, :, None] * Ppriv
        diff = float(np.max(np.abs(Qnew - Qpriv)))
        Qpriv = damping * Qnew + (1.0 - damping) * Qpriv
        iters = it + 1
        if diff <= tol * max(1.0, float(D.max())):
            break
    Qtr = D * Ptr
    total_demand = float(D.sum())
    transit_ridership = float(Qtr.sum())
    private_flow = float(Qpriv.sum())
    z_eff = float((instance.rho_private * Qpriv).sum() / total_demand)
    walk_inc = A[:, None] + B[None, :] - A0[:, None] - B0[None, :]
    z_eq = float((D.sum(axis=2) * np.maximum(walk_inc, 0.0)).sum() / total_demand)
    objective = z_eff + instance.lambda_eq * z_eq
    avg_access = float((D.sum(axis=2) * (A[:, None] + B[None, :])).sum() / total_demand)
    avg_access_inc = float((D.sum(axis=2) * walk_inc).sum() / total_demand)
    if not feasible:
        objective += 1e4
    runtime = runtime_sec if runtime_sec else time.perf_counter() - t0
    return EvaluationResult(
        strategy=strategy,
        instance=instance.name,
        y=y.copy(),
        freq=freq.copy(),
        objective=objective,
        z_eff=z_eff,
        z_eq=z_eq,
        transit_share=float(transit_ridership / total_demand),
        transit_ridership=transit_ridership,
        private_flow=private_flow,
        avg_access_min=avg_access,
        avg_access_increase_min=avg_access_inc,
        coverage=info["coverage"],
        max_zone_access_increase=info["max_zone_access_increase"],
        retained_stops=int(np.rint(y).sum()),
        avg_frequency=float(freq.mean()),
        avg_wait_min=float((30.0 / np.maximum(freq, 1e-6)).mean()),
        avg_ivh_min=float(ivh.mean()),
        feasible=feasible,
        runtime_sec=runtime,
        mip_gap=mip_gap,
        kkt_residual=kkt_residual,
        fixed_point_iters=iters,
    )
