from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np

from .afc import BASELINE_FREQ, PERIOD_LABELS, load_route438_afc_demand


PRIVATE_MODES = ("car", "ebike", "bike")


@dataclass(frozen=True)
class Instance:
    name: str
    stop_names: tuple[str, ...]
    stop_x: np.ndarray
    stop_y: np.ndarray
    origin_x: np.ndarray
    origin_y: np.ndarray
    dest_x: np.ndarray
    dest_y: np.ndarray
    origin_zone: np.ndarray
    demand: np.ndarray
    period_names: tuple[str, ...]
    freq_values: tuple[tuple[int, ...], ...]
    baseline_freq: np.ndarray
    fleet: np.ndarray
    turnaround_min: float
    stop_budget: float
    coverage_radius_km: float
    rho_min: float
    delta_acc_max_min: float
    lambda_eq: float
    alpha0: float
    alpha_stop: np.ndarray
    mu_r: float
    mu_private: np.ndarray
    theta_tr: float
    theta_private: np.ndarray
    fare_tr: float
    private_extra_cost: np.ndarray
    private_base_time: np.ndarray
    beta_tr: float
    beta_private: np.ndarray
    logit_lambda: np.ndarray
    rho_private: np.ndarray
    origin_weights: np.ndarray
    v_walk_km_min: float
    source: str = "synthetic"
    stop_lon: np.ndarray | None = None
    stop_lat: np.ndarray | None = None

    @property
    def S(self) -> int:
        return len(self.stop_names)

    @property
    def I(self) -> int:
        return len(self.origin_x)

    @property
    def J(self) -> int:
        return len(self.dest_x)

    @property
    def T(self) -> int:
        return len(self.period_names)

    @property
    def K(self) -> int:
        return len(PRIVATE_MODES)


def pairwise_distance(ax: np.ndarray, ay: np.ndarray, bx: np.ndarray, by: np.ndarray) -> np.ndarray:
    return np.sqrt((ax[:, None] - bx[None, :]) ** 2 + (ay[:, None] - by[None, :]) ** 2)


def _base_instance(
    name: str,
    stop_names: list[str],
    stop_x: np.ndarray,
    stop_y: np.ndarray,
    origin_x: np.ndarray,
    origin_y: np.ndarray,
    dest_x: np.ndarray,
    dest_y: np.ndarray,
    origin_zone: np.ndarray,
    demand: np.ndarray,
    baseline_freq: tuple[int, ...],
    source: str,
) -> Instance:
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    I, J, T = demand.shape
    K = len(PRIVATE_MODES)
    dist_od = pairwise_distance(origin_x, origin_y, dest_x, dest_y)
    private_base = np.zeros((I, J, T, K))
    # minutes: car, ebike, bike
    speeds = np.array([0.58, 0.32, 0.22])
    base_time_multipliers = np.array([1.16, 0.94, 1.08, 0.88, 1.04, 0.90])
    time_multipliers = np.resize(base_time_multipliers, T)
    for t in range(T):
        peak_mult = time_multipliers[t]
        for k, speed in enumerate(speeds):
            private_base[:, :, t, k] = 4.0 + peak_mult * dist_od / speed
    extra = np.zeros((I, J, T, K))
    extra[:, :, :, 0] = 6.0 + 0.18 * dist_od[:, :, None]
    extra[:, :, :, 1] = 1.8 + 0.08 * dist_od[:, :, None]
    extra[:, :, :, 2] = 0.8 + 0.04 * dist_od[:, :, None]
    rho_private = private_base + extra
    origin_weights = demand.sum(axis=(1, 2)) + 1.0
    freq_values = tuple(tuple(range(2, 8)) for _ in range(T))
    alpha_stop = 0.43 + 0.06 * rng.random(len(stop_names))
    default_period_names = ["AM peak", "Midday", "PM peak", "Evening", "Night", "Late night"]
    if T <= len(default_period_names):
        period_names = tuple(default_period_names[:T])
    else:
        period_names = tuple(default_period_names + [f"Period {t + 1}" for t in range(len(default_period_names), T)])
    return Instance(
        name=name,
        stop_names=tuple(stop_names),
        stop_x=stop_x,
        stop_y=stop_y,
        origin_x=origin_x,
        origin_y=origin_y,
        dest_x=dest_x,
        dest_y=dest_y,
        origin_zone=origin_zone,
        demand=demand,
        period_names=period_names,
        freq_values=freq_values,
        baseline_freq=np.asarray(baseline_freq, dtype=float),
        fleet=np.array([430.0 + 25.0 * (t % 2 == 0) for t in range(T)]),
        turnaround_min=10.0,
        stop_budget=0.30,
        coverage_radius_km=2.80,
        rho_min=0.76,
        delta_acc_max_min=5.5,
        lambda_eq=0.35,
        alpha0=18.0 + 0.20 * len(stop_names),
        alpha_stop=alpha_stop,
        mu_r=0.08,
        mu_private=np.array([0.0070, 0.0045, 0.0020]),
        theta_tr=1.00,
        theta_private=np.array([1.05, 1.15, 1.30]),
        fare_tr=2.0,
        private_extra_cost=extra,
        private_base_time=private_base,
        beta_tr=40.0,
        beta_private=np.array([0.2, 0.45, -0.15]),
        logit_lambda=np.array([0.085] * T),
        rho_private=rho_private,
        origin_weights=origin_weights,
        v_walk_km_min=0.075,
        source=source,
    )


def _route_stop_coordinates(stop_lon: np.ndarray, stop_lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon_mid = np.deg2rad(float(stop_lon.mean()))
    dx = (stop_lon - stop_lon[0]) * 111.0 * np.cos(lon_mid)
    dy = (stop_lat - stop_lat[0]) * 111.0
    increments = np.sqrt(np.diff(dx) ** 2 + np.diff(dy) ** 2).clip(0.22, 1.25)
    stop_x = np.r_[0.0, np.cumsum(increments)]
    stop_y = dy - np.interp(stop_x, [stop_x[0], stop_x[-1]], [dy[0], dy[-1]])
    return stop_x, stop_y


def _corridor_centroids(stop_x: np.ndarray, stop_y: np.ndarray, n_points: int, seed: int, lateral_scale: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    positions = np.linspace(float(stop_x[0]), float(stop_x[-1]), n_points)
    base_y = np.interp(positions, stop_x, stop_y)
    jitter_x = rng.normal(0.0, 0.045, n_points)
    jitter_y = rng.normal(0.0, lateral_scale, n_points)
    return positions + jitter_x, base_y + jitter_y


def _approximate_transit_share(instance: Instance, beta_tr: float) -> float:
    y0 = np.ones(instance.S)
    served = np.flatnonzero(y0 >= 0.5)
    odist = pairwise_distance(instance.origin_x, instance.origin_y, instance.stop_x[served], instance.stop_y[served])
    ddist = pairwise_distance(instance.dest_x, instance.dest_y, instance.stop_x[served], instance.stop_y[served])
    A = odist.min(axis=1) / instance.v_walk_km_min
    B = ddist.min(axis=1) / instance.v_walk_km_min
    wait = 30.0 / np.maximum(instance.baseline_freq, 1e-6)
    D = instance.demand
    ptr = np.zeros_like(D)
    for t in range(instance.demand.shape[2]):
        ivh = instance.alpha0 + float(np.dot(instance.alpha_stop, y0)) + instance.mu_r * instance.baseline_freq[t]
        ctr = instance.theta_tr * (A[:, None] + wait[t] + ivh + B[None, :]) + instance.fare_tr
        cpriv = np.zeros((instance.I, instance.J, instance.K))
        for k in range(instance.K):
            cpriv[:, :, k] = (
                instance.theta_private[k] * (instance.private_base_time[:, :, t, k] + instance.mu_r * instance.baseline_freq[t])
                + instance.private_extra_cost[:, :, t, k]
            )
        vtr = beta_tr - ctr
        vpriv = instance.beta_private[None, None, :] - cpriv
        util = np.concatenate([instance.logit_lambda[t] * vtr[:, :, None], instance.logit_lambda[t] * vpriv], axis=2)
        util -= util.max(axis=2, keepdims=True)
        expu = np.exp(util)
        ptr[:, :, t] = expu[:, :, 0] / expu.sum(axis=2)
    return float((D * ptr).sum() / max(1e-9, D.sum()))


def _calibrate_route_beta_tr(instance: Instance, target_share: float = 0.28) -> float:
    lo, hi = -20.0, 80.0
    for _ in range(38):
        mid = 0.5 * (lo + hi)
        share = _approximate_transit_share(instance, mid)
        if share < target_share:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def make_synthetic_instance(n_stops: int, n_origins: int, n_dests: int, n_periods: int, seed: int) -> Instance:
    rng = np.random.default_rng(seed)
    length = max(4.0, 0.55 * (n_stops - 1))
    stop_x = np.linspace(0.0, length, n_stops)
    stop_y = 0.25 * np.sin(np.linspace(0, 2.3 * np.pi, n_stops))
    origin_anchor = rng.choice(n_stops, size=n_origins, replace=True)
    dest_anchor = rng.choice(n_stops, size=n_dests, replace=True)
    origin_x = stop_x[origin_anchor] + rng.normal(0.0, 0.20, n_origins)
    dest_x = stop_x[dest_anchor] + rng.normal(0.0, 0.22, n_dests)
    origin_y = stop_y[origin_anchor] + rng.normal(0.0, 0.34, n_origins)
    dest_y = stop_y[dest_anchor] + rng.normal(0.0, 0.34, n_dests)
    origin_zone = np.clip((origin_x / (length + 1e-6) * 3).astype(int), 0, 2)
    base = rng.gamma(shape=2.2, scale=9.0, size=(n_origins, n_dests, n_periods))
    dist = pairwise_distance(origin_x, origin_y, dest_x, dest_y)
    gravity = np.exp(-0.11 * dist)
    period_scale = np.array([1.25, 0.88, 1.12, 0.72, 0.58, 0.42])[:n_periods]
    demand = np.maximum(1.0, base * gravity[:, :, None] * period_scale)
    stop_names = [f"s{s + 1}" for s in range(n_stops)]
    return _base_instance(
        name=f"synthetic_{n_stops}_stops_s{seed}",
        stop_names=stop_names,
        stop_x=stop_x,
        stop_y=stop_y,
        origin_x=origin_x,
        origin_y=origin_y,
        dest_x=dest_x,
        dest_y=dest_y,
        origin_zone=origin_zone,
        demand=demand,
        baseline_freq=tuple([4, 3, 4, 2, 2, 2][:n_periods]),
        source=f"seed={seed}",
    )


def make_route_438_instance(seed: int = 438, demand_scale: float = 1.0, scenario: str = "weekday") -> Instance:
    route_file = Path(__file__).with_name("data") / "route_438.json"
    data = json.loads(route_file.read_text(encoding="utf-8"))
    stops = data["stops"]
    stop_lon = np.asarray(data["stop_lon"], dtype=float)
    stop_lat = np.asarray(data["stop_lat"], dtype=float)
    stop_x, stop_y = _route_stop_coordinates(stop_lon, stop_lat)
    afc = load_route438_afc_demand(Path(__file__).with_name("data"))
    base_demand = afc.demand_by_scenario.get(scenario, afc.demand_by_scenario["weekday"])
    demand = np.maximum(0.02, demand_scale * base_demand)
    n_origins, n_dests, n_periods = demand.shape
    origin_x, origin_y = _corridor_centroids(stop_x, stop_y, n_origins, seed=seed + 17, lateral_scale=0.18)
    dest_x, dest_y = _corridor_centroids(stop_x, stop_y, n_dests, seed=seed + 31, lateral_scale=0.20)
    origin_zone = np.clip((np.arange(n_origins) * 8 / n_origins).astype(int), 0, 7)
    baseline_freq = tuple(int(v) for v in BASELINE_FREQ[:n_periods])
    inst = _base_instance(
        name=f"beijing_route_438_{scenario}",
        stop_names=stops,
        stop_x=stop_x,
        stop_y=stop_y,
        origin_x=origin_x,
        origin_y=origin_y,
        dest_x=dest_x,
        dest_y=dest_y,
        origin_zone=origin_zone,
        demand=demand,
        baseline_freq=baseline_freq,
        source="; ".join(note["url"] for note in data["source_notes"])
        + f"; scenario={scenario}; AFC_2021_line438_paired_trips={afc.metadata.get('paired_trips_studied_direction')}; "
        + f"expanded_by_transit_share={afc.metadata.get('transit_share_assumption')}; "
        + f"demand_scale={demand_scale}; corridor_equity_zones=8",
    )
    freq_values = []
    for f in baseline_freq:
        lo = max(2, int(f) - 1)
        hi = min(8, int(f) + 3)
        freq_values.append(tuple(range(lo, hi + 1)))
    inst = replace(
        inst,
        fleet=np.array([540.0 if f >= 5 else 440.0 for f in baseline_freq], dtype=float),
        stop_budget=0.28,
        coverage_radius_km=1.60,
        rho_min=0.75,
        delta_acc_max_min=8.0,
        lambda_eq=0.20,
        alpha0=22.0,
        period_names=PERIOD_LABELS[:n_periods],
        freq_values=tuple(freq_values),
        stop_lon=stop_lon,
        stop_lat=stop_lat,
    )
    beta_tr = _calibrate_route_beta_tr(inst, target_share=0.267)
    return replace(inst, beta_tr=beta_tr)


def with_parameters(instance: Instance, **kwargs) -> Instance:
    return replace(instance, **kwargs)
