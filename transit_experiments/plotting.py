from __future__ import annotations

from pathlib import Path
import math
from io import BytesIO
from urllib.error import URLError
from urllib.request import Request, urlopen

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd
from PIL import Image

from .evaluation import access_times
from .instances import Instance, pairwise_distance


PALETTE = {
    "Current practice": "#4E6E8E",
    "All-stops baseline": "#4E6E8E",
    "Greedy removal": "#D9822B",
    "Local search": "#3C8D7D",
    "MILP approximation": "#7B4FA3",
    "Stop-only MILP": "#C84B4B",
    "Frequency-only MILP": "#4C78A8",
    "Joint MILP": "#7B4FA3",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, out_dir: Path, name: str, tight: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def plot_synthetic_layout(instance: Instance, out_dir: Path) -> None:
    setup_style()
    demand_o = instance.demand.sum(axis=(1, 2))
    demand_d = instance.demand.sum(axis=(0, 2))
    od_demand = instance.demand.sum(axis=2)
    od_mid_x = 0.5 * (instance.origin_x[:, None] + instance.dest_x[None, :])
    od_mid_y = 0.5 * (instance.origin_y[:, None] + instance.dest_y[None, :])
    od_size = 5.0 + 18.0 * np.sqrt(od_demand / max(1e-9, float(od_demand.max())))
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.scatter(
        od_mid_x.ravel(),
        od_mid_y.ravel(),
        s=od_size.ravel(),
        color="#7A6B87",
        alpha=0.20,
        linewidth=0,
        zorder=1,
        label="OD-pair centroids",
    )
    ax.plot(instance.stop_x, instance.stop_y, color="#006D2C", linewidth=2.4, zorder=2)
    ax.scatter(instance.stop_x, instance.stop_y, s=58, color="#008B1A", edgecolor="white", linewidth=0.9, zorder=3, label="Candidate stops")
    ax.scatter(
        instance.origin_x,
        instance.origin_y,
        s=18 + 58 * demand_o / demand_o.max(),
        color="#3A66C9",
        alpha=0.72,
        edgecolor="white",
        linewidth=0.35,
        zorder=4,
        label="Origins",
    )
    ax.scatter(
        instance.dest_x,
        instance.dest_y,
        s=26 + 60 * demand_d / demand_d.max(),
        marker="^",
        color="#D55E00",
        alpha=0.76,
        edgecolor="white",
        linewidth=0.35,
        zorder=4,
        label="Destinations",
    )
    pad_x = 0.06 * (instance.stop_x.max() - instance.stop_x.min())
    pad_y = 0.45
    ax.set_xlim(instance.stop_x.min() - pad_x, instance.stop_x.max() + pad_x)
    ax.set_ylim(min(instance.origin_y.min(), instance.dest_y.min(), instance.stop_y.min()) - pad_y, max(instance.origin_y.max(), instance.dest_y.max(), instance.stop_y.max()) + pad_y)
    ax.set_xlabel("Corridor coordinate (km)")
    ax.set_ylabel("Lateral coordinate (km)")
    ax.set_title("")
    ax.set_aspect(2.4, adjustable="box")
    ax.legend(frameon=True, framealpha=0.95, loc="upper right")
    _save(fig, out_dir, "small_scale_network_layout")


def _fixed_point_choice_state(instance: Instance, y: np.ndarray, freq: np.ndarray) -> dict[str, np.ndarray]:
    y = np.asarray(y, dtype=float)
    freq = np.asarray(freq, dtype=float)
    A, B, _, _ = access_times(instance, y)
    D = instance.demand
    I, J, T, K = instance.I, instance.J, instance.T, instance.K
    Qpriv = np.zeros((instance.I, instance.J, instance.T, instance.K))
    Ptr = np.zeros((I, J, T))
    ivh = np.zeros(instance.T)
    for _ in range(500):
        total_private_mode = Qpriv.sum(axis=(0, 1))
        gamma = total_private_mode @ instance.mu_private
        wait = 30.0 / np.maximum(freq, 1e-6)
        Ppriv = np.zeros_like(Qpriv)
        for t in range(instance.T):
            ivh[t] = instance.alpha0 + np.dot(instance.alpha_stop, y) + instance.mu_r * freq[t] + gamma[t]
            ctr = instance.theta_tr * (A[:, None] + wait[t] + ivh[t] + B[None, :]) + instance.fare_tr
            cpriv = np.zeros((instance.I, instance.J, instance.K))
            for k in range(instance.K):
                private_time = instance.private_base_time[:, :, t, k] + instance.mu_r * freq[t] + gamma[t]
                cpriv[:, :, k] = instance.theta_private[k] * private_time + instance.private_extra_cost[:, :, t, k]
            vtr = instance.beta_tr - ctr
            vpriv = instance.beta_private[None, None, :] - cpriv
            util = np.concatenate([instance.logit_lambda[t] * vtr[:, :, None], instance.logit_lambda[t] * vpriv], axis=2)
            util -= util.max(axis=2, keepdims=True)
            prob = np.exp(util)
            prob /= prob.sum(axis=2, keepdims=True)
            Ptr[:, :, t] = prob[:, :, 0]
            Ppriv[:, :, t, :] = prob[:, :, 1:]
        Qnew = instance.demand[:, :, :, None] * Ppriv
        if float(np.max(np.abs(Qnew - Qpriv))) <= 1e-6 * max(1.0, float(instance.demand.max())):
            Qpriv = Qnew
            break
        Qpriv = 0.55 * Qnew + 0.45 * Qpriv
    gamma = Qpriv.sum(axis=(0, 1)) @ instance.mu_private
    return {"gamma": gamma, "ivh": ivh, "ptr": Ptr, "qtr": D * Ptr, "qpriv": Qpriv}


def _fixed_point_congestion_state(instance: Instance, y: np.ndarray, freq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    state = _fixed_point_choice_state(instance, y, freq)
    return state["gamma"], state["ivh"]


def plot_illustrative_redesign_example(instance: Instance, baseline, milp, out_dir: Path) -> None:
    setup_style()
    y0 = np.ones(instance.S)
    y1 = np.asarray(milp.y, dtype=float)
    state0 = _fixed_point_choice_state(instance, y0, baseline.freq)
    state1 = _fixed_point_choice_state(instance, y1, milp.freq)
    _, _, o_assign0, _ = access_times(instance, y0)
    _, _, o_assign1, _ = access_times(instance, y1)
    changed_origins = np.flatnonzero(o_assign0 != o_assign1)
    origin_demand = instance.demand.sum(axis=(1, 2))
    skipped = np.flatnonzero(y1 < 0.5)
    kept = np.flatnonzero(y1 >= 0.5)
    transit_gain = (state1["qtr"] - state0["qtr"]).sum(axis=2)
    od_demand = instance.demand.sum(axis=2)
    od_mid_x = 0.5 * (instance.origin_x[:, None] + instance.dest_x[None, :])
    od_mid_y = 0.5 * (instance.origin_y[:, None] + instance.dest_y[None, :])
    od_size = 4.0 + 11.0 * np.sqrt(od_demand / max(1e-9, float(od_demand.max())))
    shift_pct = 100.0 * np.maximum(transit_gain, 0.0) / np.maximum(od_demand, 1e-9)
    positive_pairs = np.argwhere((transit_gain > 1e-6) & (shift_pct >= 0.2))
    if len(positive_pairs):
        pair_order = np.argsort(shift_pct[positive_pairs[:, 0], positive_pairs[:, 1]])
        shown_pairs = positive_pairs[pair_order]
    else:
        shown_pairs = np.empty((0, 2), dtype=int)

    def bubble_size(pct: float) -> float:
        return 8.0 + 2.6 * min(float(pct), 20.0) ** 1.25

    def draw_base(ax):
        ax.scatter(
            od_mid_x.ravel(),
            od_mid_y.ravel(),
            s=od_size.ravel(),
            color="#8A8A8A",
            alpha=0.18,
            linewidth=0,
            zorder=1,
        )
        ax.plot(instance.stop_x, instance.stop_y, color="#283744", linewidth=2.2, alpha=0.75, zorder=1)
        ax.scatter(instance.stop_x[kept], instance.stop_y[kept], s=62, color="#198754", edgecolor="white", linewidth=0.8, zorder=5, label="served stops")
        if len(skipped):
            ax.scatter(instance.stop_x[skipped], instance.stop_y[skipped], s=78, facecolor="white", edgecolor="#B00020", linewidth=1.4, marker="o", zorder=6, label="skipped stops")
            ax.scatter(instance.stop_x[skipped], instance.stop_y[skipped], s=72, color="#B00020", marker="x", linewidth=1.4, zorder=7)
        ax.scatter(instance.origin_x, instance.origin_y, s=22, color="#3A66C9", alpha=0.48, edgecolor="white", linewidth=0.3, zorder=3)
        ax.scatter(instance.dest_x, instance.dest_y, s=26, marker="^", color="#D9822B", alpha=0.48, edgecolor="white", linewidth=0.3, zorder=3)
        pad_x = 0.04 * (instance.stop_x.max() - instance.stop_x.min())
        y_min = min(instance.origin_y.min(), instance.dest_y.min(), instance.stop_y.min())
        y_max = max(instance.origin_y.max(), instance.dest_y.max(), instance.stop_y.max())
        ax.set_xlim(instance.stop_x.min() - pad_x, instance.stop_x.max() + pad_x)
        ax.set_ylim(y_min - 0.16, y_max + 0.16)
        ax.set_xlabel("Corridor coordinate (km)")
        ax.set_aspect(2.4, adjustable="box")
        ax.grid(alpha=0.15)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8), sharex=True, sharey=True)
    for ax in axes:
        draw_base(ax)

    ax = axes[0]
    for count, i in enumerate(changed_origins):
        new_s = int(o_assign1[i])
        lw = 1.0 + 1.6 * origin_demand[i] / max(1.0, float(origin_demand.max()))
        ax.plot([instance.origin_x[i], instance.stop_x[new_s]], [instance.origin_y[i], instance.stop_y[new_s]], color="#D62728", linewidth=lw + 0.35, alpha=0.88, zorder=4, label="redesigned access" if count == 0 else None)
    access_box = (
        f"mean access increase: {milp.avg_access_increase_min:.2f} min"
        "\n"
        f"skipped stops: {len(skipped)}/{instance.S}"
    )
    ax.text(0.02, 0.98, access_box, transform=ax.transAxes, va="top", ha="left", fontsize=7.8, bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#8A8A8A", "alpha": 0.94})
    ax.set_title("Access reassignment after stop-skipping")
    ax.legend(frameon=True, loc="lower right", ncol=1, borderpad=0.35, handlelength=1.3)

    ax = axes[1]
    if len(shown_pairs):
        mid_x = 0.5 * (instance.origin_x[shown_pairs[:, 0]] + instance.dest_x[shown_pairs[:, 1]])
        mid_y = 0.5 * (instance.origin_y[shown_pairs[:, 0]] + instance.dest_y[shown_pairs[:, 1]])
        shown_pct = shift_pct[shown_pairs[:, 0], shown_pairs[:, 1]]
        ax.scatter(
            mid_x,
            mid_y,
            s=[bubble_size(p) for p in shown_pct],
            color="#7B4FA3",
            alpha=0.52,
            edgecolor="white",
            linewidth=0.35,
            zorder=4,
        )
        legend_levels = [level for level in (3, 6, 9, 12) if level <= max(shift_pct.max(), 0.0) + 0.5]
        if len(legend_levels) < 2:
            legend_levels = [2, 4]
        handles = [
            ax.scatter([], [], s=bubble_size(level), color="#7B4FA3", alpha=0.52, edgecolor="white", linewidth=0.35)
            for level in legend_levels[-3:]
        ]
        ax.legend(handles, [f"{level:g}% shift" for level in legend_levels[-3:]], title="OD shift", frameon=True, loc="lower right", title_fontsize=8)
    share_gain_pp = 100.0 * (milp.transit_share - baseline.transit_share)
    private_drop = 100.0 * (baseline.private_flow - milp.private_flow) / max(1e-9, baseline.private_flow)
    max_shift = float(shift_pct.max()) if shift_pct.size else 0.0
    mode_box = (
        f"OD pairs: {instance.I * instance.J}; shifted: {len(shown_pairs)}"
        "\n"
        f"transit share: +{share_gain_pp:.1f} pp"
        "\n"
        f"private flow: {private_drop:.1f}% lower"
        "\n"
        f"max OD shift: {max_shift:.1f}%"
    )
    ax.text(0.02, 0.98, mode_box, transform=ax.transAxes, va="top", ha="left", fontsize=7.8, bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#8A8A8A", "alpha": 0.94})
    ax.set_title("Private-to-transit shift by OD pair")
    axes[0].set_ylabel("Lateral coordinate (km)")
    _save(fig, out_dir, "illustrative_redesign_example")


def plot_small_bars(df: pd.DataFrame, out_dir: Path) -> None:
    setup_style()
    order = ["All-stops baseline", "Greedy removal", "Local search", "MILP approximation"]
    agg = df.groupby("strategy", as_index=False).agg(
        objective=("objective", "mean"),
        transit_share=("transit_share", "mean"),
        avg_access_increase_min=("avg_access_increase_min", "mean"),
        runtime_sec=("runtime_sec", "mean"),
    )
    agg = agg.set_index("strategy").reindex(order).reset_index()
    metrics = [
        ("objective", "Objective"),
        ("transit_share", "Transit share"),
        ("avg_access_increase_min", "Access increase (min)"),
        ("runtime_sec", "Runtime (s)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8))
    for ax, (col, title) in zip(axes.ravel(), metrics):
        colors = [PALETTE.get(s, "#666666") for s in agg["strategy"]]
        ax.bar(np.arange(len(agg)), agg[col], color=colors, width=0.62, edgecolor="black", linewidth=0.35)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(agg)))
        ax.set_xticklabels([s.replace(" ", "\n") for s in agg["strategy"]])
        if col == "transit_share":
            ax.set_ylim(0, max(0.55, agg[col].max() * 1.15))
    _save(fig, out_dir, "small_scale_grouped_metrics")


def plot_small_pareto(df: pd.DataFrame, out_dir: Path) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(5.6, 4.1))
    for strategy, sub in df.groupby("strategy"):
        ax.scatter(
            sub["z_eq"],
            sub["z_eff"],
            s=46 + 2.2 * sub["retained_stops"],
            color=PALETTE.get(strategy, "#666666"),
            label=strategy,
            alpha=0.86,
            edgecolor="white",
            linewidth=0.6,
        )
    ax.set_xlabel("Equity loss component")
    ax.set_ylabel("Private-mode cost component")
    ax.set_title("Efficiency-equity trade-off across small instances")
    ax.legend(frameon=False, loc="best")
    _save(fig, out_dir, "small_scale_pareto")


def plot_runtime_scaling(df: pd.DataFrame, out_dir: Path) -> None:
    setup_style()
    tmp = df.copy()
    tmp["n_stops"] = tmp["instance"].str.extract(r"synthetic_(\d+)_stops").astype(float)
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for strategy, sub in tmp.groupby("strategy"):
        sub = sub.sort_values("n_stops")
        ax.plot(
            sub["n_stops"],
            sub["runtime_sec"],
            marker="o",
            linewidth=1.9,
            color=PALETTE.get(strategy, "#666666"),
            label=strategy,
        )
    ax.set_xlabel("Candidate stops")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Computational scaling")
    ax.legend(frameon=False)
    _save(fig, out_dir, "small_scale_runtime")


def _route_proxy_lon_lat(instance: Instance) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if instance.stop_lon is not None and instance.stop_lat is not None:
        lon = np.asarray(instance.stop_lon, dtype=float)
        lat = np.asarray(instance.stop_lat, dtype=float)
        ox = np.interp(instance.origin_x, instance.stop_x, lon)
        oy = np.interp(instance.origin_x, instance.stop_x, lat) + 0.010 * instance.origin_y
        dx = np.interp(instance.dest_x, instance.stop_x, lon)
        dy = np.interp(instance.dest_x, instance.stop_x, lat) + 0.010 * instance.dest_y
        return lon, lat, ox, oy, dx, dy
    lon0, lat0 = 116.20, 40.005
    lon = lon0 + 0.17 * (instance.stop_x - instance.stop_x.min()) / (instance.stop_x.max() - instance.stop_x.min())
    lat = lat0 + 0.035 * instance.stop_y
    ox = lon0 + 0.17 * (instance.origin_x - instance.stop_x.min()) / (instance.stop_x.max() - instance.stop_x.min())
    oy = lat0 + 0.035 * instance.origin_y
    dx = lon0 + 0.17 * (instance.dest_x - instance.stop_x.min()) / (instance.stop_x.max() - instance.stop_x.min())
    dy = lat0 + 0.035 * instance.dest_y
    return lon, lat, ox, oy, dx, dy


def _lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    lat_rad = math.radians(lat)
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def _tile_to_lonlat(x: int, y: int, zoom: int) -> tuple[float, float]:
    n = 2**zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lon, lat


def _draw_google_tiles(ax: plt.Axes, lon_min: float, lon_max: float, lat_min: float, lat_max: float, zoom: int = 12) -> bool:
    x0, y1 = _lonlat_to_tile(lon_min, lat_min, zoom)
    x1, y0 = _lonlat_to_tile(lon_max, lat_max, zoom)
    xs = range(min(x0, x1), max(x0, x1) + 1)
    ys = range(min(y0, y1), max(y0, y1) + 1)
    if len(list(xs)) * len(list(ys)) > 48:
        zoom -= 1
        x0, y1 = _lonlat_to_tile(lon_min, lat_min, zoom)
        x1, y0 = _lonlat_to_tile(lon_max, lat_max, zoom)
        xs = range(min(x0, x1), max(x0, x1) + 1)
        ys = range(min(y0, y1), max(y0, y1) + 1)
    xs = list(xs)
    ys = list(ys)
    mosaic = Image.new("RGB", (256 * len(xs), 256 * len(ys)), color=(245, 245, 245))
    try:
        for ix, x in enumerate(xs):
            for iy, y in enumerate(ys):
                tile = None
                for server in ["mt0", "mt1", "mt2", "mt3"]:
                    try:
                        url = f"https://{server}.google.com/vt/lyrs=m&x={x}&y={y}&z={zoom}"
                        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
                        with urlopen(req, timeout=12) as resp:
                            tile = Image.open(BytesIO(resp.read())).convert("RGB")
                        break
                    except (OSError, URLError, TimeoutError, ValueError):
                        tile = None
                if tile is None:
                    return False
                mosaic.paste(tile, (256 * ix, 256 * iy))
    except (OSError, URLError, TimeoutError, ValueError):
        return False
    west, north = _tile_to_lonlat(xs[0], ys[0], zoom)
    east, south = _tile_to_lonlat(xs[-1] + 1, ys[-1] + 1, zoom)
    ax.imshow(mosaic, extent=[west, east, south, north], aspect="auto", zorder=0)
    ax.text(
        0.01,
        0.015,
        "Map data Google",
        transform=ax.transAxes,
        fontsize=6.5,
        color="#555555",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5},
        zorder=6,
    )
    return True


def plot_route_line_map(instance: Instance, out_dir: Path) -> None:
    setup_style()
    lon, lat, *_ = _route_proxy_lon_lat(instance)
    y_all = np.ones(instance.S)
    _, _, origin_stop, dest_stop = access_times(instance, y_all)
    origin_demand = instance.demand.sum(axis=(1, 2))
    dest_demand = instance.demand.sum(axis=(0, 2))
    stop_demand = np.zeros(instance.S)
    for i, s in enumerate(origin_stop):
        stop_demand[int(s)] += origin_demand[i]
    for j, s in enumerate(dest_stop):
        stop_demand[int(s)] += dest_demand[j]
    demand_scaled = stop_demand / max(1e-9, float(stop_demand.max()))
    marker_size = 34.0 + 245.0 * np.sqrt(demand_scaled)
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    lon_pad = 0.018
    lat_pad = 0.010
    has_tiles = _draw_google_tiles(ax, float(lon.min() - lon_pad), float(lon.max() + lon_pad), float(lat.min() - lat_pad), float(lat.max() + lat_pad), zoom=12)
    if not has_tiles:
        ax.set_facecolor("#F7F6F1")
        for x in np.linspace(lon.min() - lon_pad, lon.max() + lon_pad, 7):
            ax.axvline(x, color="#E8E0D2", linewidth=1.0, zorder=0)
        for y in np.linspace(lat.min() - lat_pad, lat.max() + lat_pad, 5):
            ax.axhline(y, color="#E8E0D2", linewidth=1.0, zorder=0)
    suburban_split = lon[min(max(int(0.42 * len(lon)), 1), len(lon) - 2)]
    downtown_split = lon[min(max(int(0.68 * len(lon)), 1), len(lon) - 2)]
    ax.axvspan(
        lon.min() - lon_pad,
        suburban_split,
        color="#E9F4D8",
        alpha=0.28,
        zorder=0.5,
        label="Suburban-side segment",
    )
    ax.axvspan(
        downtown_split,
        lon.max() + lon_pad,
        color="#D9ECFF",
        alpha=0.28,
        zorder=0.5,
        label="Downtown-side segment",
    )
    ax.plot(lon, lat, color="#2B6B4B", linewidth=3.0, zorder=3, label="Route 438 line")
    sc = ax.scatter(
        lon,
        lat,
        s=marker_size,
        c=stop_demand,
        cmap="YlOrRd",
        alpha=0.92,
        edgecolor="white",
        linewidth=0.85,
        zorder=4,
        label="Stops sized by baseline demand",
    )
    ax.scatter(
        [lon[0], lon[-1]],
        [lat[0], lat[-1]],
        s=[marker_size[0] + 55, marker_size[-1] + 55],
        c=[stop_demand[0], stop_demand[-1]],
        cmap="YlOrRd",
        edgecolor="#202020",
        linewidth=1.1,
        zorder=5,
    )
    terminal_labels = [
        (0, "Erli Zhuang"),
        (len(lon) // 2, "Xiyuan area"),
        (len(lon) - 1, "Yongfeng Bus Station"),
    ]
    for idx, text in terminal_labels:
        if idx == 0:
            x_offset, ha = -8, "right"
        elif idx == len(lon) - 1:
            x_offset, ha = 8, "left"
        else:
            x_offset, ha = 8, "left"
        ax.annotate(
            text,
            xy=(lon[idx], lat[idx]),
            xytext=(x_offset, 10 if idx != len(lon) // 2 else -14),
            textcoords="offset points",
            ha=ha,
            va="bottom" if idx != len(lon) // 2 else "top",
            fontsize=7.2,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#666666", "alpha": 0.88},
            arrowprops={"arrowstyle": "-", "color": "#555555", "linewidth": 0.7},
            zorder=7,
        )
    ax.text(0.18, 0.93, "suburban side", transform=ax.transAxes, fontsize=8, color="#526A2D", ha="center", va="center", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.6}, zorder=6)
    ax.text(0.82, 0.62, "downtown side", transform=ax.transAxes, fontsize=8, color="#285A7A", ha="center", va="center", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.6}, zorder=6)
    ax.set_xlim(lon.min() - lon_pad, lon.max() + lon_pad)
    ax.set_ylim(lat.min() - lat_pad, lat.max() + lat_pad)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Beijing Route 438 corridor")
    ax.legend(frameon=True, framealpha=0.96, loc="upper right")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Stop-area baseline demand")
    _save(fig, out_dir, "route438_line_map")


def _pattern_to_array(pattern: str, n: int) -> np.ndarray:
    values = np.array([1.0 if c == "1" else 0.0 for c in str(pattern).strip()])
    if len(values) != n:
        raise ValueError(f"Expected stop pattern of length {n}, got {len(values)}")
    return values


def _freq_to_array(pattern: str, n: int) -> np.ndarray:
    values = np.array([float(v) for v in str(pattern).split(",") if v != ""])
    if len(values) != n:
        raise ValueError(f"Expected frequency pattern of length {n}, got {len(values)}")
    return values


def _private_flows_for_design(instance: Instance, y: np.ndarray, freq: np.ndarray) -> np.ndarray:
    A, B, _, _ = access_times(instance, y)
    D = instance.demand
    I, J, T, K = instance.I, instance.J, instance.T, instance.K
    Qpriv = np.zeros((I, J, T, K))
    damping = 0.55
    for _ in range(500):
        total_private_mode = Qpriv.sum(axis=(0, 1))
        gamma = total_private_mode @ instance.mu_private
        wait = 30.0 / np.maximum(freq, 1e-6)
        Ppriv = np.zeros_like(Qpriv)
        for t in range(T):
            ivh = instance.alpha0 + np.dot(instance.alpha_stop, y) + instance.mu_r * freq[t] + gamma[t]
            ctr = (
                instance.theta_tr * (A[:, None] + wait[t] + ivh + B[None, :])
                + instance.fare_tr
            )
            cpriv = np.zeros((I, J, K))
            for k in range(K):
                private_time = instance.private_base_time[:, :, t, k] + instance.mu_r * freq[t] + gamma[t]
                cpriv[:, :, k] = instance.theta_private[k] * private_time + instance.private_extra_cost[:, :, t, k]
            vtr = instance.beta_tr - ctr
            vpriv = instance.beta_private[None, None, :] - cpriv
            util = np.concatenate(
                [instance.logit_lambda[t] * vtr[:, :, None], instance.logit_lambda[t] * vpriv],
                axis=2,
            )
            util -= util.max(axis=2, keepdims=True)
            prob = np.exp(util)
            prob /= prob.sum(axis=2, keepdims=True)
            Ppriv[:, :, t, :] = prob[:, :, 1:]
        Qnew = D[:, :, :, None] * Ppriv
        diff = float(np.max(np.abs(Qnew - Qpriv)))
        Qpriv = damping * Qnew + (1.0 - damping) * Qpriv
        if diff <= 1e-6 * max(1.0, float(D.max())):
            break
    return Qpriv


def _route_area_indicators(instance: Instance, y: np.ndarray, freq: np.ndarray) -> pd.DataFrame:
    Qpriv = _private_flows_for_design(instance, y, freq)
    A, B, _, _ = access_times(instance, y)
    D = instance.demand
    rows = []
    for zone in sorted(set(instance.origin_zone.tolist())):
        idx = np.where(instance.origin_zone == zone)[0]
        zone_demand = float(D[idx, :, :].sum())
        if zone_demand <= 0:
            continue
        efficiency = float((instance.rho_private[idx, :, :, :] * Qpriv[idx, :, :, :]).sum() / zone_demand)
        access_burden = float((D[idx, :, :].sum(axis=2) * (A[idx, None] + B[None, :])).sum() / zone_demand)
        rows.append(
            {
                "zone": int(zone),
                "efficiency": efficiency,
                "access_burden": access_burden,
                "demand": zone_demand,
            }
        )
    return pd.DataFrame(rows)


def _corridor_zone_polygons(instance: Instance, n_zones: int) -> list[tuple[int, np.ndarray, tuple[float, float]]]:
    lon, lat, *_ = _route_proxy_lon_lat(instance)
    if len(set(instance.origin_zone.tolist())) == n_zones:
        x_edges = np.empty(n_zones + 1)
        x_edges[0] = float(instance.stop_x.min())
        x_edges[-1] = float(instance.stop_x.max())
        for zone in range(1, n_zones):
            left = instance.origin_x[instance.origin_zone == zone - 1]
            right = instance.origin_x[instance.origin_zone == zone]
            if len(left) and len(right):
                x_edges[zone] = 0.5 * (float(left.max()) + float(right.min()))
            else:
                x_edges[zone] = float(instance.stop_x.min()) + zone * (float(instance.stop_x.max() - instance.stop_x.min()) / n_zones)
        x_edges = np.maximum.accumulate(x_edges)
    else:
        x_edges = np.linspace(float(instance.stop_x.min()), float(instance.stop_x.max()), n_zones + 1)
    width = max(0.008, 0.12 * float(lat.max() - lat.min()))
    polygons = []
    for zone in range(n_zones):
        xs = np.linspace(x_edges[zone], x_edges[zone + 1], 48)
        lons = np.interp(xs, instance.stop_x, lon)
        lats = np.interp(xs, instance.stop_x, lat)
        dlon = np.gradient(lons)
        dlat = np.gradient(lats)
        length = np.sqrt(dlon**2 + dlat**2)
        length[length == 0] = 1.0
        nx = -dlat / length
        ny = dlon / length
        upper = np.column_stack([lons + width * nx, lats + width * ny])
        lower = np.column_stack([lons - width * nx, lats - width * ny])[::-1]
        poly = np.vstack([upper, lower])
        polygons.append((zone, poly, (float(lons.mean()), float(lats.mean()))))
    return polygons


def _draw_area_map_panel(
    ax: plt.Axes,
    instance: Instance,
    polygons: list[tuple[int, np.ndarray, tuple[float, float]]],
    values: dict[int, float],
    norm: Normalize,
    cmap,
    title: str,
    lon_bounds: tuple[float, float],
    lat_bounds: tuple[float, float],
    show_ylabel: bool = False,
    show_xlabel: bool = False,
) -> None:
    lon, lat, ox, oy, dx, dy = _route_proxy_lon_lat(instance)
    has_tiles = _draw_google_tiles(ax, lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1], zoom=11)
    if not has_tiles:
        ax.set_facecolor("#F7F6F1")
        for x in np.linspace(lon_bounds[0], lon_bounds[1], 7):
            ax.axvline(x, color="#E8E0D2", linewidth=0.9, zorder=0)
        for y in np.linspace(lat_bounds[0], lat_bounds[1], 6):
            ax.axhline(y, color="#E8E0D2", linewidth=0.9, zorder=0)
    for zone, poly, center in polygons:
        val = values.get(zone, np.nan)
        patch = Polygon(
            poly,
            closed=True,
            facecolor=cmap(norm(val)) if np.isfinite(val) else "#E6E6E6",
            edgecolor="#4B4B4B",
            linewidth=0.8,
            alpha=0.76,
            zorder=2,
        )
        ax.add_patch(patch)
        if np.isfinite(val):
            ax.text(
                center[0],
                center[1],
                f"Z{zone + 1}\n{val:.2f}",
                ha="center",
                va="center",
                fontsize=7.3,
                color="#202020",
                bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 1.4},
                zorder=5,
            )
    ax.plot(lon, lat, color="#173B4D", linewidth=1.6, zorder=4)
    ax.scatter(lon, lat, s=14, color="#173B4D", edgecolor="white", linewidth=0.35, zorder=5)
    ax.scatter(ox, oy, s=18, color="#111111", alpha=0.52, edgecolor="white", linewidth=0.25, zorder=5)
    ax.scatter(dx, dy, s=20, marker="^", color="#7A4E1D", alpha=0.48, edgecolor="white", linewidth=0.25, zorder=5)
    ax.set_xlim(*lon_bounds)
    ax.set_ylim(*lat_bounds)
    ax.set_title(title, fontsize=10, pad=5)
    if show_xlabel:
        ax.set_xlabel("Longitude")
    else:
        ax.set_xlabel("")
        ax.set_xticklabels([])
    if show_ylabel:
        ax.set_ylabel("Latitude")
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])


def plot_route_area_efficiency_equity(route_df: pd.DataFrame, instance: Instance, out_dir: Path, scenario: str = "weekday") -> None:
    setup_style()
    tmp = route_df.copy()
    if "scenario" in tmp.columns:
        tmp = tmp[tmp["scenario"].astype(str).eq(scenario)]
    current = tmp[tmp["strategy"].astype(str).eq("Current practice")].iloc[0]
    joint = tmp[tmp["strategy"].astype(str).eq("Joint MILP")].iloc[0]
    designs = [
        ("Current practice", "Before redesign", current),
        ("Joint MILP", "After joint redesign", joint),
    ]
    indicator_frames = []
    for strategy, stage, row in designs:
        y = _pattern_to_array(row["y_pattern"], instance.S)
        freq = _freq_to_array(row["freq_pattern"], instance.T)
        indicators = _route_area_indicators(instance, y, freq)
        indicators["strategy"] = strategy
        indicators["stage"] = stage
        indicator_frames.append(indicators)
    indicators = pd.concat(indicator_frames, ignore_index=True)
    indicators.to_csv(out_dir.parent / "route438_area_indicators.csv", index=False, encoding="utf-8-sig")

    n_zones = int(indicators["zone"].max()) + 1
    polygons = _corridor_zone_polygons(instance, n_zones)
    lon, lat, *_ = _route_proxy_lon_lat(instance)
    lon_pad = 0.014
    lat_pad = 0.013
    lon_bounds = (float(lon.min() - lon_pad), float(lon.max() + lon_pad))
    lat_bounds = (float(lat.min() - lat_pad), float(lat.max() + lat_pad))
    eff_norm = Normalize(vmin=float(indicators["efficiency"].min()), vmax=float(indicators["efficiency"].max()))
    eq_norm = Normalize(vmin=float(indicators["access_burden"].min()), vmax=float(indicators["access_burden"].max()))
    eff_cmap = plt.get_cmap("YlGnBu")
    eq_cmap = plt.get_cmap("YlOrRd")

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 7.2), sharex=True, sharey=True)
    for row_idx, (_strategy, stage, _row) in enumerate(designs):
        sub = indicators[indicators["stage"].eq(stage)]
        eff_values = dict(zip(sub["zone"].astype(int), sub["efficiency"].astype(float)))
        eq_values = dict(zip(sub["zone"].astype(int), sub["access_burden"].astype(float)))
        _draw_area_map_panel(
            axes[row_idx, 0],
            instance,
            polygons,
            eff_values,
            eff_norm,
            eff_cmap,
            f"{stage}: efficiency",
            lon_bounds,
            lat_bounds,
            show_ylabel=True,
            show_xlabel=False,
        )
        _draw_area_map_panel(
            axes[row_idx, 1],
            instance,
            polygons,
            eq_values,
            eq_norm,
            eq_cmap,
            f"{stage}: access burden",
            lon_bounds,
            lat_bounds,
            show_xlabel=False,
        )
    eff_sm = mpl.cm.ScalarMappable(norm=eff_norm, cmap=eff_cmap)
    eff_sm.set_array([])
    eq_sm = mpl.cm.ScalarMappable(norm=eq_norm, cmap=eq_cmap)
    eq_sm.set_array([])
    cbar_eff = fig.colorbar(eff_sm, ax=axes[:, 0], orientation="horizontal", fraction=0.042, pad=0.075)
    cbar_eff.set_label("Efficiency indicator")
    cbar_eq = fig.colorbar(eq_sm, ax=axes[:, 1], orientation="horizontal", fraction=0.042, pad=0.075)
    cbar_eq.set_label("Access-equity indicator (min)")
    fig.suptitle("Spatial performance by corridor analysis zone", y=0.982, fontsize=10.5)
    fig.text(
        0.5,
        0.012,
        "Zones are generated corridor analysis zones based on synthetic OD origins; darker colors indicate larger indicator values.",
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.93, wspace=0.08, hspace=0.15)
    _save(fig, out_dir, "route438_area_efficiency_equity_maps", tight=False)

    panel_specs = [
        ("current_efficiency", "Current practice: efficiency", "efficiency", eff_norm, eff_cmap),
        ("joint_efficiency", "Joint MILP: efficiency", "efficiency", eff_norm, eff_cmap),
        ("current_access_burden", "Current practice: access burden", "access_burden", eq_norm, eq_cmap),
        ("joint_access_burden", "Joint MILP: access burden", "access_burden", eq_norm, eq_cmap),
    ]
    for name, title, metric, norm, cmap in panel_specs:
        stage = "Before redesign" if name.startswith("current") else "After joint redesign"
        sub = indicators[indicators["stage"].eq(stage)]
        values = dict(zip(sub["zone"].astype(int), sub[metric].astype(float)))
        fig_single, ax = plt.subplots(figsize=(5.4, 4.0))
        _draw_area_map_panel(
            ax,
            instance,
            polygons,
            values,
            norm,
            cmap,
            title,
            lon_bounds,
            lat_bounds,
            show_ylabel=True,
            show_xlabel=True,
        )
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig_single.colorbar(sm, ax=ax, fraction=0.042, pad=0.025)
        cbar.set_label("Efficiency indicator" if metric == "efficiency" else "Access-equity indicator (min)")
        _save(fig_single, out_dir, f"route438_area_{name}")


def plot_route_pattern(route_df: pd.DataFrame, stop_x: np.ndarray, out_dir: Path) -> None:
    setup_style()
    strategies = ["Current practice", "Stop-only MILP", "Frequency-only MILP", "Joint MILP"]
    if "scenario" in route_df.columns:
        route_df = route_df[route_df["scenario"].eq("weekday")]
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    y_levels = np.arange(len(strategies))[::-1]
    for level, strategy in zip(y_levels, strategies):
        row = route_df.loc[route_df["strategy"] == strategy].iloc[0]
        pattern = np.array([int(c) for c in row["y_pattern"]])
        ax.hlines(level, stop_x.min(), stop_x.max(), color="#C8CDD2", linewidth=1.0)
        ax.scatter(stop_x[pattern == 1], np.full((pattern == 1).sum(), level), s=44, color=PALETTE.get(strategy, "#555555"), zorder=3)
        if (pattern == 0).any():
            ax.scatter(stop_x[pattern == 0], np.full((pattern == 0).sum(), level), s=38, color="white", edgecolor="#A13E3E", linewidth=1.2, zorder=4)
    ax.set_yticks(y_levels)
    ax.set_yticklabels(strategies)
    ax.set_xlabel("Route-order distance proxy (km)")
    ax.set_title("Route 438 retained and skipped stops by strategy")
    ax.set_ylim(-0.7, len(strategies) - 0.3)
    _save(fig, out_dir, "route438_stop_patterns")


def plot_route_dashboard(route_df: pd.DataFrame, out_dir: Path) -> None:
    setup_style()
    order = ["Current practice", "Stop-only MILP", "Frequency-only MILP", "Joint MILP"]
    scenarios = ["weekday", "weekend", "holiday"] if "scenario" in route_df.columns else ["weekday"]
    metrics = [("transit_share", "Transit share"), ("objective", "Objective")]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    width = 0.18
    x = np.arange(len(scenarios))
    for ax, (col, title) in zip(axes.ravel(), metrics):
        for idx, strategy in enumerate(order):
            vals = []
            for scenario in scenarios:
                sub = route_df[(route_df["strategy"] == strategy)]
                if "scenario" in sub.columns:
                    sub = sub[sub["scenario"] == scenario]
                vals.append(float(sub.iloc[0][col]))
            ax.bar(x + (idx - 1.5) * width, vals, width=width, color=PALETTE.get(strategy, "#777777"), edgecolor="black", linewidth=0.3, label=strategy)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([s.capitalize() for s in scenarios])
        if col == "transit_share":
            ax.set_ylim(0, max(0.80, route_df[col].max() * 1.12))
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    _save(fig, out_dir, "route438_strategy_dashboard")


def plot_route_scenario_surface(route_df: pd.DataFrame, out_dir: Path) -> None:
    setup_style()
    scenarios = ["weekend", "weekday", "holiday"]
    strategies = ["Current practice", "Stop-only MILP", "Frequency-only MILP", "Joint MILP"]
    Z = np.zeros((len(scenarios), len(strategies)))
    for i, scenario in enumerate(scenarios):
        for j, strategy in enumerate(strategies):
            Z[i, j] = route_df[(route_df["scenario"] == scenario) & (route_df["strategy"] == strategy)]["transit_share"].iloc[0]
    X, Y = np.meshgrid(np.arange(len(strategies)), np.arange(len(scenarios)))
    fig = plt.figure(figsize=(6.7, 4.7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap="jet",
        linewidth=0.55,
        edgecolor="black",
        antialiased=True,
        rstride=1,
        cstride=1,
    )
    ax.set_xticks(np.arange(len(strategies)))
    ax.set_xticklabels(["Current", "Stop", "Freq.", "Joint"], rotation=18, ha="right")
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_yticklabels(["Weekend", "Weekday", "Holiday"])
    ax.set_xlabel("Strategy", labelpad=8)
    ax.set_ylabel("Scenario", labelpad=9)
    ax.set_zlabel("Transit share", labelpad=8)
    ax.set_zlim(0.585, 0.715)
    ax.set_title("Route 438 scenario response", pad=8)
    ax.view_init(elev=26, azim=-128)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((0.96, 0.96, 0.96, 0.45))
        axis.pane.set_edgecolor((0.80, 0.80, 0.80, 0.65))
    cbar = fig.colorbar(surf, ax=ax, fraction=0.03, pad=0.06, shrink=0.72)
    cbar.set_label("Transit share")
    fig.subplots_adjust(left=0.02, right=0.91, bottom=0.02, top=0.92)
    _save(fig, out_dir, "route438_scenario_surface", tight=False)


def plot_mu_surfaces(mu_df: pd.DataFrame, out_dir: Path) -> None:
    setup_style()
    milp_eff = mu_df[mu_df["strategy_short"] == "MILP"].pivot_table(
        index="mu_ebike", columns="mu_car", values="z_eff", aggfunc="mean"
    )
    best_benchmark = (
        mu_df[mu_df["strategy_short"].isin(["C", "S", "F"])]
        .groupby(["mu_ebike", "mu_car"], as_index=False)["z_eff"]
        .min()
        .pivot_table(index="mu_ebike", columns="mu_car", values="z_eff", aggfunc="mean")
    )
    X, Y = np.meshgrid(milp_eff.columns.values, milp_eff.index.values)
    fig = plt.figure(figsize=(9.0, 4.6))
    for idx, (Z, title, zlabel, cmap) in enumerate(
        [
            (milp_eff.values, "(a) MILP efficiency objective", "Efficiency objective", "viridis"),
            (best_benchmark.values - milp_eff.values, "(b) Improvement over best benchmark", "Efficiency improvement", "magma"),
        ],
        start=1,
    ):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0.35, edgecolor="#444444", antialiased=True)
        ax.set_title(title, pad=6)
        ax.set_xlabel(r"$\mu_{\mathrm{car}}$", labelpad=5)
        ax.set_ylabel(r"$\mu_{\mathrm{e-bike}}$", labelpad=5)
        ax.set_zlabel("")
        ax.view_init(elev=27, azim=-130)
        ax.tick_params(axis="both", labelsize=7)
        ax.tick_params(axis="z", labelsize=7)
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.set_facecolor((0.96, 0.96, 0.96, 0.42))
            axis.pane.set_edgecolor((0.80, 0.80, 0.80, 0.65))
        cbar = fig.colorbar(surf, ax=ax, fraction=0.034, pad=0.06, shrink=0.68)
        cbar.set_label(zlabel, fontsize=8, labelpad=6)
        cbar.ax.tick_params(labelsize=7)
    fig.subplots_adjust(left=0.02, right=0.96, bottom=0.02, top=0.92, wspace=0.16)
    _save(fig, out_dir, "route438_mu_efficiency_surface", tight=False)

    equity_cmap = LinearSegmentedColormap.from_list("paper_equity", ["#F8FBF7", "#8DC5BA", "#3F7F93", "#4B3F72"])
    equity_values = mu_df[mu_df["strategy_short"].isin(["S", "MILP"])]["z_eq"]
    equity_norm = Normalize(vmin=0.0, vmax=max(0.75, float(equity_values.max())))
    fig = plt.figure(figsize=(9.0, 4.6))
    for idx, label in enumerate(["S", "MILP"], start=1):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        pivot = mu_df[mu_df["strategy_short"] == label].pivot_table(
            index="mu_ebike", columns="mu_car", values="z_eq", aggfunc="mean"
        )
        X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)
        surf = ax.plot_surface(
            X,
            Y,
            pivot.values,
            cmap=equity_cmap,
            norm=equity_norm,
            linewidth=0.35,
            edgecolor="#444444",
            antialiased=True,
        )
        ax.contour(X, Y, pivot.values, zdir="z", offset=0.0, levels=5, colors="#4A4A4A", linewidths=0.45, alpha=0.55)
        panel = "(a)" if idx == 1 else "(b)"
        ax.set_title(f"{panel} {label} equity objective", pad=6)
        ax.set_xlabel(r"$\mu_{\mathrm{car}}$", labelpad=5)
        ax.set_ylabel(r"$\mu_{\mathrm{e-bike}}$", labelpad=5)
        ax.set_zlabel("")
        ax.set_zlim(0.0, equity_norm.vmax)
        ax.view_init(elev=27, azim=-130)
        ax.tick_params(axis="both", labelsize=7)
        ax.tick_params(axis="z", labelsize=7)
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.set_facecolor((0.96, 0.96, 0.96, 0.42))
            axis.pane.set_edgecolor((0.80, 0.80, 0.80, 0.65))
        cbar = fig.colorbar(surf, ax=ax, fraction=0.034, pad=0.06, shrink=0.68)
        cbar.set_label("Equity objective", fontsize=8, labelpad=6)
        cbar.ax.tick_params(labelsize=7)
    fig.subplots_adjust(left=0.02, right=0.96, bottom=0.02, top=0.92, wspace=0.16)
    _save(fig, out_dir, "route438_mu_equity_surface", tight=False)


def plot_equity_tradeoff(trade_df: pd.DataFrame, out_dir: Path) -> None:
    setup_style()
    milp = trade_df[trade_df["strategy_short"] == "MILP"].sort_values("lambda_eq").copy()
    current = trade_df[(trade_df["strategy_short"] == "C")].iloc[0]
    x = np.arange(len(milp))
    labels = [f"{v:g}" for v in milp["lambda_eq"]]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.35))

    axes[0].plot(
        x,
        milp["z_eff"],
        marker="o",
        linewidth=2.0,
        color="#2F6F73",
        label="Efficiency",
    )
    axes[0].plot(
        x,
        milp["objective"],
        marker="s",
        linewidth=1.6,
        color="#7B4FA3",
        label="Total objective",
    )
    axes[0].axhline(float(current["z_eff"]), color="#4E6E8E", linestyle="--", linewidth=1.2, label="Current efficiency")
    axes[0].set_ylabel("Objective value")
    axes[0].set_title("(a) Efficiency response")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")

    axes[1].plot(
        x,
        milp["z_eq"],
        marker="o",
        linewidth=2.0,
        color="#C84B4B",
        label="Equity objective",
    )
    axes[1].bar(
        x,
        milp["avg_access_increase_min"],
        width=0.56,
        color="#E8A35D",
        alpha=0.48,
        edgecolor="#8A5A1D",
        linewidth=0.35,
        label="Added walk",
    )
    axes[1].set_ylabel("Minutes")
    axes[1].set_title("(b) Access-equity burden")
    axes[1].legend(frameon=False, fontsize=7, loc="upper right")

    axes[2].step(
        x,
        milp["retained_stops"],
        where="mid",
        linewidth=2.0,
        color="#505050",
        label="Retained stops",
    )
    ax2 = axes[2].twinx()
    ax2.plot(
        x,
        milp["avg_frequency"],
        marker="D",
        linewidth=1.7,
        color="#4C78A8",
        label="Frequency",
    )
    axes[2].set_ylabel("Retained stops")
    ax2.set_ylabel("Frequency (veh/h)")
    axes[2].set_title("(c) Design adjustment")
    handles1, labels1 = axes[2].get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    axes[2].legend(handles1 + handles2, labels1 + labels2, frameon=False, fontsize=7, loc="center right")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_xlabel(r"Equity weight $\lambda^{eq}$")
        ax.grid(alpha=0.18)
    fig.subplots_adjust(left=0.065, right=0.94, bottom=0.24, top=0.84, wspace=0.36)
    _save(fig, out_dir, "route438_equity_tradeoff")


def plot_sensitivity_heatmaps(sens_df: pd.DataFrame, out_dir: Path) -> None:
    setup_style()
    cmap = LinearSegmentedColormap.from_list("paper_teal_gold", ["#F7F4EA", "#9DC6BE", "#2F6F73"])
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    for ax, value, title in [
        (axes[0], "objective", "Objective"),
        (axes[1], "transit_share", "Transit share"),
    ]:
        pivot = sens_df.pivot_table(index="lambda_eq", columns="stop_budget", values=value, aggfunc="mean")
        im = ax.imshow(pivot.values, origin="lower", cmap=cmap, aspect="auto")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([f"{c:.2f}" for c in pivot.index])
        ax.set_xlabel("Stop-removal allowance")
        ax.set_ylabel("Equity weight")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    _save(fig, out_dir, "route438_sensitivity_heatmaps")


def plot_sensitivity_surface(sens_df: pd.DataFrame, out_dir: Path) -> None:
    setup_style()
    fig = plt.figure(figsize=(6.2, 4.6))
    ax = fig.add_subplot(111, projection="3d")
    pivot = sens_df.pivot_table(index="lambda_eq", columns="stop_budget", values="transit_share", aggfunc="mean")
    X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)
    Z = pivot.values
    ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0.2, edgecolor="white", alpha=0.92)
    ax.set_xlabel("Stop-removal allowance")
    ax.set_ylabel("Equity weight")
    ax.set_zlabel("Transit share")
    ax.set_title("Route 438 sensitivity surface")
    ax.view_init(elev=26, azim=-128)
    _save(fig, out_dir, "route438_sensitivity_surface")
