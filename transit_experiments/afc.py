from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


BOARD = "\u4e0a\u8f66"
ALIGHT = "\u4e0b\u8f66"
DOWN = "\u4e0b\u884c"

PERIOD_BOUNDS_MIN = np.array([360, 420, 480, 540, 600, 720, 840, 960, 1020, 1110, 1350])
PERIOD_LABELS = (
    "06:00-07:00",
    "07:00-08:00",
    "08:00-09:00",
    "09:00-10:00",
    "10:00-12:00",
    "12:00-14:00",
    "14:00-16:00",
    "16:00-17:00",
    "17:00-18:30",
    "18:30-22:30",
)
BASELINE_FREQ = np.array([3, 5, 5, 3, 3, 3, 3, 4, 5, 3], dtype=float)


@dataclass(frozen=True)
class AFCDemand:
    demand_by_scenario: dict[str, np.ndarray]
    metadata: dict


def _discover_raw_root() -> Path | None:
    env = os.environ.get("TRANSIT_AFC_RAW_DIR")
    if env:
        p = Path(env)
        if p.exists():
            return p
    return None


def _read_route_438_records(raw_root: Path) -> pd.DataFrame:
    usecols = list(range(10))
    names = ["date", "pay_type", "card_id", "vehicle_id", "line", "direction", "stop_seq", "stop_name", "event", "time"]
    frames: list[pd.DataFrame] = []
    for file in sorted(raw_root.rglob("*.gz")):
        for chunk in pd.read_csv(
            file,
            compression="gzip",
            header=0,
            usecols=usecols,
            names=names,
            dtype=str,
            chunksize=300_000,
            encoding="utf-8",
        ):
            sub = chunk[chunk["line"].astype(str).str.strip().eq("438")].copy()
            if not sub.empty:
                frames.append(sub)
    if not frames:
        raise FileNotFoundError(f"No Route 438 AFC records found under {raw_root}")
    df = pd.concat(frames, ignore_index=True)
    df["time_dt"] = pd.to_datetime(df["time"], errors="coerce")
    df["seq"] = pd.to_numeric(df["stop_seq"], errors="coerce")
    df = df.dropna(subset=["time_dt", "seq"])
    df["seq"] = df["seq"].astype(int)
    return df


def _pair_board_alight(records: pd.DataFrame) -> pd.DataFrame:
    keys = ["date", "pay_type", "card_id", "vehicle_id", "line", "direction"]
    rows: list[dict] = []
    ordered = records.sort_values(keys + ["time_dt"])
    for _, group in ordered.groupby(keys, sort=False):
        board = None
        for row in group.itertuples(index=False):
            if row.event == BOARD:
                board = row
            elif row.event == ALIGHT and board is not None:
                if row.seq > board.seq and row.time_dt > board.time_dt:
                    rows.append(
                        {
                            "date": board.date,
                            "pay_type": board.pay_type,
                            "direction": board.direction,
                            "board_seq": int(board.seq),
                            "alight_seq": int(row.seq),
                            "board_stop": board.stop_name,
                            "alight_stop": row.stop_name,
                            "board_time": board.time_dt,
                            "alight_time": row.time_dt,
                        }
                    )
                board = None
    return pd.DataFrame(rows)


def _period_index(times: pd.Series) -> np.ndarray:
    minutes = times.dt.hour.to_numpy() * 60 + times.dt.minute.to_numpy()
    idx = np.searchsorted(PERIOD_BOUNDS_MIN, minutes, side="right") - 1
    return np.clip(idx, 0, len(PERIOD_LABELS) - 1)


def _map_seq_to_index(seq: pd.Series, max_seq: int, n_points: int) -> np.ndarray:
    pos = (seq.to_numpy(dtype=float) - 1.0) / max(1.0, float(max_seq - 1))
    return np.clip(np.rint(pos * (n_points - 1)).astype(int), 0, n_points - 1)


def build_route438_afc_cache(
    data_dir: Path,
    raw_root: Path | None = None,
    n_origins: int = 26,
    n_dests: int = 24,
    transit_share: float = 0.28,
    write_paired_trips: bool = False,
) -> AFCDemand:
    raw_root = raw_root or _discover_raw_root()
    if raw_root is None:
        raise FileNotFoundError("Cannot locate Beijing AFC raw-data folder. Set TRANSIT_AFC_RAW_DIR to the raw AFC directory.")

    data_dir.mkdir(parents=True, exist_ok=True)
    records = _read_route_438_records(raw_root)
    pairs = _pair_board_alight(records)
    direction_pairs = pairs[pairs["direction"].eq(DOWN)].copy()
    if direction_pairs.empty:
        raise ValueError("No paired Route 438 AFC trips found for the studied direction.")

    max_seq = int(direction_pairs["alight_seq"].max())
    direction_pairs["origin"] = _map_seq_to_index(direction_pairs["board_seq"], max_seq, n_origins)
    direction_pairs["dest"] = _map_seq_to_index(direction_pairs["alight_seq"], max_seq, n_dests)
    direction_pairs["period"] = _period_index(direction_pairs["board_time"])

    def aggregate(date_filter: pd.Series) -> np.ndarray:
        arr = np.zeros((n_origins, n_dests, len(PERIOD_LABELS)), dtype=float)
        sub = direction_pairs[date_filter]
        for (i, j, t), count in sub.groupby(["origin", "dest", "period"]).size().items():
            arr[int(i), int(j), int(t)] = float(count)
        return arr

    weekday_transit = aggregate(direction_pairs["date"].eq("2021-12-17"))
    weekend_transit = aggregate(direction_pairs["date"].eq("2021-12-18"))
    combined_transit = 0.5 * (weekday_transit + weekend_transit)
    rng = np.random.default_rng(202112)
    dispersed = np.zeros_like(combined_transit)
    for i in range(n_origins):
        for j in range(n_dests):
            if j >= i:
                corridor_distance = abs(j / max(1, n_dests - 1) - i / max(1, n_origins - 1))
                dispersed[i, j, :] = np.exp(-2.2 * corridor_distance)
    if dispersed.sum() > 0:
        period_profile = np.maximum(1.0, combined_transit.sum(axis=(0, 1)))
        period_profile = period_profile / period_profile.sum()
        dispersed *= period_profile[None, None, :]
        dispersed *= combined_transit.sum() / dispersed.sum()
        dispersed *= rng.uniform(0.85, 1.15, size=dispersed.shape)

    observed = {
        "weekday": weekday_transit,
        "weekend": weekend_transit,
        "holiday": 1.25 * (0.75 * combined_transit + 0.25 * dispersed),
    }
    demand_by_scenario = {}
    for scenario, transit_arr in observed.items():
        total = transit_arr / transit_share
        positive = total[total > 0]
        eps = 0.02 if positive.size == 0 else min(0.08, max(0.02, 0.001 * float(positive.mean())))
        demand_by_scenario[scenario] = total + eps

    long_rows = []
    for scenario, arr in demand_by_scenario.items():
        for i in range(n_origins):
            for j in range(n_dests):
                for t, label in enumerate(PERIOD_LABELS):
                    long_rows.append(
                        {
                            "scenario": scenario,
                            "origin": i,
                            "dest": j,
                            "period": t,
                            "period_label": label,
                            "total_demand": float(arr[i, j, t]),
                        }
                    )
    pd.DataFrame(long_rows).to_csv(data_dir / "route438_afc_demand.csv", index=False, encoding="utf-8-sig")
    if write_paired_trips:
        direction_pairs.to_csv(data_dir / "route438_afc_paired_trips.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "raw_data": "not distributed; set TRANSIT_AFC_RAW_DIR to rebuild the aggregate cache from authorized raw AFC files",
        "line": "438",
        "studied_direction": "downbound/toward Yongfeng Bus Station",
        "records_line_438": int(len(records)),
        "paired_trips_all_directions": int(len(pairs)),
        "paired_trips_studied_direction": int(len(direction_pairs)),
        "n_origins": n_origins,
        "n_dests": n_dests,
        "potential_od_pairs": int(n_origins * n_dests),
        "period_labels": list(PERIOD_LABELS),
        "baseline_frequency_vph": BASELINE_FREQ.tolist(),
        "transit_share_assumption": transit_share,
        "positive_observed_od_period_cells": int((direction_pairs.groupby(["origin", "dest", "period"]).size() > 0).sum()),
        "unique_observed_stop_od_pairs": int(direction_pairs.groupby(["board_seq", "alight_seq"]).ngroups),
        "source_note": "Beijing bus AFC card/QR records for 2021-12-17 and 2021-12-18; Route 438 records filtered and paired by card, vehicle, direction, and transaction time.",
    }
    (data_dir / "route438_afc_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return AFCDemand(demand_by_scenario=demand_by_scenario, metadata=metadata)


def load_route438_afc_demand(data_dir: Path) -> AFCDemand:
    demand_file = data_dir / "route438_afc_demand.csv"
    meta_file = data_dir / "route438_afc_metadata.json"
    if not demand_file.exists() or not meta_file.exists():
        return build_route438_afc_cache(data_dir)
    df = pd.read_csv(demand_file, encoding="utf-8-sig")
    metadata = json.loads(meta_file.read_text(encoding="utf-8"))
    n_origins = int(metadata["n_origins"])
    n_dests = int(metadata["n_dests"])
    n_periods = len(metadata["period_labels"])
    demand_by_scenario: dict[str, np.ndarray] = {}
    for scenario, sub in df.groupby("scenario"):
        arr = np.zeros((n_origins, n_dests, n_periods), dtype=float)
        for row in sub.itertuples(index=False):
            arr[int(row.origin), int(row.dest), int(row.period)] = float(row.total_demand)
        demand_by_scenario[str(scenario)] = arr
    return AFCDemand(demand_by_scenario=demand_by_scenario, metadata=metadata)
