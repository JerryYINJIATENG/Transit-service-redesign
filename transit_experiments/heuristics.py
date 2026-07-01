from __future__ import annotations

from itertools import product
import time

import numpy as np

from .evaluation import EvaluationResult, feasibility, fixed_point_evaluate
from .instances import Instance


def current_practice(instance: Instance, strategy: str = "Current practice") -> EvaluationResult:
    y = np.ones(instance.S)
    return fixed_point_evaluate(instance, y, instance.baseline_freq, strategy=strategy)


def best_frequency_for_y(instance: Instance, y: np.ndarray, strategy: str = "Frequency search") -> EvaluationResult:
    combo_count = int(np.prod([len(vals) for vals in instance.freq_values]))
    if combo_count > 3500:
        freq = instance.baseline_freq.astype(float).copy()
        best = fixed_point_evaluate(instance, y, freq, strategy=strategy)
        improved = True
        while improved:
            improved = False
            for t, vals in enumerate(instance.freq_values):
                local_best = best
                local_freq = freq.copy()
                for val in vals:
                    cand_freq = freq.copy()
                    cand_freq[t] = float(val)
                    cand = fixed_point_evaluate(instance, y, cand_freq, strategy=strategy)
                    cycle = 2.0 * cand.avg_ivh_min + instance.turnaround_min
                    if cand_freq[t] * cycle <= instance.fleet[t] + 1e-6 and cand.objective + 1e-7 < local_best.objective:
                        local_best = cand
                        local_freq = cand_freq
                if local_best.objective + 1e-7 < best.objective:
                    best = local_best
                    freq = local_freq
                    improved = True
        return best

    best = None
    for combo in product(*instance.freq_values):
        freq = np.asarray(combo, dtype=float)
        res = fixed_point_evaluate(instance, y, freq, strategy=strategy)
        fleet_ok = True
        cycle = 2.0 * res.avg_ivh_min + instance.turnaround_min
        for t, f in enumerate(freq):
            if f * cycle > instance.fleet[t] + 1e-6:
                fleet_ok = False
                break
        if fleet_ok and (best is None or res.objective < best.objective):
            best = res
    return best if best is not None else fixed_point_evaluate(instance, y, instance.baseline_freq, strategy=strategy)


def greedy_stop_removal(instance: Instance) -> EvaluationResult:
    t0 = time.perf_counter()
    y = np.ones(instance.S)
    current = fixed_point_evaluate(instance, y, instance.baseline_freq, strategy="Greedy removal")
    improved = True
    while improved:
        improved = False
        best_candidate = current
        best_y = y.copy()
        for s in range(1, instance.S - 1):
            if y[s] < 0.5:
                continue
            cand_y = y.copy()
            cand_y[s] = 0.0
            ok, _ = feasibility(instance, cand_y)
            if not ok:
                continue
            cand = fixed_point_evaluate(instance, cand_y, instance.baseline_freq, strategy="Greedy removal")
            if cand.objective + 1e-7 < best_candidate.objective:
                best_candidate = cand
                best_y = cand_y
        if best_candidate.objective + 1e-7 < current.objective:
            y = best_y
            current = best_candidate
            improved = True
    current.runtime_sec = time.perf_counter() - t0
    return current


def alternating_local_search(instance: Instance, max_iter: int = 20, strategy: str = "Local search") -> EvaluationResult:
    t0 = time.perf_counter()
    y = np.ones(instance.S)
    current = best_frequency_for_y(instance, y, strategy=strategy)
    for _ in range(max_iter):
        moved = False
        best_candidate = current
        best_y = y.copy()
        candidates = []
        for s in range(1, instance.S - 1):
            cand = y.copy()
            cand[s] = 1.0 - cand[s]
            candidates.append(cand)
        skipped = np.flatnonzero(y < 0.5)
        kept = np.flatnonzero(y > 0.5)
        for a in skipped:
            for b in kept:
                if b in (0, instance.S - 1):
                    continue
                cand = y.copy()
                cand[a] = 1.0
                cand[b] = 0.0
                candidates.append(cand)
        for cand_y in candidates:
            ok, _ = feasibility(instance, cand_y)
            if not ok:
                continue
            cand = best_frequency_for_y(instance, cand_y, strategy=strategy)
            if cand.objective + 1e-7 < best_candidate.objective:
                best_candidate = cand
                best_y = cand_y
        if best_candidate.objective + 1e-7 < current.objective:
            y = best_y
            current = best_candidate
            moved = True
        if not moved:
            break
    current.runtime_sec = time.perf_counter() - t0
    return current
