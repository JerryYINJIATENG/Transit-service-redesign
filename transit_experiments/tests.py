from __future__ import annotations

import numpy as np

from .evaluation import feasibility, fixed_point_evaluate
from .instances import make_synthetic_instance
from .milp import solve_milp


def run_smoke_tests() -> None:
    inst = make_synthetic_instance(n_stops=8, n_origins=4, n_dests=5, n_periods=2, seed=100)
    y = np.ones(inst.S)
    ok, info = feasibility(inst, y)
    assert ok, f"all-stops pattern should be feasible: {info}"
    bad = y.copy()
    bad[0] = 0
    ok, _ = feasibility(inst, bad)
    assert not ok, "first stop must be retained"
    bad = y.copy()
    bad[2] = 0
    bad[3] = 0
    ok, _ = feasibility(inst, bad)
    assert not ok, "consecutive skipped stops should be infeasible"
    res = fixed_point_evaluate(inst, y, inst.baseline_freq, strategy="smoke")
    assert res.feasible
    assert 0.0 < res.transit_share < 1.0
    milp_res = solve_milp(inst, time_limit=12, strategy="smoke MILP")
    assert milp_res.feasible
    assert len(milp_res.y) == inst.S
    assert len(milp_res.freq) == inst.T


if __name__ == "__main__":
    run_smoke_tests()
    print("smoke tests passed")
