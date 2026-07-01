# Public transit stop-frequency redesign experiments

This repository contains the Python code and processed replication data for the
numerical experiments in the paper:

**Public Transit Service Redesign under Private Transport Competition: Balancing
Equity and Efficiency**

The experiments study a bilevel stop-frequency redesign model with multimodal
stochastic user equilibrium and MFD-based congestion feedback. The package
generates illustrative instances, solves MILP and heuristic benchmarks, evaluates
all designs with the fixed-point mode-choice response, and produces the CSV,
table, and figure outputs used in the manuscript.

## Contents

- `transit_experiments/`: Python package for data loading, instance generation,
  equilibrium evaluation, heuristics, MILP models, plotting, and experiment
  runners.
- `transit_experiments/data/route438_afc_demand.csv`: processed aggregate
  OD-period demand for the Beijing Route 438 case study.
- `transit_experiments/data/route438_afc_metadata.json`: metadata for the
  processed AFC-derived demand.
- `transit_experiments/data/route_438.json`: cached Route 438 service and stop
  information used by the case study.
- `transit_experiments/outputs/`: CSV, LaTeX table, and figure outputs generated
  by the experiment scripts.

The raw AFC records are not included. The public data are aggregate OD-period
demand tables constructed from the raw records and are sufficient to reproduce
the reported experiments.

## Requirements

The code was developed with Python 3.11. Install the required packages with:

```bash
pip install -r requirements.txt
```

The MILP routines require a working Gurobi installation and license. If Gurobi is
not available, the data processing and plotting code can still be inspected, but
the full optimization experiments cannot be rerun.

## Reproducing the experiments

From the repository root, run:

```bash
python -m transit_experiments.run_all
```

Outputs are written to:

- `transit_experiments/outputs/*.csv`
- `transit_experiments/outputs/tables/*.tex`
- `transit_experiments/outputs/figures/*.pdf`
- `transit_experiments/outputs/figures/*.png`

The included processed Route 438 demand cache is used automatically. To rebuild
that cache from authorized raw AFC records, set `TRANSIT_AFC_RAW_DIR` to the raw
AFC directory and call `build_route438_afc_cache` in `transit_experiments/afc.py`.
The raw AFC records themselves are not distributed in this repository.

## Data notice

The processed Route 438 file is an aggregate OD-period demand table. Individual
paired AFC trip records and all raw AFC files are excluded from the public
release.

