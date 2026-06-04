#!/usr/bin/env bash
# Generate the progress-report data by sweeping the two headline examples with Hydra.
#
# Each invocation writes report/data/<tag>.{npz,json} via the examples' `report=` key (the
# rt_oac.report.dump hook). `-m` is Hydra multirun: one config axis -> several sequential jobs
# in one process (JAX/XLA stays warm across jobs). The data is the single source for
# report/tabulate.py (the ablation tables) and report/figures.py (the comparative plots).
#
# Run from the repo root:  bash report/sweep.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export JAX_PLATFORMS=cpu
DATA=report/data
QUAD=examples/quadrotor_cooperative_navigation.py
PLANAR=examples/simple_robot_cooperative_navigation.py

echo "### quadrotor open-loop orbit (soft-min ring, perfect feedback): hard vs soft constraint"
uv run python "$QUAD" -m soft=false,true report="$DATA"   # mode=open default

echo "### planar: log-det vs balanced (hybrid) x hard vs soft constraint"
uv run python "$PLANAR" -m hybrid=false,true soft=false,true report="$DATA"

echo "### carried-estimation trichotomy (future-work diagnostic)"
uv run python report/trichotomy.py

echo "### done -- wrote:"
ls -1 "$DATA"/*.json
