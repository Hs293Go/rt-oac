"""Generate the carried-estimation trichotomy data for the report's future-work section.

The finding-#8 divergence is isolated by three estimator setups on the *same* pure-observability
OA trajectory (mode=estimation), moving leader, 80 steps:

* ``reanchor``      -- the EKF mean is pinned to truth each step (the companion/paper validation:
  only the covariance envelope propagates along the fixed OA trajectory; cannot diverge);
* ``planontruth``  -- the EKF mean is *carried* (drifts) but the control is still planned on truth;
* ``closed``       -- the mean is carried AND the control is planned on it (the real #8 loop).

For each setup we (a) dump the seed-0 run (NEES/error time series) for the figure, and
(b) sweep seeds 0..N-1 and record the chi-square in-band fraction + median NEES for the table.
Writes ``report/data/quad_estimation_{reanchor,planontruth,}.{npz,json}`` (seed 0) and
``report/data/trichotomy.json`` (the robustness summary).

    JAX_PLATFORMS=cpu uv run python report/trichotomy.py
"""

import json
import os
import pathlib
import re
import subprocess  # noqa: S404  (fixed, trusted commands only)

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
DATA = HERE / "data"
QUAD = "examples/quadrotor_cooperative_navigation.py"
NEES_BAND = (2.70, 19.02)  # chi^2_9, 2.5--97.5%
SEEDS = 5

# (key, label, extra Hydra flags) -- the three estimator setups.
SETUPS = [
    ("reanchor", "Re-anchor mean to truth (paper validation)", ["reanchor=true"]),
    ("planontruth", "Carry mean, plan on truth", ["plan_on_truth=true"]),
    ("closed", "Carry mean, plan on estimate (#8)", []),
]
BASE = ["mode=estimation", "steps=80", "world_leader=true"]


def run(extra, capture, multirun=False):
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    cmd = ["uv", "run", "python", QUAD]
    if multirun:  # Hydra requires the -m flag before the overrides
        cmd.append("-m")
    cmd += [*BASE, *extra]
    return subprocess.run(  # noqa: S603  (fixed argv, no shell, trusted)
        cmd, cwd=REPO, env=env, capture_output=capture, text=True, check=True
    )


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    summary = {"nees_band": list(NEES_BAND), "seeds": SEEDS, "setups": {}}
    for key, label, flags in SETUPS:
        # (a) seed-0 run -> dumps the time series npz/json under the tagged name for the figure.
        run([*flags, "seed=0", f"report={DATA}"], capture=False)
        # (b) seed sweep -> parse the printed NEES median per seed for the in-band fraction.
        seeds = ",".join(str(s) for s in range(SEEDS))
        out = run([*flags, f"seed={seeds}"], capture=True, multirun=True)
        nees = [
            float(x) for x in re.findall(r"NEES median.*?([0-9.]+) \(E=9\)", out.stdout)
        ]
        errs = [
            float(x)
            for x in re.findall(r"error final [0-9.]+ m, max ([0-9.]+)", out.stdout)
        ]
        in_band = [NEES_BAND[0] <= n <= NEES_BAND[1] for n in nees]
        summary["setups"][key] = {
            "label": label,
            "n_in_band": int(sum(in_band)),
            "n_seeds": len(nees),
            "nees_median": float(sorted(nees)[len(nees) // 2]),
            "nees_worst": float(max(nees)),
            "errmax_median": float(sorted(errs)[len(errs) // 2]),
            "diverged_any": bool(max(errs) > 5.0),
        }
        print(f"{key}: {sum(in_band)}/{len(nees)} in-band, worst NEES {max(nees):.0f}")
    (DATA / "trichotomy.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {DATA / 'trichotomy.json'}")


if __name__ == "__main__":
    main()
