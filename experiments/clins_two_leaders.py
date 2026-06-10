r"""Two-leaders empirical test + the 2-RANGE OA objective.

Step 1 (§6.5) showed a 2nd KNOWN leader + range fusion fixes the carried loop (25% -> 90-95% bounded) even
when the OA plans on leader 1 ONLY (isolating the filter benefit). Step 2 (this file's OA-2rng column) lets
the PLANNER exploit both anchors: a 12-state STLOG over [leader1, follower, leader2] whose worst-eigenvalue
targets the follower fused by BOTH ranges (clins_closed_loop.build_oac2). Hypothesis: planning on 2 ranges
should raise %bounded further AND let the follower maneuver LESS (two anchors already localize, so less
excitation is needed) -- buying back trackability/feasibility. cmd_spd = mean commanded maneuver speed.

    PYTHONPATH=src JAX_PLATFORMS=cpu uv run python experiments/clins_two_leaders.py --seeds 20
"""

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from clins_closed_loop import build_oac, build_oac2, run

# 2nd-leader placements (constant offset from leader 1; follower nominal rel pos R0 = [0,5,0]):
PLACEMENTS = [
    ("lateral flanks (collinear)", np.array([0.0, 10.0, 0.0])),
    ("one ahead (tangential)", np.array([8.0, 0.0, 0.0])),
    ("ahead + above (3D)", np.array([6.0, 0.0, 4.0])),
]


def _row(ctrl, l2, oac2, seeds):
    out = []
    for s in range(seeds):
        errs, nees, dists, _w, _vt, vig_c, _va = run(
            "oac", ctrl, s, driven="imu", l2_off=l2, oac2=oac2
        )
        out.append((
            errs[-1],
            np.median(nees),
            errs[-1] < 1.5 and np.max(dists) < 15.0,
            np.median(dists),
            np.mean(vig_c),
        ))
    r = np.array(out)
    return (
        100 * np.mean(r[:, 2]),
        np.median(r[:, 0]),
        np.median(r[:, 1]),
        np.median(r[:, 3]),
        np.median(r[:, 4]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=16)
    args = ap.parse_args()
    ctrl1, ctrl2 = build_oac(), build_oac2()
    print(
        "=== 2-range OA: does letting the planner exploit BOTH anchors beat planning on leader 1? ==="
    )
    print(
        f"{args.seeds} seeds; INS fuses both ranges throughout. OA-1rng plans on leader 1 only; OA-2rng "
        f"uses the 12-state 2-range STLOG. cmd_spd = mean commanded maneuver speed (lower = gentler).\n"
    )
    print(
        f"{'placement':>28} {'OA':>5} {'%bnd':>6} {'rec':>6} {'NEES':>7} {'distMed':>8} {'cmd_spd':>8}"
    )
    b, rec, ne, dm, cs = _row(ctrl1, None, False, args.seeds)  # 1-leader baseline
    print(
        f"{'1 leader (baseline)':>28} {'1rng':>5} {b:>5.0f}% {rec:>6.2f} {ne:>7.1f} {dm:>8.2f} {cs:>8.2f}\n"
    )
    for label, l2 in PLACEMENTS:
        for tag, ctrl, o2 in [("1rng", ctrl1, False), ("2rng", ctrl2, True)]:
            b, rec, ne, dm, cs = _row(ctrl, l2, o2, args.seeds)
            print(
                f"{label:>28} {tag:>5} {b:>5.0f}% {rec:>6.2f} {ne:>7.1f} {dm:>8.2f} {cs:>8.2f}"
            )
        print()
    print(
        "READ: does OA-2rng raise %bounded vs OA-1rng AND lower cmd_spd (maneuver less, since 2 anchors "
        "already localize)? If both, the 2-range objective is a strict win: more bounded + gentler/more "
        "trackable. Watch the 3D placement (weakest at 1rng)."
    )


if __name__ == "__main__":
    main()
