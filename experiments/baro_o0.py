r"""(E1,O0) DEPLOYABLE corner: does a barometer reach the 2-leader boundedness?

The barometer matched the 2nd leader on the O1 corner (baro_o1.py), but that corner is control-capped at 45%.
The real question is the DEPLOYABLE (E1,O0) corner -- the flat-output OA + geometric tracker + carried
IMU-driven translation INS that reaches ~95-100% bounded with TWO known leaders (clins_closed_loop /
clins_two_leaders / the imu_driven headline). Can a single onboard barometer replace the 2nd known anchor
THERE and still get high boundedness?

The barometer measures the follower's absolute altitude = the relative vertical position r_z (leader known),
a direct linear measurement fused into the translation INS, and the follower altitude x[6] is added to the
flat OA's STLOG observation (build_oac_baro). Held identical to clins_closed_loop's (E1,O0) loop; only the
observation set changes. Compares 1 range / 1 range + baro / 2 ranges.

    PYTHONPATH=src JAX_PLATFORMS=cpu uv run python experiments/baro_o0.py --seeds 20
"""

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import clins_closed_loop as clins

D12 = np.array([
    0.0,
    10.0,
    0.0,
])  # 2nd leader (lateral) -- the clins/imu_driven ~95% placement


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument(
        "--baro-std", type=float, default=0.3, help="barometer altitude noise std [m]"
    )
    ap.add_argument(
        "--only",
        nargs="+",
        default=["1range", "1range+baro", "2range"],
        choices=["1range", "1range+baro", "2range"],
    )
    args = ap.parse_args()
    clins.BARO_STD = args.baro_std
    configs = {
        "1range": (clins.build_oac, {"l2_off": None, "oac2": False, "baro": False}),
        "1range+baro": (
            clins.build_oac_baro,
            {"l2_off": None, "oac2": False, "baro": True},
        ),
        "2range": (clins.build_oac2, {"l2_off": D12, "oac2": True, "baro": False}),
    }
    print(
        "=== (E1,O0) DEPLOYABLE corner: barometer vs 2nd leader (flat-plan + tracker + translation INS) ==="
    )
    print(
        f"{clins.STEPS} steps, {args.seeds} seeds, baro_std={args.baro_std} m; "
        "bounded = recovery<1.5 m AND formation<15 m.\n"
    )
    print(
        f"{'config':>16} {'rec_med':>8} {'rec_p90':>8} {'NEES':>7} {'%bnd':>6} {'distMed':>8}"
    )
    for name in args.only:
        build, kw = configs[name]
        ctrl = build()
        rows = []
        for s in range(args.seeds):
            errs, nees, dists, *_ = clins.run("oac", ctrl, s, driven="imu", **kw)
            bounded = errs[-1] < 1.5 and np.max(dists) < 15.0
            rows.append((errs[-1], np.median(nees), bounded, np.median(dists)))
        r = np.array(rows)
        print(
            f"{name:>16} {np.median(r[:, 0]):>8.2f} {np.percentile(r[:, 0], 90):>8.2f} "
            f"{np.median(r[:, 1]):>7.1f} {100 * np.mean(r[:, 2]):>5.0f}% {np.median(r[:, 3]):>8.2f}"
        )
    print(
        "\nVERDICT: does '1range+baro' reach the 2-leader deployable boundedness on the (E1,O0) corner -- "
        "i.e. can a cheap barometer replace the 2nd known anchor in the DEPLOYABLE loop, not just the "
        "control-capped O1 one?"
    )


if __name__ == "__main__":
    main()
