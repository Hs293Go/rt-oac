"""M0 + M1 evaluation for the carried-estimation action plan.

M0 (consistency audit): per-sub-block NEES (pos / rot / vel, each ~chi^2_3, E=3) on the
**re-anchored** run (mean pinned to truth, so it is correct) vs the **carried** run -- which
subspace is the filter over-confident in?

M1 (training-wheels discriminator): anchor the carried ESEKF to truth for the first ``hold``
steps at strength ``alpha0`` (optionally annealing over ``anneal`` more steps), then release to
carried. The "wheels-off" metrics (NEES / max error over the alpha=0 steps after warmup) say
whether the loop **survives release**. If releasing at any hold diverges like carried, the #8
instability is steady-state, not a startup transient -- which dictates the rest of the program.

Runs the Hydra quadrotor example via multirun (one JIT compile amortized over the seed sweep)
and parses its printed metrics. Sim-only: the anchor uses ground truth.

    JAX_PLATFORMS=cpu uv run python experiments/anchor_anneal_eval.py [--seeds N] [--steps S]
"""

import argparse
import os
import pathlib
import re
import subprocess  # noqa: S404  (fixed, trusted commands only)
import threading
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
QUAD = "examples/quadrotor_cooperative_navigation.py"
BAND = (2.70, 19.02)  # chi^2_9 95%
DIVERGE_M = 5.0  # the example's divergence threshold (err or dist > 5 m)
IDLE_TIMEOUT = (
    200.0  # liveliness watchdog: kill a run that emits no output for this long
)

RE_NEES = re.compile(r"NEES median \(after step \d+\) ([0-9.]+) \(E=9\)")
RE_BLOCK = re.compile(
    r"block NEES \(after \d+\) pos ([0-9.]+) rot ([0-9.]+) vel ([0-9.]+)"
)
RE_OFF = re.compile(
    r"wheels-off \(alpha=0, frac [0-9.]+\): NEES median ([0-9.]+) err max ([0-9.]+)"
)
RE_ERRMAX = re.compile(r"rel-pos error final [0-9.]+ m, max ([0-9.]+)")


def sweep(flags, seeds, steps, label="run"):
    """Run mode=estimation over seeds 0..N-1, STREAMING progress with a liveliness watchdog.

    Liveliness/progress protocol (so a stall is visible, not an invisible hang):
      * the child runs unbuffered (``python -u`` + PYTHONUNBUFFERED) and its stdout is streamed,
        not captured -- each parsed seed prints a flushed ``[+<elapsed>s] <label>: seed k/N`` line;
      * an inactivity watchdog (a Timer reset on every output line) KILLS the run if it emits
        nothing for ``IDLE_TIMEOUT`` s (catches a hung JIT compile / stuck solve) and flags it;
      * partial results are returned for whatever completed before a stall.
    Run the harness itself with ``python -u`` so the redirect file updates live.
    """
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONUNBUFFERED": "1"}
    seed_arg = ",".join(str(s) for s in range(seeds))
    cmd = [
        "uv",
        "run",
        "python",
        "-u",
        QUAD,
        "-m",
        "mode=estimation",
        f"steps={steps}",
        f"seed={seed_arg}",
        *flags,
    ]
    parsed = {"nees": [], "block": [], "off": [], "errmax": []}
    t0 = time.monotonic()
    proc = subprocess.Popen(  # noqa: S603  (fixed argv, no shell, trusted)
        cmd,
        cwd=str(REPO),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    stalled = {"v": False}

    def _stall():
        stalled["v"] = True
        proc.kill()

    timer = threading.Timer(IDLE_TIMEOUT, _stall)
    timer.start()
    done = 0
    for raw in (
        proc.stdout
    ):  # yields on every \n or \r (tqdm), so any activity resets the watchdog
        timer.cancel()
        timer = threading.Timer(IDLE_TIMEOUT, _stall)
        timer.start()
        line = raw.rstrip()
        if m := RE_NEES.search(line):
            parsed["nees"].append(float(m.group(1)))
        if m := RE_ERRMAX.search(line):
            parsed["errmax"].append(float(m.group(1)))
        if m := RE_OFF.search(line):
            parsed["off"].append((float(m.group(1)), float(m.group(2))))
        if m := RE_BLOCK.search(line):
            parsed["block"].append(tuple(float(a) for a in m.groups()))
            done += 1
            em = parsed["errmax"][-1] if parsed["errmax"] else float("nan")
            el = time.monotonic() - t0
            print(
                f"  [+{el:5.0f}s] {label}: seed {done}/{seeds} posNEES={m.group(1)} "
                f"errmax={em:.2f}",
                flush=True,
            )
    timer.cancel()
    proc.wait()
    if stalled["v"]:
        el = time.monotonic() - t0
        print(
            f"  [+{el:.0f}s] {label}: STALLED (>{IDLE_TIMEOUT:.0f}s no output) -- KILLED, "
            f"{done}/{seeds} seeds done",
            flush=True,
        )
    return parsed


def med(xs):
    return float(np.median(xs)) if len(xs) else float("nan")


def consistency_audit(seeds, steps):
    print(
        "\n=== M0  consistency audit: per-block NEES (E=3 each), median over seeds ==="
    )
    print(
        f"{'setup':<28}{'pos':>8}{'rot':>8}{'vel':>8}   (>>3 = over-confident in that block)"
    )
    for label, flags in [
        ("re-anchor (mean=truth)", ["reanchor=true"]),
        ("carried (#8)", []),
    ]:
        r = sweep(flags, seeds, steps)
        b = np.array(r["block"])  # (seeds, 3)
        print(
            f"{label:<28}{med(b[:, 0]):>8.1f}{med(b[:, 1]):>8.1f}{med(b[:, 2]):>8.1f}"
        )


def anneal_discriminator(seeds, steps):
    print(
        "\n=== M1  training-wheels discriminator: survival after release (alpha0=1) ==="
    )
    print(
        f"{'release schedule':<28}{'off-NEES':>10}{'off-errmax':>12}{'bounded':>9}{'diverged':>10}"
    )
    configs = [
        ("carried (no wheels)", ["anchor_alpha0=0.0"]),
        (
            f"sharp release @ {steps // 4}",
            ["anchor_alpha0=1.0", f"anchor_hold={steps // 4}"],
        ),
        (
            f"sharp release @ {steps // 2}",
            ["anchor_alpha0=1.0", f"anchor_hold={steps // 2}"],
        ),
        (
            "gradual wean (hold 10, anneal)",
            ["anchor_alpha0=1.0", "anchor_hold=10", f"anchor_anneal={steps // 2}"],
        ),
    ]
    for label, flags in configs:
        r = sweep(flags, seeds, steps)
        off = np.array(r["off"]) if r["off"] else np.empty((0, 2))
        errmax = np.array(r["errmax"])
        nees_off = off[:, 0] if len(off) else np.array([])
        err_off = off[:, 1] if len(off) else np.array([])
        bounded = float(np.mean(err_off < 1.5)) if len(err_off) else float("nan")
        diverged = float(np.mean(errmax > DIVERGE_M)) if len(errmax) else float("nan")
        print(
            f"{label:<28}{med(nees_off):>10.1f}{med(err_off):>12.2f}"
            f"{bounded:>9.2f}{diverged:>10.2f}"
        )
    print(f"(bounded = frac of seeds with wheels-off max err < 1.5 m; band {BAND})")


def m2_consistency_sweep(seeds, steps):
    print(
        "\n=== M2 falsification: carried loop + position-covariance floor (truth-free) ==="
    )
    print(
        f"{'config':<26}{'pos-NEES':>10}{'errmax(med)':>13}{'bounded':>9}{'diverged':>10}"
    )
    for label, flags in [
        ("carried (no floor)", ["p_pos_floor=0.0"]),
        ("floor 0.1", ["p_pos_floor=0.1"]),
        ("floor 0.2", ["p_pos_floor=0.2"]),
        ("floor 0.5", ["p_pos_floor=0.5"]),
        ("floor 1.0", ["p_pos_floor=1.0"]),
    ]:
        r = sweep(flags, seeds, steps)
        block = np.array(r["block"]) if r["block"] else np.empty((0, 3))
        errmax = np.array(r["errmax"])
        pos_n = med(block[:, 0]) if len(block) else float("nan")
        bounded = float(np.mean(errmax < 1.5)) if len(errmax) else float("nan")
        diverged = float(np.mean(errmax > DIVERGE_M)) if len(errmax) else float("nan")
        print(
            f"{label:<26}{pos_n:>10.1f}{med(errmax):>13.2f}"
            f"{bounded:>9.2f}{diverged:>10.2f}"
        )
    print(
        "(bounded = frac errmax < 1.5 m; diverged = frac errmax > 5 m; pos-NEES target ~3)"
    )


def filter_compare(seeds, steps):
    print("\n=== EKF vs UKF: carried loop, per-block NEES + robustness ===")
    print(
        f"{'filter':<12}{'pos-NEES':>10}{'rot-NEES':>10}{'errmax(med)':>13}"
        f"{'bounded':>9}{'diverged':>10}"
    )
    for label, flags in [("EKF", ["filter=ekf"]), ("UKF", ["filter=ukf"])]:
        r = sweep(flags, seeds, steps, label=label)
        block = np.array(r["block"]) if r["block"] else np.empty((0, 3))
        errmax = np.array(r["errmax"])
        bounded = float(np.mean(errmax < 1.5)) if len(errmax) else float("nan")
        diverged = float(np.mean(errmax > DIVERGE_M)) if len(errmax) else float("nan")
        pos_n = med(block[:, 0]) if len(block) else float("nan")
        rot_n = med(block[:, 1]) if len(block) else float("nan")
        print(
            f"{label:<12}{pos_n:>10.1f}{rot_n:>10.1f}{med(errmax):>13.2f}"
            f"{bounded:>9.2f}{diverged:>10.2f}"
        )
    print(
        "(per-block NEES target ~3; bounded = frac errmax < 1.5 m; diverged = frac > 5 m)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument(
        "--m2", action="store_true", help="run only the M2 floor falsification"
    )
    ap.add_argument(
        "--filters", action="store_true", help="run only the EKF-vs-UKF compare"
    )
    args = ap.parse_args()
    print(f"anchor anneal eval: {args.seeds} seeds, {args.steps} steps, moving leader")
    if args.m2:
        m2_consistency_sweep(args.seeds, args.steps)
        return
    if args.filters:
        filter_compare(args.seeds, args.steps)
        return
    consistency_audit(args.seeds, args.steps)
    anneal_discriminator(args.seeds, args.steps)


if __name__ == "__main__":
    main()
