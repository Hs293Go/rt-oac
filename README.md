# rt-oac — Real-Time Observability-Aware Control

The **canonical "future work on OAC" repository**: solving the observability-aware control
(OAC) problem fast enough for **real-time, receding-horizon** use. It is a focused layer on
top of the companion repo `observability_aware_control` (reused verbatim as an editable
dependency — no fork of the STLOG / Lie-derivative / integrator / models / manifold), so the
companion stays the pristine, reproducible artifact for the JGCD paper while this repo hosts
the real-time method, smarter estimation, experiments, and findings.

**Two headline examples** mirror the companion's two, on the frontier method:
`examples/quadrotor_cooperative_navigation.py` and
`examples/simple_robot_cooperative_navigation.py`. Both stream a live **rerun**
visualization (3D/2D scene + trajectories + per-tick observability, plan-time, and
inter-vehicle-distance time-series) and render a comprehensive multi-panel matplotlib
figure afterwards. Config is **Hydra** (`examples/conf/`): the quadrotor's `mode` group is a
pedagogical arc through finding #8 (`mode=open|estimation|hybrid`); other knobs are overrides
(`soft=true`, `spawn=true` for the live viewer, `steps=...`). Headless (default) writes a
`.rrd` recording (open later with `rerun results/example_*.rrd`) plus `results/example_*.png`.
Status & findings: **`PROGRESS.md`** (+ `results/RESULTS.md`, `results/phase0_findings.md`,
`results/report.tex`).

Today the companion's Observability-Predictive Controller (OPC) runs only as an *offline*
trajectory generator in flight experiments because of its computational cost.

## The bottleneck (measured)

In the companion repo's saved 2399-solve quadrotor run, **every** solve hits the
`maxiter=40` cap (`nit ≡ 40`, never reaching `gtol`), at ~0.525 s/solve. STLOG evaluation
itself is ~68 µs and is already JIT-compiled. **So the bottleneck is the optimizer's
iteration budget, not Gramian evaluation.** Levers that cut iterations — warm-starting
(currently discarded each step), fused value+grad, 2nd-order info, smoother objective,
learned initial guess — are the highest-ROI path.

## Target

**10 Hz guidance on CPU (Jetson TX2 NX): ≤ 100 ms/solve** (≈5× the current 0.525 s).
A faster solver must keep **closed-loop EKF estimation accuracy non-inferior** to the
offline-converged OPC — that is the decisive acceptance criterion, not raw speed.

## Layout

```
src/rt_oac/
  controller.py     # fork of the companion controller: fused value_and_grad, warm-start, hessp
  metrics.py        # smooth surrogate objectives (logdet, soft-min, sum-k-smallest, ...)
  warmstart.py      # shift-append previous solution; learned-init hook
  transcription.py  # (Phase 3) multiple-shooting / collocation
  solvers/          # scipy path; (stretch) jax-native SQP
  learned/          # (Phase 4) warm-start net, surrogate metric net
benchmarks/profile_solve.py            # Phase 0 attribution harness (the gate)
experiments/                           # trimmed, headless forks of the companion examples
config/                                # scenario + ablation-matrix configs
```

## Vendored core

The verified math (STLOG, Lie derivatives, integrator, models, EKF) is **vendored
verbatim** from the companion repo into `src/`, as the `observability_aware_control` and
`example_lib` packages alongside `rt_oac`. They import normally — no editable dependency,
no `sys.path`/`PYTHONPATH` bootstrap. rt-oac declares only the deps the vendored modules
actually import (`jax`, `equinox`, `scipy`, `numpy`, `matplotlib`, `rerun`); the
companion's heavy deps (`jax[cuda12]`, casadi, pyqt5, minsnap-trajectories) are never
pulled in, keeping the install minimal and CPU-only. The vendored trees are excluded from
ruff so they stay byte-for-byte auditable against the source.

`import rt_oac` enables JAX x64 and the persistent compile cache. `COMPANION_SRC`
(override with `OAC_SRC`) optionally points at a sibling companion checkout, used only to
find reference-data files — not code.

## Running

```bash
uv sync                                        # installs rt-oac (core is vendored under src/)
JAX_PLATFORMS=cpu uv run python benchmarks/profile_solve.py
uv run pytest                                  # tests
```
`source env.sh` is optional — it just exports `JAX_PLATFORMS=cpu`. The ruff-format
pre-commit hook reformats on the first commit attempt and fails; re-stage and re-commit.
