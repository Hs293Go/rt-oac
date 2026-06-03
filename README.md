# rt-oac — Real-Time Observability-Aware Control

The **canonical "future work on OAC" repository**: solving the observability-aware control
(OAC) problem fast enough for **real-time, receding-horizon** use. It is a focused layer on
top of the companion repo `observability_aware_control` (reused verbatim as an editable
dependency — no fork of the STLOG / Lie-derivative / integrator / models / manifold), so the
companion stays the pristine, reproducible artifact for the JGCD paper while this repo hosts
the real-time method, smarter estimation, experiments, and findings.

**Two headline examples** mirror the companion's two, on the frontier method:
`examples/quadrotor_cooperative_navigation.py` and
`examples/simple_robot_cooperative_navigation.py`. Status & findings: **`PROGRESS.md`**
(+ `results/RESULTS.md`, `results/phase0_findings.md`, `results/report.tex`).

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

## Reuse, not copy

The verified math (STLOG, Lie derivatives, integrator, models, EKF) is reused verbatim
from the companion repo, declared as a **local editable dependency** via
`[tool.uv.sources]` in `pyproject.toml`:

```toml
[tool.uv.sources]
observability_aware_control = { path = "../observability_aware_control", editable = true }
```

The editable install exposes *both* companion packages (`observability_aware_control` and
`example_lib`), so they import normally — no `sys.path`/`PYTHONPATH` bootstrap needed. The
companion's heavy deps (`jax[cuda12]`, casadi, pyqt5, minsnap-trajectories) are excluded
via `[tool.uv] override-dependencies` (the modules we use don't import them), keeping the
install minimal and CPU-only. `import rt_oac` still enables JAX x64 and the persistent
compile cache; it also appends the companion `src` to `sys.path` as a fallback for an
*uninstalled* checkout (override with `OAC_SRC`).

## Running

```bash
uv sync                                        # installs rt-oac + companion (editable)
JAX_PLATFORMS=cpu uv run python benchmarks/profile_solve.py
uv run pytest                                  # tests
```
`source env.sh` is optional — it just exports `JAX_PLATFORMS=cpu`. The ruff-format
pre-commit hook reformats on the first commit attempt and fails; re-stage and re-commit.
