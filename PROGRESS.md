# RT-OAC — Progress Report (real-time observability-aware control)

This repo is the **focused "future work on OAC" layer** spun off per the JGCD paper's
`CLAUDE.md` mandate. It reuses the companion repo `observability_aware_control` *verbatim*
(editable dependency — no fork of the STLOG / Lie-derivative / integrator / models /
manifold), and adds the real-time machinery, smarter estimation, experiments, and findings.
The companion repo remains the single source of truth for the math and the published paper
results; this repo is where the real-time / future-work exploration lives.

Detailed write-ups: `results/RESULTS.md`, `results/phase0_findings.md`, `results/report.tex`.

## 1. The problem & plan
Make the observability-aware control (OAC) problem solvable in **real time** (the paper runs
the OPC offline in flight). Approved plan: profile first (gate), then objective reframing,
lean solver, structure, learning, validated by closed-loop EKF accuracy. Target: 10 Hz / ≤100
ms on CPU. Plan file: `~/.claude/plans/create-a-plan-to-stateful-wind.md`.

## 2. What's built
- `src/rt_oac/controller.py` — `RTController`: pluggable lean solver (SLSQP default), warm-start,
  early-stop, jitted runtime-arg interface (no per-step recompile).
- `src/rt_oac/metrics.py` — smooth surrogate objectives (`neg_logdet`, `neg_softmin_eig`, …),
  injected via `ObservabilityCost(gramian_metric=…)` (no core fork).
- `src/rt_oac/warmstart.py` — shift-append warm start.
- `src/rt_oac/error_state_ekf.py` — manifold-aware error-state EKF (9×9 tangent covariance,
  `boxplus`/`Log` residuals) — fixes the quaternion divergence.
- `src/rt_oac/scenario.py` — single source of truth for the quadrotor problem + controller factory.
- `benchmarks/` — profiling/diagnostics (the gate, old-vs-new, method×problem matrix).
- `experiments/` — closed-loop rollouts, planar EKF eval, drone EKF eval, orbit exploration.

## 3. Key findings
1. **Bottleneck is the optimizer, not the STLOG.** Per solve: ~6% in observability compute,
   ~94% in scipy `trust-constr` internals. Accelerating the STLOG is a dead end.
2. **The paper's per-node min-eigenvalue objective is numerically degenerate** (floored by the
   short-time `T^(2r*+1)=T^11` scaling → ~0 gradient). **`log-det` restores the gradient** and
   drives full observability. This is the central reframing.
3. **Frontier method = log-det + SLSQP + early-stop@6.** Quadrotor solve **~10 s → ~100 ms**
   maintaining full observability; *faster than the paper's reported ~0.557 s sim solve*. The
   2×2 gate shows the efficacy gain is the **objective** (30–80×), not early-stopping; speed is
   the **solver** (~11× leaner/iter) + early-stop. Early-stop is "free" (observability plateaus
   by ~6 iters), not "giving up."
4. **The "0.5 s vs 10 s" puzzle is resolved:** the fast historical number used a *different,
   easier model era* (stacked dynamics, order 1, loose tolerances) — bisected to the Aug-2025
   switch to the relative-pose model. Current OG code (trust-constr+min-eig, order 5) is
   genuinely ~10 s; not a recoverable regression.
5. **Planar EKF validation:** carried-estimate closed loop — OAC cuts follower-position final
   error ~13× and covariance ~20× vs no-OAC, and **early-stop@6 preserves the full benefit**.
6. **Error-state EKF fixes the quaternion divergence:** vs the Euclidean placeholder (diverged
   to ~7 m, overconfident), the ESEKF stays consistent (error within 3σ, NEES in band on gentle
   maneuvers, unit quaternion to 1e-15).
7. **Clear orbits** emerge with a tight distance band + long horizon: soft-min gives a clean
   12-revolution planar orbit (perfect feedback). The distance constraint is essential and
   cheap (~+25%), and it *bounds* the behavior (loose bound → fly-away).
8. **Negative result — control/estimation coupling:** the *tight, aggressive* orbit is unstable
   under **estimate feedback** (diverges to 100s–1000s of m; worse with a better initial
   estimate → fundamental, not transient). The observability-greedy objective destabilizes the
   very estimator it relies on. The **gentle** config (ESEKF + [1,3] m + log-det) is the stable
   realistic closed loop.

## 4. Fidelity to the companion example & paper
- Structurally faithful to `examples/quadrotor_cooperative_navigation.py` (same model, leader,
  order/window/dt, constraint, **perfect-feedback receding horizon**). Deviations are our
  innovations: objective (log-det), solver (SLSQP@6), and a non-MUC start.
- Reproduces the paper's **qualitative orbiting** and is **faster than its reported solve time**;
  does **not** reproduce the paper's exact min-eig trajectory or quantitative RMSE/3σ (different
  objective; quadrotor estimation not yet validated to flight grade).

## 5. Open / future work
- **Hybrid tracking + observability objective** — required: pure observability orbits/wanders and
  destabilizes the estimate-feedback loop (finding #8).
- **Dual / uncertainty-aware control** — plan on the covariance so maneuvers don't outrun the
  estimator (the coupling instability).
- **Quadrotor estimation to flight grade** — validate ESEKF on the real target; 2 followers,
  full 120 s, world-frame trajectories.
- **JAX-native solver / learned warm-start** — push under 100 ms with margin / reach 50 Hz.
- **Sim-to-real ladder** — PX4 SITL → HIL on Jetson → flight (controller fixed, swap plant/sensors).

## 6. Repository strategy (recommendation)
**Promote this repo as the canonical "future work on OAC" repo** (two headline examples
mirroring the companion's two: quadrotor + planar, on the frontier method). Keep the companion
repo pristine/reproducible for the paper. **Selectively upstream** only the small, additive,
general-purpose pieces to the companion when its WIP settles — `neg_logdet`/`neg_softmin` as
extra `gramian_metric` options (non-default) and `ErrorStateEKF` as a new file — and add a
one-line pointer in the companion README to this repo. The controller fork, early-stop, and
experiments stay here (they are the *departure* from the published method).
