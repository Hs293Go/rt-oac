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
- `src/rt_oac/{tracking_cost,balanced_cost,balance_constraints,dual_control}.py` — the
  balanced-cost machinery for the coupling exercise (§7): standoff/tracking term, combiners,
  soft tube, covariance-scheduled weight.
- `examples/` — the two rerun-instrumented headline examples (§6): quadrotor (soft-min orbit +
  level-cruise leader, perfect feedback) and planar (OAC vs no-OAC with a carried EKF).
- `benchmarks/` — profiling/diagnostics (the gate, old-vs-new, method×problem matrix) plus the
  orbit band/objective sweep, the objective solver-profile + eval microbench, the leader-speed
  sweep, the coupling Pareto sweep, and the penalty-vs-hard-constraint solver probe.
- `experiments/` — closed-loop rollouts, planar/drone EKF eval, orbit exploration, and the
  control/estimation-coupling evals (`planar_coupling_eval.py`, `drone_coupling_eval.py`).

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
9. **Coupling resolved by a standoff anchor (+ dual control)** — see §7. Mechanically reusing
   the tracking worktree's balanced-cost machinery (ported to `src/rt_oac/{tracking_cost,
   balanced_cost,balance_constraints,dual_control}.py`), a Euclidean standoff anchor competes
   with observability. On the quadrotor #8 testbed (soft-min + tight band, 80 steps) pure
   observability destabilizes the ESEKF (NEES ~260, distance escapes the band to ~6.5 m); a
   **velocity-damped** standoff (relative position → standoff AND relative velocity → 0) restores
   a bounded, consistent loop (normalized: error 0.46 m, **NEES 14.3 in the χ² 95% band [2.7, 19]**,
   distance 1.60 m), and the **covariance-scheduled** weight keeps NEES lowest (11.8). A
   *position-only* anchor cannot arrest the relative-velocity drift and still diverges — velocity
   damping is essential. **Partial, not permanent:** over longer horizons a slow residual drift
   develops (by ~120 steps the error/NEES roughly double), so this is a bounded, near-consistent
   loop at the validated horizon, not a proven fixed point — full long-horizon stability is open.

## 4. Fidelity to the companion example & paper
- Structurally faithful to `examples/quadrotor_cooperative_navigation.py` (same model, leader,
  order/window/dt, constraint, **perfect-feedback receding horizon**). Deviations are our
  innovations: objective (log-det), solver (SLSQP@6), and a non-MUC start.
- Reproduces the paper's **qualitative orbiting** and is **faster than its reported solve time**;
  does **not** reproduce the paper's exact min-eig trajectory or quantitative RMSE/3σ (different
  objective; quadrotor estimation not yet validated to flight grade).

## 5. Open / future work
- **Hybrid tracking + observability objective** — *addressed* (§7, finding #9): a velocity-damped
  standoff anchor balanced with observability resolves the finding-#8 divergence.
- **Dual / uncertainty-aware control** — *partly addressed* (§7): a covariance→weight schedule
  helps; the remaining step is true belief-space planning (plan on the full covariance) and full
  NEES consistency in the most aggressive regime.
- **Quadrotor estimation to flight grade** — validate ESEKF on the real target; 2 followers,
  full 120 s, world-frame trajectories.
- **JAX-native solver / learned warm-start** — push under 100 ms with margin / reach 50 Hz.
  *De-risked* (`benchmarks/penalty_solver_probe.py`): baking the distance band into the cost as a
  smooth one-sided penalty and dropping to a box-bounded **unconstrained L-BFGS-B** solve is
  **~1.6–4.0× faster** than the hard-constrained SLSQP (penalty 32 ms at w=1, 52 ms at w=10, 80 ms
  at w=100 vs 128 ms hard) with **zero band violation** — at a modest observability cost (the
  penalty keeps the follower off the bound that log-det rides; higher w is tighter/feasibler but
  slower). No active-set machinery → fully JIT-able, the foundation for a JAX-native solver.
  Now a first-class option: `RTController(constraint_mode="soft")` folds the *existing* constraint
  fn into the objective and solves box-bounded L-BFGS-B (a `--soft` toggle on both examples). The
  branching stayed clean — encapsulated in the controller, the examples pass one param — so soft was
  **not** promoted to the default. Finding: soft is faster and feasible for the open-loop orbit
  (~85 vs ~115 ms) and the planar case, but the *delicate quadrotor hybrid #8-loop destabilizes
  under it* (it was validated only with the hard constraint), so hard stays the default there.
- **Sim-to-real ladder** — PX4 SITL → HIL on Jetson → flight (controller fixed, swap plant/sensors).

## 6. Headline examples & live visualization
The two headline examples are now the canonical demo — the standalone `experiments/headline_sim.py`
was **retired** in favor of them. Both stream a live **rerun** scene (3D/2D trajectories, body
frames, covariance ellipse, range) plus per-tick time-series (observability, plan time, distance,
estimation error) and render a comprehensive multi-panel matplotlib figure afterwards. Run
`--spawn` for the live viewer or headless (default) for a `.rrd` + `.png`.

A `--hybrid` toggle activates the tracking/observability hybridization (the §7 golden form —
NormalizedWeightedSum with a velocity-damped standoff anchor) **and the closed-loop estimator**:
the quadrotor switches from the open-loop soft-min orbit to the ESEKF estimate-feedback loop
that resolves #8 (bounded, estimator near-consistent — NEES in the χ² band at ~80 steps, ~22 just
above it by 120 as the slow drift sets in); the planar swaps its log-det OAC for the balanced
cost. The hybrid quadrotor runs against the as-validated (default) leader — the
level-cruise override is open-loop only, since under estimate feedback a fast leader destabilizes
the standoff-tracking loop.

Three tuning nuances, each backed by a benchmark:
- **Quadrotor objective/band → soft-min `[1, 2] m`** (`benchmarks/quad_constraint_sweep.py`).
  Log-det (D-optimality) maximizes information *volume* and fills a 3D shell; **soft-min
  (E-optimality)** fixes the *worst* direction and settles into a clean planar orbit. A narrow band
  bracketing the start radius makes the motion a constant-radius ring rather than a
  baseline-maximizing arc. (The planar example keeps log-det.)
- **Soft-min is ~18% faster than log-det — but not because it is cheaper to evaluate.** Identical
  iteration/eval counts (nit 6, nfev 7) and identical value+grad cost (3.44 vs 3.48 ms/call,
  `benchmarks/objective_eval_microbench.py`). The gap is **constraint activity**
  (`benchmarks/objective_solver_profile.py`): log-det pins the follower to the max-distance bound
  (active inequality → costlier SLSQP QP each iteration), while soft-min orbits the band interior
  (constraints slack). Log-det with the tight `[1, 2]` band actually *diverges*.
- **Leader speed is free** (`benchmarks/leader_speed_sweep.py`). The paper's leader is a min-snap
  10 m / 120 s hop (~0.16 m/s) — only ~0.16 m over the 15 s window, visually frozen. The relative
  dynamics are **Galilean-invariant**, so a level constant-velocity leader at 1/2/3 m/s gives an
  *identical* relative orbit, observability, and solve time, only stretching the world-frame scene
  into a helix. The example uses a level **2 m/s** cruise (~30 m / 15 s).

## 7. Resolving the control/estimation coupling (balanced + dual-control OAC)
The exercise that addresses finding #8, *mechanically the same* as balancing tracking and
observability. The tracking worktree's balanced-cost machinery is ported into rt-oac
(`src/rt_oac/tracking_cost.py`, `balanced_cost.py`, `balance_constraints.py`,
`dual_control.py`; `RTController.solve` now threads `p_ref`/`weight` as runtime args, compiled
once). A standoff/formation **anchor** competes with observability via swappable combiners
(`normalized` / `gradnorm` / soft `tube`; the min-eig `floor` is omitted — non-smooth/broken).
The trade-off weight can be **covariance-scheduled**: spend the observability budget only where
the estimator can keep up.

- **Quadrotor (`experiments/drone_coupling_eval.py`, the #8 testbed; 80 steps):** soft-min + tight
  band + estimate feedback. Pure observability destabilizes the ESEKF (NEES ~260, distance escapes
  the band to ~6.5 m). A **velocity-damped** standoff (relative position → standoff *and* relative
  velocity → 0) keeps the loop bounded and consistent (normalized: error 0.46 m, NEES 14.3 in the
  χ² band, distance 1.60 m); the **scheduled** weight keeps NEES lowest (11.8). A position-only
  anchor still diverges — damping the relative velocity is essential. The resolution is *partial*
  (a slow residual drift develops over longer horizons), not a proven fixed point.
- **Planar (`experiments/planar_coupling_eval.py`, fast dev):** observability is the *stabilizer*
  (the follower must maneuver to localize), so the schedule is reversed (up-weight observability
  when uncertain = active perception). The schemes trace a clean estimation-vs-formation Pareto
  (`benchmarks/coupling_pareto_sweep.py`): normalized λ≈1 is the knee, `gradnorm` over-weights the
  steep observability gradient and is dominated here.
- **Open:** full NEES consistency in the most aggressive regime, and belief-space (covariance)
  planning rather than a scalar covariance→weight schedule.

## 8. Repository strategy (recommendation)
**Promote this repo as the canonical "future work on OAC" repo** (two headline examples
mirroring the companion's two: quadrotor + planar, on the frontier method). Keep the companion
repo pristine/reproducible for the paper. **Selectively upstream** only the small, additive,
general-purpose pieces to the companion when its WIP settles — `neg_logdet`/`neg_softmin` as
extra `gramian_metric` options (non-default) and `ErrorStateEKF` as a new file — and add a
one-line pointer in the companion README to this repo. The controller fork, early-stop, and
experiments stay here (they are the *departure* from the published method).
