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
9. **The carried/closed-loop quadrotor estimator is an open problem (#8 unresolved).** We built the
   balanced-cost + dual-control machinery (`src/rt_oac/{tracking_cost,balanced_cost,
   balance_constraints,dual_control}.py` — a velocity-damped standoff anchor, swappable combiners,
   a covariance/innovation-scheduled weight) to attack the #8 coupling, but **it does not robustly
   resolve it.** On a single seed a velocity-damped, scheduled hybrid can look bounded (err 0.46 m,
   NEES 14.3), yet **across 5 seeds it diverges on most** (NEES into the 100s–900s): the loop is
   seed-fragile, so only multi-seed numbers mean anything. Covariance scheduling **cannot** catch
   the failure — it is *confident-wrong* (P small while the true error is large), so a weight keyed
   on `trace(P)` does not back off in time, and a one-step innovation gate is too late. A three-way
   **trichotomy** (`report/trichotomy.py`) isolates the cause: **re-anchoring the EKF mean to
   truth** each step — the paper's validation (the companion's `x_op = x[i-1]`; only the covariance
   envelope propagates) — is robustly bounded (err ~0.4 m, NEES ~2) and confirms the OA trajectory
   IS observable; **carrying** the mean breaks consistency; **closing** the loop on the carried
   estimate is catastrophic (worst NEES ~940). The leader's *motion* is irrelevant (the relative
   dynamics are Galilean-invariant — proven). The planar unicycle does not show this (robustly
   observable). Plan: `docs/carried_estimation_plan.md`; write-up:
   `report/improving_beginners_oac.tex` (timing + soft constraint are the solid wins).
11. **[2026-06-04] #8 is a STARTUP TRANSIENT, localized to the position block — and tractable.**
    Two experiments (`experiments/anchor_anneal_eval.py`, 12 seeds, moving leader; built on the new
    annealed truth-anchor `ErrorStateEKF.update_anchor` + `anchor_alpha0/hold/anneal` example
    flags). **M0 (consistency):** per-block NEES (E=3) is fine when the mean is correct (re-anchored
    pos/rot/vel = 0.6/1.1/0.0) but the carried filter is catastrophically over-confident in
    **position** (118 / 5.8 / 0.6). **M1 (training-wheels discriminator):** anchoring the EKF mean
    to truth for the first ~20–40 steps and then **fully releasing** to carried gives **12/12 seeds
    bounded, 0 diverged, wheels-off NEES 5–7 (in the χ² band)** vs pure carried (4/12 bounded, 2
    diverged, NEES 122). So the carried loop is **locally stable around a converged estimate**; the
    divergence is the startup transient (aggressive observability-seeking on a not-yet-converged,
    position-over-confident estimate), not a steady-state instability. The truth anchor is a sim-only
    scaffold; the deployable fix (next) is to cross the same transient **without** truth — position-
    covariance consistency (M2) + a "soft start" that defers aggressive observability until the
    filter has converged, gauged by covariance/innovation (M3). Belief-space MPC (D) now likely
    unneeded. See `docs/carried_estimation_plan.md` §2 facts #6–#7, §6.
12. **[2026-06-04] The first-order EKF IS the over-confidence culprit — but consistency ≠ stability
    (the filter↔control coupling).** The position is observable only at HIGHER ORDER (STLOG r*≥5;
    a single range pins only the radial direction, leaving a non-Gaussian spherical-shell
    posterior). The first-order EKF's update manufactures tangential certainty the measurement
    does not provide. A **manifold UKF** (`src/rt_oac/unscented_kf.py`, derivative-free sigma points
    on R^3×SO(3)×R^3, subclassing the ESEKF; `filter=ukf`) was built and compared head-to-head
    (`experiments/anchor_anneal_eval.py --filters`, 16 seeds). **UKF fixes the consistency
    completely and robustly: position-block NEES 195 → 0.6** (E=3) on every seed — confirming the
    diagnosis. **But the UKF carried loop diverges WORSE (16/16 vs the EKF's 4/16):** an honest
    (large) P_pos makes the Kalman gain responsive, so the mean wanders in the unobservable
    tangential direction (errmax ~16 m), while the over-confident EKF was coincidentally stiff.
    **Lesson: a consistent covariance does not stabilize a loop whose controller plans on the MEAN
    (covariance-blind).** This explains why the finding-#9 dual-control failed — it keyed on the
    EKF's P, which *lied*. The honest UKF covariance is the missing **precondition** for
    covariance-aware control: the deployable path is **UKF + dual-control / soft-start** (back off
    observability while P_pos is genuinely large), with the gate re-scaled to the UKF's covariance
    magnitude (its P_pos ~400 vs the EKF's lying ~0.2). A Rao-Blackwellized PF would represent the
    shell exactly but is ~20× over the 100 ms budget (CPU) — overkill for a transient. Liveliness:
    `anchor_anneal_eval.py` now STREAMS per-seed heartbeats + an inactivity watchdog (after an
    88-min invisible hang from a self-matching `pgrep` wait-loop).
13. **[2026-06-04] The EKF's over-confidence is LOAD-BEARING; honest covariance alone is
    destabilizing — the schedule rescaling of #12 is not the lever.** Two decisive sweeps refute the
    "UKF + rescaled dual-control" path #12 proposed:
    (a) **UKF with the control on TRUTH** (`plan_on_truth`, carry the mean) is bounded and
    consistent on **16/16 seeds** (pos-NEES **0.1**, errmax 1.35 m), vs the EKF's 8/16 at NEES 40.8
    (`--filters --pot`). So the UKF is a *correct* filter — the carried divergence is **purely the
    controller acting on the estimate**, not any filter defect.
    (b) But swapping the UKF into the **deployable `mode=hybrid`** loop (the golden velocity-damped
    standoff anchor) **diverges across the entire schedule range** — `s0=1` (formation-only) 21.8 m,
    `s0=200` (back off harder) 15.8 m — while the EKF hybrid holds at **1.21 m**. Block-NEES stays
    in band throughout (pos 0.2–1.8): the UKF *honestly* reports the growing uncertainty, so it is
    "consistent" all the way out to 16 m. The mechanism: the formation anchor faithfully drives the
    *estimate* to the standoff; the EKF's stiff (frozen, over-confident) mean barely moves, so it
    pins the true position too, whereas the UKF's honest, responsive mean **wanders in the
    unobservable tangential subspace** and the controller tracks the phantom. The over-confidence is
    a load-bearing bug. `commit-don't-replan` (C, `replan_every`) does not rescue it either —
    `replan_every=4` on the UKF still diverges (10.5 m): a stale OA horizon executed open-loop on a
    wrong estimate just flies the follower out faster.
    **Implication (supersedes #12's path):** the tangential position is *structurally* unobservable
    from a single range until motion makes it observable (STLOG r*≥5), so no controller can pin it
    on the bare mean while un-localized. The deployable design must (i) keep the follower bounded by
    a **fixed / dithered excitation that does NOT track the estimate's absolute position** while
    un-localized, then (ii) hand off to formation-keeping once the estimate has *earned* trust —
    gauged by the **UKF's honest `trace(P_pos)`** crossing a threshold (the EKF can't gate this: it
    always claims localized). The UKF's value is this trustworthy convergence gauge, not a drop-in
    stabilizer. M3 is now precisely specified as this dither-until-localized → formation handoff.
    **(Update — see #14: the P_pos gauge this proposed is itself refuted; the UKF's P_pos never
    drops. Read #14 before acting on #13's design.)**
14. **[2026-06-04] De-risk that kills the P_pos gauge: under *perfect* excitation the tangential
    uncertainty does NOT shrink — there is no calibrated covariance to gate on.** Ran both filters as
    passive observers on the SAME truth-planned OA orbit (`plan_on_truth`, seed 0, the true error
    held at ~1.2 m the whole time) and logged 3σ_pos (`report=` dump, `sig` series):
    - **UKF 3σ_pos GROWS monotonically 2.8 → 4.6 → 8.6 → 15.3 → 23.7 m** while the true error stays
      1.2 m (end err/σ = 0.05, wildly *under*-confident). A single range is too weak to constrain the
      tangential direction recursively, so process noise inflates P_pos without bound — even though
      the STLOG Gramian is full-rank (instantaneously observable ≠ recursively informative).
    - **EKF 3σ_pos stays ~1.5 m** (end err/σ = 0.92) but NEES median 93 — it *manufactures* tangential
      certainty (over-confident through the run).
    So forcing a Gaussian onto the weakly-observable thin-ridge posterior has exactly two failure
    modes — EKF collapses it, UKF inflates it — and **neither is calibrated**. Consequences:
    (1) the M3 handoff gauge (#13: "wait for trace(P_pos) to cross low") is **dead** — UKF P_pos
    never drops, EKF P_pos is a lie; (2) naive covariance-aware/belief-space control is harder than
    thought (no trustworthy P to plan against — the UKF's is too pessimistic, the EKF's too
    optimistic); (3) what keeps the deployable EKF-hybrid bounded is the EKF's **stiff mean staying
    near truth** + the formation anchor, NOT genuine localization. A faithful posterior needs a
    non-Gaussian filter (PF) — but RBPF is ~20× over budget. Reproduce:
    `mode=estimation filter={ekf,ukf} plan_on_truth=true report=/tmp/x` then inspect `sig`.
15. **[2026-06-04] VERDICT — no deployable carried loop exists yet; the scheduled EKF-hybrid is
    seed-fragile.** The 16-seed sweep of the prior "golden" loop (`mode=hybrid filter=ekf`, the
    covariance+innovation dual-control schedule) gives errmax **median 3.6 m, only 19% bounded
    (<1.5 m), 50% limping [1.5, 5 m), 31% diverged (>5 m)** — seed 0's 1.21 m was luck. The UKF
    hybrid diverges on **100%** of seeds (median 28 m). This confirms finding #9 survives the
    dual-control schedule and closes the question the report deferred: the carried closed-loop OAC is
    **genuinely open**, and headlining only timing + soft constraint was correct.
    **Root-cause synthesis (#8→#15):** range-only relative localization is structurally weakly
    observable in the tangential subspace (observable only at high Lie order r*≥5; a single range
    pins radial only). The recursive posterior is a thin ridge no Gaussian filter calibrates — the
    EKF collapses it (overconfident, but its stiff mean coincidentally stays near truth, which is the
    *only* reason any seed is bounded), the UKF inflates it (P_pos grows unbounded). Closing the
    control loop around that miscalibrated mean diverges, and no scalar covariance→weight schedule
    fixes it. **Three genuinely-open paths (none is a quick knob):** (a) **robust-to-persistent-
    uncertainty control** — accept the tangential uncertainty never resolves and keep the follower
    bounded *given* a permanent tangential error set (tube/min-max MPC); the principled version of the
    EKF-hybrid's accidental stiffness; (b) a **non-Gaussian filter** (PF/RBPF) to represent the ridge,
    budget permitting; (c) **reformulate the OA objective** — the STLOG short-time Gramian is full-
    rank on the orbit yet the recursive filter does not converge (fact #14), so a metric targeting
    *recursive information accumulation* rather than instantaneous observability (ties to CLAUDE.md's
    "learn observability metrics from data"). Reproduce: `mode=hybrid filter={ekf,ukf} seed=0,..,15`.

## 4. Fidelity to the companion example & paper
- Structurally faithful to `examples/quadrotor_cooperative_navigation.py` (same model, leader,
  order/window/dt, constraint, **perfect-feedback receding horizon**). Deviations are our
  innovations: objective (log-det), solver (SLSQP@6), and a non-MUC start.
- Reproduces the paper's **qualitative orbiting** and is **faster than its reported solve time**;
  does **not** reproduce the paper's exact min-eig trajectory or quantitative RMSE/3σ (different
  objective; quadrotor estimation not yet validated to flight grade).

## 5. Open / future work
- **Carried estimation in the loop (#8)** — *open* (§7, finding #9). The balanced-cost +
  dual-control machinery is built but does not robustly resolve the coupling (seed-fragile; the
  trichotomy isolates the cause). The program to address it — estimator consistency first, then
  surprise-gated / commit-don't-replan / belief-space control, on multi-seed evaluation — is in
  `docs/carried_estimation_plan.md`.
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
  fn into the objective and solves box-bounded L-BFGS-B (a `soft=true` toggle on both examples). The
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
`spawn=true` for the live viewer or headless (default) for a `.rrd` + `.png`.

The quadrotor example is a **three-mode arc through #8**: default = open-loop soft-min orbit
(perfect feedback, the pretty orbit); `mode=estimation` = **estimation in the loop** (pure
observability on the ESEKF estimate) → *diverges* (error 2.93 m, NEES 260, distance escapes the
band to 6.5 m at 80 steps, seed 0); `mode=hybrid` = estimation in the loop + the §7 balanced cost
(velocity-damped standoff, covariance/innovation-scheduled) — an **exploratory, seed-fragile**
attempt that can look bounded on a lucky seed but does **not** robustly resolve #8 (finding #9).
The robust, non-diverging baseline is `mode=estimation reanchor=true` — the paper's validation,
which pins the EKF mean to truth each step. `soft=true` (any open-loop mode) folds the band into
the objective for an L-BFGS-B solve. The planar example is *inherently* estimation-in-the-loop (a
carried EKF, robustly observable here); its `hybrid=true` swaps log-det OAC for the balanced cost.
The leader's **motion is free** — the relative dynamics are Galilean-invariant (proven: a level
cruise at any speed gives the identical relative trajectory), so a moving leader is a pure
visualization choice (the closed-loop modes default to a 2 m/s cruise via `world_leader`).

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

## 7. The control/estimation coupling: balanced + dual-control attempts (open)
An attempt at finding #8 — *mechanically the same* as balancing tracking and
observability — that does **not** robustly resolve the carried coupling (finding #9; the
machinery and what it teaches are recorded here). The tracking worktree's balanced-cost machinery
is ported into rt-oac
(`src/rt_oac/tracking_cost.py`, `balanced_cost.py`, `balance_constraints.py`,
`dual_control.py`; `RTController.solve` now threads `p_ref`/`weight` as runtime args, compiled
once). A standoff/formation **anchor** competes with observability via swappable combiners
(`normalized` / `gradnorm` / soft `tube`; the min-eig `floor` is omitted — non-smooth/broken).
The trade-off weight can be **covariance-scheduled**: spend the observability budget only where
the estimator can keep up.

- **Quadrotor (`experiments/drone_coupling_eval.py`, the #8 testbed; 80 steps):** soft-min + tight
  band + estimate feedback. Pure observability destabilizes the ESEKF (NEES ~260, distance escapes
  the band to ~6.5 m). A **velocity-damped** standoff (relative position → standoff *and* relative
  velocity → 0) can look bounded on a single seed (error 0.46 m, NEES 14.3), and a position-only
  anchor is clearly worse — but **across seeds the hybrid is seed-fragile and diverges on most**
  (finding #9). It does not robustly resolve #8; see the trichotomy (`report/trichotomy.py`) and
  the plan (`docs/carried_estimation_plan.md`).
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
