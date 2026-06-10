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
    **GUARDRAILS for the clean-branch quad climb distilled from this post-mortem (so we don't repeat the
    class of mistake): `docs/quad_climb_guardrails.md`** — the class of mistake, what the ladder does and
    does NOT prove (the flat-output/bridge "wins" escape the regime where the CRLB floor bites; the bridge
    is open-loop tracking + a reduced post-hoc EKF, not a carried-quad loop), G1–G16, and a measured
    go/no-go (currently **NO-GO** on closing the OA loop on a full carried quad estimate). Two adversarial
    critics' code-verified catches are folded in (single-seed headline, 5 m "stable" bar vs 1.5 m,
    NEES `median(neess[20:])` blind to the startup transient, ref-velocity recovery EKF).
16. **[2026-06-04] Phase A (objective reformulation, chosen path c): the PCRB diagnostic locates the
    real bottleneck — recursive position localizability is comparable to the formation scale.**
    `experiments/fim_diagnostic.py` reconstructs, along the truth-planned OA orbit, the per-node STLOG
    (the optimizer's metric), the trajectory-accumulated first-order Gramian `W_o = Σ Φ(k,0)ᵀHᵀR⁻¹HΦ`
    (what the recursive filter actually accumulates), and the **posterior CRLB** (the Riccati at
    truth with the real Q/R — the floor for ANY recursive filter). Reusing the ESEKF's own F/H/G
    linearization; the example now dumps `xrel_full`/`u_applied` to feed it. Findings (seed 0):
    - **Per-node STLOG min-eig ≈ 1e-12** — FLOORED by the `T^(2r*+1)` short-time scaling (r*=5), so the
      worst direction is numerically singular per node. The log-det/softmin objective therefore
      optimizes observability *volume*, which loads onto the already-easy attitude/radial directions.
    - **`W_o` is ill-conditioned (cond 1.7e6); its worst-informed directions are tangential POSITION +
      VELOCITY** (λ≈40 vs the attitude/radial λ≈1e7). Tangential info *does* accumulate (λ_min ×96 over
      the run) but stays ~6 orders below the easy directions.
    - **The decisive number — recursive PCRB 3σ_pos ≈ 3.6 m (≈1.2 m std), vs the ~1 m standoff.** The
      best *any* recursive filter can localize the follower tangentially is about the size of the
      formation itself, so the carried controller never has a tight-enough estimate → it wanders. The
      batch CRLB (x₀, no process noise) is 0.25 m — the info exists but process noise erodes it
      recursively. The UKF's 23.7 m is ~6× *over* the PCRB floor (over-conservative extraction); the
      EKF's claimed ~0.5 m is *below* the floor (impossible ⇒ overconfident, NEES 93 — confirmed).
    **Conclusion:** the STLOG objective fails to drive down the *recursive tangential-position PCRB* —
    the quantity the control needs. Reformulation target (Phase B): an objective that minimizes the
    recursive tangential PCRB / maximizes the worst-direction accumulated information *rate against
    process noise*, not the per-node Gramian volume. Open caveat for Phase C: confirm a different orbit
    can actually push the PCRB below the formation scale (i.e. the OA orbit is not already PCRB-optimal).
    Reproduce: `examples/...quadrotor... mode=estimation filter=ukf plan_on_truth=true report=/tmp/derisk`
    then `experiments/fim_diagnostic.py --npz /tmp/derisk/quad_estimation_planontruth.npz`.
17. **[2026-06-04] Phase B headroom de-risk: PCRB-targeted orbit shaping IS worth it — ~2.3–2.9×
    tighter localization while holding formation.** `experiments/pcrb_optimize.py` is a self-contained
    differentiable optimizer: decision vars = the follower's N×4 input sequence (leader fixed),
    forward = integrate the relative dynamics → run the EKF Riccati at truth (F/G/H reused from the
    ESEKF) → per-step trace(P[pos]); objective = mean steady-state trace(P[pos]) + a soft
    standoff-band penalty (so it can't cheat by flying out to a huge parallax). JAX value_and_grad +
    scipy L-BFGS-B. Result (N=30): the recursive PCRB 3σ_pos drops **3.6 m (STLOG) → 1.61 m** with the
    band HELD (dist [0.43, 1.20], violation ~3e-4) — ~2.3× tighter than the STLOG orbit, 2.9× tighter
    than the naive reference. So the answer to the #16 caveat is YES: the STLOG-volume objective leaves
    real recursive localizability on the table; a PCRB-aware objective recovers it. Note the optimizer
    drives to the *inner* band edge (tighter standoff = larger parallax-per-range = more observable) —
    an explicit observability↔formation trade-off. Caveats/next: (i) 1.6 m 3σ is still ~0.5 m std (~half
    the standoff) — better than 3.6 m but maybe not yet tight enough to bound the carried loop (Phase C
    decides); (ii) the Riccati-in-the-objective is nested autodiff (2nd-order) — a real-time concern for
    wiring into RTController, so Phase B should also try the cheaper accumulated-`W_o` min-eig surrogate
    (no per-step inverse). Reproduce: `experiments/pcrb_optimize.py --steps 30 --iters 80`.
18. **[2026-06-04] Exhaustive PCRB-optimizer sweep (5 parallel investigations + 3 adversarial
    verifiers + synthesis; `report/figures/pcrb_sweep.png`). The reformulation helps a lot and is
    robust — but a structural ceiling above the standoff means it likely cannot close the gap alone.**
    The generalized `experiments/pcrb_optimize.py` (objective forms / band / horizon / leader /
    p0-scale / restarts, JSON dump) was swept:
    - **Floor: ~1.4–1.6 m (from STLOG 3.6 m / planar warm-start 5.6 m), a robust ~3–4× gain.**
      Counter-intuitively the *absolute* floor is LOWEST at the SHORTEST horizon (N=20→1.23 m, N=80→
      2.05 m); long horizons only *look* better via `tighter_x` because the base orbit rots faster. And
      **iterations, not horizon, are the binding limit** (N=40: 1.67→1.58→1.44 m at 80/150/300 iters,
      no plateau).
    - **Best objective: `maxeig` or `trace` on the recursive EKF covariance (tie ~1.56–1.58 m, band
      held).** `posvel` is fractionally lower and best-on-velocity (3.4 m) but dips below the standoff
      floor (a real pos/vel/band three-way trade). **The batch-`W_o` objectives are VERIFIED-POOR
      (4.4–4.6 m, ~2.3–3.3× worse, break the band)** — i.e. minimizing observability *volume* (the
      STLOG-style target) is the wrong objective for recursive localizability. Confirmed non-tautologically
      (logdet-vs-wo_mineig, different reducer than the yardstick).
    - **Pareto: monotone, ≈0.47 m of localization error per metre of standoff backoff.** The strong
      "inner-edge / closer-is-better" claim **FAILED verification** — a close band ([0.3,0.5]) can't be
      held and forcing it makes PCRB *worse*. The real lever is band **width/slack**, not center: a wide
      band [0.3,2.0] is the global best (1.44 m) because slack lets the optimizer dip toward the inner
      edge voluntarily while staying feasible. **Wire a soft, WIDE band — not a tight clamp.**
    - **Robust, not overfit:** opt 3σ_pos = 1.53–1.66 m (~8% spread) across p0-scale 0.3–10×, seeds,
      and — notably — **leader maneuver gives ZERO improvement** (1.578 vs 1.580 m). The floor is a
      geometry property.
    - **Mechanism:** the optimizer maximizes **out-of-plane LOS rotation** (5–9× more swept angle, via a
      vertical weave/spiral, not a faster horizontal circle) **plus range-scale diversity** — which are
      *complementary*: an over-tight band swept the *most* LOS (9.5 rad) yet localized *worst* (1.81 m)
      by starving range diversity.
    - **THE STRUCTURAL CEILING (biggest open question):** even the best orbit (~1.4 m 3σ_pos) stays
      *above* the ~1 m standoff, and leader maneuvering can't help. A single-scalar-range-per-step,
      hover-leader geometry appears unable to localize the follower tighter than the formation itself.
      So objective reformulation is necessary and ~3–4× helpful but probably **not sufficient alone** to
      bound the carried loop. **Phase C must confront this:** (a) break the one-scalar-per-step bottleneck
      (a second range anchor / heterogeneous measurement), and/or (b) accept ~1.4 m and design the
      control to tolerate position uncertainty ~the standoff (ties back to the robust-control path).
    Adversarial verdicts: headroom-is-real **SURVIVED** (strict band wband=500/2000 still 1.65–1.74 m,
    viol ≤1e-6; the win is position-only — velocity does not improve); recursive-beats-batch
    **SURVIVED**; strong inner-edge **FAILED**. Reproduce via the sweep workflow / the tool's `--dump`.
19. **[2026-06-04] REFRAME (per user): the ESTIMATOR is the primary design variable to make OAC/STLOG
    pay off — and a barometer + the UKF does it.** "First-order objectives won" only because they were
    scored against a first-order EKF; keep the STLOG observability objective and design the estimator.
    `experiments/estimator_ladder.py` carries EKF/IEKF/UKF/bootstrap-PF from a wrong prior along a fixed
    orbit (Monte-Carlo), reporting true position error split radial / tangential / vertical-z.
    - **Estimator ladder, STLOG orbit, baro OFF (10 seeds):** the **UKF (2nd-order) recovers the
      tangential (high-order) direction** — RMS 0.38 m vs the EKF's 2.2 m — but its radial blows up
      (5.3 m). Root cause is finding #14's P_pos inflation: for h=‖r‖² the UKF predicts
      ‖μ‖²+tr(P_pos), so an over-inflated P pulls ‖μ‖ inward (radial bias). The IEKF (Gauss-Newton MAP,
      `ErrorStateEKF.update_iterated`) does NOT fix it (over-corrects under the wide prior, 3.07 vs EKF
      2.49 m). So the high-order tangential observability IS recoverable; no Gaussian filter yet gets
      BOTH radial and tangential.
    - **Barometer on/off is decisive (`--baro`, relative-altitude channel via a residual mixin →
      BaroEKF/BaroUKF; companion model untouched).** Adding a baro (σ=0.2 m): (i) crushes the vertical
      error (EKF z 1.5→0.5, UKF z 2.6→0.5 m), and (ii) **CURES the UKF's radial bias (5.3→0.67 m)** —
      observing a weak direction bounds P_pos, so the tr(P_pos) Jensen bias collapses. **Result: with a
      barometer the UKF is the BEST estimator (pos 1.18 m) and recovers BOTH radial (0.67) AND
      tangential (0.98) AND z (0.50).** The thesis realized: keep the STLOG orbit, add a cheap baro, use
      the 2nd-order UKF, and the high-order observability finally pays off (UKF tangential 0.98 < EKF
      1.55). The barometer is a *synergistic* sensor, not a complication. (The bootstrap PF degrades
      under the tighter 6-d likelihood — underpowered, weight-collapse — not a fair gold standard; the
      RBPF marginalizing the observed directions is the right PF.) Caveats: 10 seeds, one orbit; the
      decisive next test is the **carried CLOSED loop with baro+UKF** (does it beat the seed-fragile
      19%?), plus a baro-σ sweep and the spectrum {range / range+baro / 2-range}. Reproduce:
      `experiments/estimator_ladder.py --npz <stlog dump> [--baro]`.
20. **[2026-06-04] FIRM-UP of #19 (5 probes + 3 adversarial verifiers; `report/figures/baro_ladder.png`).
    The baro+UKF result is real and the mechanism is rock-solid — but the win is *modest-margin* and
    *geometry-conditional*, and the closed-loop GO comes with a tight envelope.** Tooling upgraded:
    per-seed mean±std (significance), `mean_trP_pos` logged (inflation gauge), `--baro-std/--p0-scale/
    --baro-bias` knobs.
    - **Significance (40 seeds, STLOG): firm on substance, modest margin.** Baro-off the UKF tangential
      0.42 m (std 0.19, ~11× std below EKF 2.39) — high-order recovery unambiguous — while its radial
      blows up (5.5 m, tr(P_pos) 94.5). Baro-on the UKF is the best estimator (pos **1.01 m**, std 0.30)
      recovering radial 0.49 + tangential 0.87 + z 0.51, tr(P_pos) 0.79. **But** the UKF↔EKF pos gap
      (0.72 m) only ≈ the EKF's std (0.72) — bands overlap; report as "best, modest margin," not
      decisively separated (the UKF's *own* spread is tight, so it's 2.4× the UKF std).
    - **Mechanism CONFIRMED (Spearman 1.0).** UKF radial tracks tr(P_pos) monotonically (OFF 94.7/5.39 m
      → std0.2 0.78/0.49 m); the **EKF tr(P_pos) never inflates** (control — the ‖r‖²-Jensen term is
      carried only by the sigma-point UKF). So **tr(P_pos) is a reliable online early-warning gauge**.
    - **Baro-σ requirement: solid cure ≤ 0.5 m std, headline ≤ 0.2 m; knee ~1 m; useless > ~2 m.**
      Degradation is GRACEFUL (trP rolls 0.68→11.7 with noise) — the ~95× inflation appears only as a
      *discontinuous jump* when the channel is fully removed. A realistic baro (≤0.5 m) works.
    - **Robustness: bad prior FULLY robust** (10×-wrong prior → pos 1.18 m, recovers both; PF diverges).
      **Biased baro tolerable to ~0.2 m**, advantage gone by 0.5 m, broken at 1.0 m bias.
    - **CROSS-ORBIT = GEOMETRY-DEPENDENT (the key caveat).** Wins on the STLOG and PCRB-optimized
      (observability-designed) orbits, but **FAILS on the naive warm-start orbit**: the baro fixes z
      (→0.12 m) yet leaves radial (1.63→1.65) and tr(P_pos) untouched. The baro→radial cure only
      translates when the radial-unobservable direction is **geometrically coupled to altitude**; on a
      near-static side-by-side geometry (horizontal LOS) it is decoupled and the cure does nothing.
    **GO for the closed-loop test, tight envelope:** STLOG/PCRB-style geometry, baro std ≤ 0.2 m, bias
    < 0.2 m. **Biggest risk = geometry coupling:** the closed-loop controller continuously reshapes the
    geometry, so it can steer into radial-⊥-altitude configurations that silently re-open the P_pos
    inflation the open-loop test never sees. **Mitigation: monitor tr(P_pos) online** (Spearman-1.0
    early warning) and gate/penalize geometries where it climbs. Reproduce: the firm-up workflow / the
    ladder's `--seeds/--baro-std/--p0-scale/--baro-bias` knobs; figure generator
    `report/figures/make_baro_ladder.py`.

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

**`examples/flat_robot_cooperative_navigation.py` — the bottom-up range-only ladder's frontier (rung
3a; `conf/flat_robot.yaml`).** A 3D FLAT-OUTPUT `[x,y,z,psi]` leader-follower (the quad's flat outputs;
world-frame velocity kinematics), **range-only** cooperative localization (leader pose + both yaws
measured, follower position only via the inter-robot range), carried EKF, the rung-2 stable recipe
(softmin-eig + bounded standoff + symmetry-break). Same rerun + matplotlib instrumentation as the other
two. Headline result: from a 2.26 m initial error in the *unobservable* (tangential) direction, OA
drives the carried-EKF follower-position error **2.32 → 0.26 m (~9x; 20-seed median ~17x)** while no-OAC
(fly straight) stays stuck, at ~3 ms/solve. This is the validated, beneficial+stable range-only OAC the
quad lacked (#32-35) — and `experiments/flatout_bridge.py` shows the OA flat-output trajectory is
quad-realizable (the benefit transfers ~6x through the geometric tracker, gated by trackability). The
consolidated ladder (planar 12x → 3D 13x → flat-output 17x → quad bridge 6x, all range-only) is
`report/figures/range_only_ladder.png`; see §9.

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

## 9. Bottom-up OAC ladder (CURRENT DIRECTION — 2026-06-05)
**Strategy.** A separate investigation (preserved on the `quad-baro-estimation` branch) exhaustively
searched the quadrotor relative-pose system for a regime where observability-aware control *pays off* and
found **none**: its sensors (range + DIRECT relative attitude + barometer) observe every limiting
direction *passively*, so observability-seeking is redundant where the loop is stable and impossible
where it is not (strip a sensor → the gap is structural, no trajectory recovers it). OAC pays off only
when observability is **controllable AND marginal** (motion-dependent), which that rich-sensor system
never is. **So: bottom-up — start from the simplest model where OAC IS beneficial, climb, carrying the
design principles up.** (Full quad negative result + the baro+UKF/two-leader estimation tuning live on
the `quad-baro-estimation` branch; this branch keeps only the clean ladder, fewer confoundable knobs.)

**Rung 1 — planar unicycle, range-only** (`experiments/planar_oac_validate.py`). Leader-follower where
the leader pose is measured but the FOLLOWER's position is observable only via the inter-robot range, so
its tangential (cross-range) direction is motion-observable only — the controllable-marginal regime. OAC
drives the unobservable initial error down where nothing else can: on a TANGENTIAL 2.26 m initial error,
final follower-position error noac **2.20 m** (NEES 105, overconfident-wrong) → OAC **0.19 m** (NEES 6) =
**11.8×**, 100% stable, ~3 ms/solve. Airtight via 4 controls: (i) directional sweep — benefit is largest
on the tangential, smallest on the radial (range sees it anyway) = mechanism; (ii) a dumb sinusoidal
weave gives **1.0×** — arbitrary motion fails, only the observability-AWARE optimization finds the
maneuver that excites the unobservable mode; (iii) `oac_tight` — **11.9× at FIXED distance**, so it is the
maneuver geometry, not getting closer; (iv) NEES-consistent + 100% stable across 20 seeds.

**Rung 2 — 3D point-mass, range-only** (`experiments/point3d_oac_validate.py`). 6-state single-integrator
point pair, follower 3D position via range only — now TWO unobservable tangential directions. Beneficial
OAC **survives the dimensionality jump**, and the climb surfaced **three transferable design discoveries**
(none visible at rung 1): (1) the symmetric straight-line config is a **DEGENERATE CRITICAL POINT** of the
observability objective (gradient ~0, the optimizer can't move, OAC≡noac) → perturb the first guess to
break symmetry; (2) **log-det DRIVES AWAY** — maximizing observability *volume* is baseline-maximizing, so
on a few noise seeds the follower runs out and the estimate diverges (the SAME instability that sank the
quad); (3) **softmin-eig** (E-optimality, worst-direction) is the **stable metric** — it stops once the
worst axis is observed instead of spreading. **Stable + beneficial recipe = softmin-eig + bounded
formation distance + symmetry-break: 100% stable across all directions, 7× (radial) / 13× / 15× (the two
tangentials) lower error, NEES ~3–7 (vs noac 84–105), formation held at ~5 m.**

**Rung 3a — FLAT-OUTPUT OA + the quad-realizability bridge** (`experiments/flatout_oac_validate.py`,
`experiments/flatout_bridge.py`). Reframed per the lead (differential flatness, Mellinger-Kumar): a
quad's flat outputs are `sigma=[x,y,z,psi]`; any smooth `sigma(t)` is dynamically feasible. So the
principled OA-planning model is the FLAT-OUTPUT kinematics — the smallest state that is the quad's
genuine planning space AND guaranteed realizable by the subservient (geometric) tracker. That is rung 2
+ a yaw state, and it's a *cleaner* rung 3 than full quad dynamics. **3a-core:** flat-output `[x,y,z,psi]`
OA (world-frame velocity so psi is yaw-decoupled; omni range so psi is OA-INERT, a free flat output;
yaws measured) reproduces rung 2 exactly — 20 seeds: tang 17–26×, radial 5.6×, 95–100% stable, recipe
(softmin-eig + bounded dist + symmetry-break) carries. **3a-bridge (the milestone):** track the OA
flat-output reference on the 10-state `quadrotor` via `TrackingController`+`AttitudeController`. **The OA
benefit TRANSFERS to the quad** — a range-only EKF on the quad's ACTUAL trajectory recovers the
unobservable error **up to 6× lower than no-OAC** (0.33 vs 1.97 m). **KEY NEW COUPLING the flat-output
view exposes:** OA-aggressiveness ↔ trackability. The naive velocity-bounded OA plan is physically
aggressive (peak **22 m/s² ≈ 2.3g** accel, **4.6g jerk** between steps — a quad does ~1g lateral), so it
tracks imperfectly (~1 m); a 10× faster inner loop helps (1.74→0.99 m), and a *gentler* plan (scaled
velocity bounds) tracks cleanly (~0.4 m) but excites less → smaller benefit (2.3× vs 6×). So the premise
holds **conditionally**: OA flat-output trajectories are quad-realizable *as long as the planner
respects feasibility* (bound accel/jerk / smoothness) — a real OA↔control coupling absent at the
point-mass rungs, and exactly where the rung-2 recipe (tame the drive-away) pays off again.
(A body-fixed BEARING sensor would make `psi` an active OA lever, but bearing is a CONFOUND vs the
range-only JGCD paper — and has been stripped from the OG repo — so it is OFF the table; in range-only,
`psi` is correctly a free flat output and the only OA lever is translation.)

**3a-trackability RESOLVED — the derivative-bounded planner is DOMINATED + the gap is not the blocker
(`experiments/flatout_di_oac.py`).** Tested the natural fix — plan one integration order higher, a
DOUBLE-INTEGRATOR flat-output model with bounded ACCELERATION input (+ optional `--jerk` penalty) — so
the trajectory is smooth/feasible by construction with a clean accel feedforward. It FAILS as an
improvement, on an 8-seed bridge head-to-head (recovery vs no-OAC ~2.0 m): (1) per-component accel-bounds
still hit 1.4 g and make the plan *jerkier* (the optimizer shifts aggressiveness to the next derivative,
jerk max 88.7 > 45.7); (2) a within-window jerk penalty doesn't constrain the *applied* receding-horizon
trajectory; (3) at the SAME ~0.6 m quad track error, simply **gentling** the velocity-bounded plan
(`--vscale 0.7`) gives **4.2×** vs the DI's **2.7×** — scaling the velocity bound preserves the
observability-effective maneuver GEOMETRY, re-planning in accel space loses it; (4) it's moot: the
**aggressive** velocity-bounded plan transfers **6.3×** *despite* ~1 m (max 2.4 m) track error — the OA
benefit comes from the quad **maneuvering vigorously, not tracking precisely**, so it's robust to track
error. So OA-vs-trackability is a **fundamental Pareto traced by the velocity-bound scale** (aggressive
6.3×@0.99 m → gentle 4.2×@0.62 m); adding integrator order doesn't beat it, and the "trackability gap" is
a formation-keeping cost during the maneuver, NOT a blocker on the localization benefit. **Next:** climb
to the full quad with the SAME range-only sensing, picking a point on this Pareto by how much
formation-keeping fidelity the mission needs.

**Ladder so far (CONSOLIDATED, all RANGE-ONLY; figure `report/figures/range_only_ladder.png`):** planar
11.8× (rung 1) → 3D point-mass 7–15× @100% (rung 2) → flat-output `[x,y,z,psi]` 17–26× + quad-realizable,
6× benefit transfers (rung 3a). Each rung: from a 2.26 m TANGENTIAL (motion-observable-only) initial
error, no-OAC stays stuck ~2.2 m while OA recovers it ~12–17×; the recipe (softmin-eig + bounded standoff
+ symmetry-break) carries up. **Headline demo:** `examples/flat_robot_cooperative_navigation.py` (rerun +
matplotlib, OA vs no-OAC, the polished rung-3a face; the planar headline `simple_robot_cooperative_
navigation.py` is rung 1). Reproduce: `experiments/planar_oac_validate.py --seeds 20`;
`experiments/point3d_oac_validate.py --seeds 20 --metric neg_softmin_eig`; `experiments/flatout_oac_validate.py
--seeds 20`; `experiments/flatout_oac_validate.py --dump /tmp/oa.npz && experiments/flatout_bridge.py
--ref /tmp/oa.npz`; figure `experiments/plot_range_only_ladder.py`.

**Aside — does YAW matter? (the original quad's range+attitude obs; `experiments/freeze_yaw_test.py`,
adversarially verified workflow `wj0rog3ox`).** Theory: yaw is rotation about the thrust axis, and the
squared range + its time-derivatives are yaw-invariant (the `cross(r,ω)` transport terms cancel:
`d²|r|²/dt² = 2|v|² + 2r·(R(q)t_l − t_f)`, both attitude terms yaw-invariant for a level/hover config).
So for a pure range world yaw is a symmetry → inert (which is exactly what the world-frame flat-output
model encodes). **But empirically the original quad OA commands HEAVY yaw** (RMS ~4, saturating the
bound) — because the original ALSO observes the relative *attitude* `q_fl` (yaw-driven) and uses
body-frame relative coords. **Freeze-yaw test verdict (after 3 wrong guesses — inert → dominant →
counterproductive — all corrected by verification):** the ONLY robust conclusion is that the **STLOG
objective is nearly FLAT in yaw** (frozen vs free <0.2% at every horizon) — the optimizer is
near-indifferent to yaw, so the heavy commanded yaw is a near-null direction of the cost. Whether yaw
helps the recursive *position* CRLB is **CONFOUNDED, not answerable** from freeze-vs-free: freezing yaw
makes the optimizer re-plan a *different orbit* (pins the standoff to the 1 m min-distance bound vs 3 m
→ closer = better range² SNR), and the position ratio *reverses* with horizon (0.46×@40 → 0.57×@80 →
**1.96×@120**, frozen worse); the same-orbit counterfactual (zero yaw at fixed inputs) flips it again
(P_pos↑1.33×, standoff→10.8 m). So **yaw is objective-flat but dynamically entangled** — its direct
estimation effect isn't separable from the standoff/orbit. **Lesson:** OA control DOFs are coupled
(yaw↔roll/pitch↔standoff); clean attribution needs a fixed-standoff counterfactual. **Closing the yaw
question:** in a strictly RANGE-ONLY world (the JGCD paper), yaw is a symmetry of the objective → inert,
full stop; the flat-output model is right to make `psi` a free output. The only way to make yaw matter
is a yaw-sensitive (bearing/FOV) sensor — but that is a CONFOUND vs the range-only paper and has been
stripped from the OG repo, so it is OFF the table. (Originally mis-scoped as "rung 3b"; retracted.)
Reproduce: `experiments/freeze_yaw_test.py --steps {40,80,120}`.
