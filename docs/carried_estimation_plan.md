# Research plan — carried estimation in the loop (the finding-#8 coupling)

Status: **open problem.** This is the central robustness gap of real-time OAC and the gate to a
learned policy (a policy that imitates a diverging controller inherits the divergence). This plan
records what is known, what was ruled out, and a ranked program to address it.

## 1. The problem

A real receding-horizon OAC controller acts on a **carried** EKF estimate: the estimate `x_hat`
drifts from truth, accumulates, and feeds back into the next plan. On the range-only **quadrotor**
relative-pose model this loop is unstable — the aggressive observability-seeking maneuver,
computed at a wrong estimate, drives the true state somewhere uninformative, the estimate worsens,
and the relative-position error escapes past several metres with NEES in the hundreds.

The published method never exercises this. Its quadrotor estimator validation propagates the EKF
**covariance** along a fixed offline-optimized trajectory with the mean **re-anchored to truth
each step** (`observability_aware_control/examples/simple_robot_cooperative_navigation.py:147`,
`x_op = x[i-1]`). The mean cannot drift, so divergence is impossible by construction. The carried
closed loop is genuinely new.

## 2. What the diagnostic established (do not re-litigate)

A three-way comparison on the *same* pure-observability OA trajectory (moving leader, 80 steps, 5
seeds; `report/trichotomy.py`, reproduced in the report's future-work section):

| Setup | EKF mean | Control plans on | seeds in `χ²₉` band | typical NEES |
|---|---|---|---|---|
| **re-anchor** (paper) | pinned to truth | ≈truth | **5/5** | ~2 |
| **carry, plan on truth** | carried (drifts) | truth | ~2/5 | 6–90+ |
| **carry, closed loop (#8)** | carried | the estimate | **0/5** | 100s–900s |

Established facts:

1. **The leader's motion is irrelevant.** Speed-0 and speed-2 cruising leaders give
   byte-identical relative trajectories (Galilean invariance, verified). The moving-leader
   "absolute frame" question is a *visualization* matter, not a dynamics one; it does not affect
   the coupling. Do not conflate the two again.
2. **Loop-closing is the dominant destabilizer**, with carrying the mean a secondary one:
   re-anchor (5/5) ≫ carry-plan-on-truth (~2/5) ≫ closed loop (0/5).
3. **It is fragile, not biased.** Trajectories track closely then bifurcate late; a 6e-5 change in
   leader micro-inputs flips a seed from 0.46 m to 1.45 m. **All evaluation must be multi-seed**: a
   covariance schedule scoring 0.20 m / NEES 15 on seed 0 diverges on seeds 1–4 (NEES 80–590).
   Single-seed numbers are meaningless for this loop.
4. **Covariance scheduling cannot fix it.** The failure is *confident-wrong* (P small while true
   error is large), so a weight scheduled on `trace(P[pos])` does not back off in time. A one-step
   range-innovation gate (`nis0=4`) was also too late (1/5). Ruled out as a standalone fix.
5. **The planar unicycle does not have this problem.** Its lower-dimensional, range-only geometry
   is robustly observable, so its carried closed loop is stable (8-seed validated). The quadrotor
   relative-pose geometry (indirectly observed velocity, attitude coupling) is the hard case. Use
   the contrast as a controlled comparison.
6. **[M0, 2026-06-04] The inconsistency is localized to the POSITION block.** Per-sub-block NEES
   (`experiments/anchor_anneal_eval.py`, 12 seeds, E=3 each): re-anchored (mean correct) pos/rot/vel
   = 0.6 / 1.1 / 0.0 (consistent-to-conservative); carried = **118 / 5.8 / 0.6** — position is
   catastrophically over-confident, attitude mildly, velocity fine. H1 confirmed; the lever is the
   position-covariance scaling, not the whole filter.
7. **[M1, 2026-06-04] The divergence is a STARTUP TRANSIENT, not steady-state.** Anchoring the EKF
   mean to truth for the first ~20–40 steps via the annealed pseudo-measurement and then **fully
   releasing** to carried gives 12/12 seeds bounded, 0 diverged, wheels-off NEES 5–7 (inside the
   χ²₉ band) — vs pure carried 4/12 bounded, 2 diverged, NEES 122. Sharp release @20, @40, and a
   gradual wean all succeed. So the carried loop is **locally stable around a converged estimate**;
   #8 is triggered by aggressive observability-seeking on a not-yet-converged (position-overconfident)
   estimate. The truth anchor is a sim-only scaffold — the deployable task is to *cross the same
   transient without truth*.
8. **[2026-06-04] The EKF's over-confidence is LOAD-BEARING; an honest covariance alone is
   destabilizing.** (a) With the control on TRUTH (`plan_on_truth`), the **UKF** is bounded +
   consistent on 16/16 seeds (pos-NEES 0.1) vs the EKF's 8/16 — so the UKF is a *correct* filter and
   the carried divergence is purely the controller-on-estimate coupling. (b) But in the deployable
   `mode=hybrid` loop the UKF **diverges across the whole schedule** (`s0`=1→200: 21.8→15.8 m, block-
   NEES in band the whole way) while the EKF holds at 1.21 m: the formation anchor drives the
   *estimate* to the standoff, and the EKF's stiff/frozen mean coincidentally pins truth whereas the
   UKF's responsive mean wanders in the unobservable tangential subspace. `commit-don't-replan`
   (`replan_every=4`) does not rescue it (10.5 m). **Consequence:** honest covariance is necessary as
   a *convergence gauge*, not sufficient as a stabilizer; the #12 "rescale the dual-control gate"
   path is refuted, and M3 is respecified (dither-until-localized → formation handoff). Reproduce:
   `quadrotor_cooperative_navigation.py mode=hybrid filter={ekf,ukf} [mode.s0=200] [replan_every=4]`
   and `anchor_anneal_eval.py --filters --pot`.

## 3. Root-cause hypotheses (H1 confirmed by M0/M1)

- **H1 — estimator inconsistency (primary).** The ESEKF is likely **overconfident** in the weakly
  observable directions: P shrinks faster than the true error, so the controller trusts a bad
  estimate. Confident-wrong is the signature of inconsistency, not of bad control.
- **H2 — greedy re-planning amplifies estimate error.** Per-step re-optimization on the estimate
  is a positive feedback path (wrong estimate → "informative" but misdirected maneuver → worse
  estimate). Committing to a trajectory breaks the path (fact #2).
- **H3 — the objective is evaluated at the wrong state.** Both observability and any anchor are
  computed at `x_hat`; when `x_hat` is wrong, the "optimal" excitation is actively harmful.

## 4. Program, ranked by expected leverage

### A. Fix the estimator first (tests H1)
A consistent filter is a precondition for trusting the estimate in any loop. Before any control
change:
- Diagnose consistency on the **re-anchored** trajectory (where the mean is correct): is NEES
  biased low (over-confident) in the velocity/tangential subspace? Plot per-direction
  normalized error.
- Evaluate **observability-constrained EKF** (no spurious information gain in unobservable
  directions) and **process-noise inflation** / adaptive Q. Target: NEES in band on the
  *carried, plan-on-truth* setup (fact #2 says the trajectory is fine; the filter must be too).
- Deliverable: a carried estimator that is consistent under truth-planned control across ≥20
  seeds. This alone may lift carry-plan-on-truth from ~2/5 to robust.

### B. Surprise-gated control (tests H1/H3; complements A)
Gate observability-seeking on **estimator surprise**, not claimed uncertainty:
- A **windowed** normalized-innovation-squared (NIS) statistic or a **CUSUM** change detector with
  memory, not the one-step gate that proved too late (fact #4). Freeze to formation-keeping while
  the detector is tripped; resume when it clears.
- Couple to A: surprise gating buys time for a consistent filter to recover.

### C. Commit, don't re-plan greedily (tests H2)
- Plan an OA trajectory over a horizon and **execute it open-loop between infrequent re-plans**
  (move-blocking / shrinking-horizon), rather than re-solving every step on the estimate. Fact #2
  predicts a large gain. Sweep the re-plan period.

### D. Robust / belief-space MPC (principled, higher cost)
- **Belief-space planning**: plan on the full covariance (proper dual control), optimizing the
  *expected posterior* uncertainty rather than a scalar weight — the real "plan on the belief."
- **Tube / min-max MPC**: bound the control's reliance on the estimate over an uncertainty set so
  the plan is robust to estimate error by construction.

### E. State-independent excitation (fallback)
- If greedy observability is irrecoverably fragile, superimpose a small **persistent-excitation
  dither** (state-independent) on formation-keeping. It cannot be misdirected by a wrong estimate,
  trading optimality for robustness. Quantify the observability/robustness trade vs. A–D.

## 5. Evaluation protocol (non-negotiable)

- **≥ 20 seeds.** Report the **fraction bounded** and the **NEES distribution** (median, worst,
  in-band %), never a single seed (fact #3).
- Sweep **initial offset**, **noise level**, **horizon length** (the 80→120-step drift is real).
- Baselines: the trichotomy of §2 (re-anchor = upper bound; closed loop = current).
- Controlled comparison against the **planar** case (fact #5) — a fix should not regress it.
- Operating point: `r=5, N=20, T=0.2 s, dt=0.05 s`, ESEKF, moving leader. Run env
  `JAX_PLATFORMS=cpu uv run`.

## 6. Sequencing (revised by the M0/M1 verdict)

M0/M1 done (facts #6, #7): the inconsistency is in the position block, and the divergence is a
**startup transient** — crossing it (truth anchor for ~20–40 steps) yields a robustly stable
carried loop. This sharply narrows the program: the goal is to **cross the same transient without
truth**, not to fight a steady-state instability (so D is likely unnecessary).

1. ✅ **M0 consistency audit** — position-block over-confidence (fact #6).
2. ✅ **M1 training-wheels discriminator** — transient verdict (fact #7).
3. **M2 — position-covariance consistency (A).** Make the carried filter stop being confident-wrong
   in position during the transient: process-noise inflation on the position block / adaptive Q /
   observability-constrained update. Validate per-block NEES in band under `plan_on_truth` (≥20
   seeds). This directly attacks the cause M0 found.
4. **M3 — deployable handoff, RESPECIFIED by fact #8 (the load-bearing-overconfidence result).**
   The original "start formation-dominant, ramp observability" plan is **refuted**: with the honest
   UKF the *formation anchor itself* tracks the wandering mean and diverges across the whole schedule
   range (`s0`=1→200, 21.8→15.8 m), while the over-confident EKF's stiff mean coincidentally holds
   (1.21 m). The tangential position is structurally unobservable on the bare mean until motion makes
   it observable, so the controller must NOT drive the follower on the estimate's absolute position
   while un-localized. Respecified design: **(i) bounded/dithered excitation that does not track the
   estimate's absolute position** (E, fixed body-frame pattern) until **(ii)** the **UKF's honest
   `trace(P_pos)`** crosses a localized threshold, then **hand off to formation-keeping** on the
   now-trustworthy estimate. The UKF is the convergence *gauge* (the EKF always claims localized and
   cannot gate this), not a drop-in stabilizer. Target ≥95% of ≥20 seeds bounded, wheels off.
5. **D** (belief-space / robust MPC) only if M2+M3 plateau. The fact-#8 result (a mean-tracking
   controller is destabilized by an *honest* covariance) raises D's likelihood vs the earlier
   transient-only reading: planning on the covariance (not the bare mean) may be the principled cure
   if the M3 dither-handoff proves brittle.
6. **Learning curriculum** — well-enabled: the across-rollout truth-anchor anneal reliably produces
   non-divergent rollouts (fact #7), the expert/data a learned policy needs.

## 7. Success criterion

A **carried, closed-loop** quadrotor OAC loop that is bounded and NEES-consistent on **≥ 95% of
≥ 20 seeds** with a substantially moving leader, at the validated horizon — without re-anchoring to
truth. Partial credit: a characterized, monotone trade between observability aggressiveness and
robustness (the A–E Pareto) that a mission planner can dial.
