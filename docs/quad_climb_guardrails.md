# Quad-climb guardrails — don't repeat the carried-loop divergence

**Status (2026-06-06): NO-GO on closing the OA loop on a full carried quad estimate.** This doc is the
governing constraint for climbing from the flat-output rung toward the full quadrotor. It is the product
of a forensic re-reading of findings #8–#15 + `carried_estimation_plan.md`, an audit of what the current
ladder *actually* does, and two adversarial critics — whose code-verified catches are folded in below.

---

## 1. The class of mistake (what the full-quad-first attempt did wrong)

Generalizable, repeatable errors — each defensible from the forensics:

1. **Jumped to the hardest closed-loop case first.** Full 10-state quaternion-manifold relative pose
   (indirectly-observed velocity + attitude coupling) — and a rich-sensor regime with *no regime where
   OA pays off* (passive sensing already observes the limiting directions). OA deployed where it is
   neither controllable nor marginal. *(#5/#13, #9)*
2. **Closed the OA loop on an overconfident carried estimate.** The controller planned aggressive
   maneuvers on the bare EKF *mean*, which was **confident-wrong**: P small while true error large
   (NEES 93; claimed ~0.5 m sits **below** the recursive CRLB floor ≈ 3.6 m — information-theoretically
   impossible). Wrong-estimate maneuver → uninformative true state → worse estimate → escape to 3–6 m,
   worst NEES ~940. *(trichotomy, #8/#9, #16)*
3. **Confused estimator consistency with loop stability.** The manifold UKF *fixed* consistency
   (pos-block NEES 195→0.6, 16/16) yet the carried loop diverged **worse** (16/16 vs EKF 4/16): an honest
   large P made the gain responsive, the mean *wandered* in the unobservable tangential direction, and the
   covariance-blind controller tracked the phantom. **Consistency ≠ stability.** *(#12/#13)*
4. **Attacked the symptom with scalar schedules.** Every covariance→weight / innovation-gate keyed on a
   gauge that lies (EKF P overconfident, never backs off; UKF P_pos *grows* 2.8→23.7 m under perfect
   excitation). One-step gate too late. **No calibrated covariance exists to schedule or gate on.** *(#4/#5/#7)*
5. **Mistook truth-anchored success for deployability.** Re-anchor (5/5, NEES ~2) and anchor-then-release
   (12/12) pin the mean to truth → divergence impossible *by construction*. Sim-only scaffolds; they prove
   the orbit is observable, not that the loop is stable. *(#1/#2)*
6. **Declared victory on a transient and on single seeds.** The failure is *fragile, not biased*: a 6e-5
   leader-input change flips a seed 0.46↔1.45 m; a schedule at 0.20 m/NEES 15 on seed 0 diverges on seeds
   1–4 (NEES 80–590). *(#3, #15)*
7. **Optimized the wrong observability quantity.** Maximized *instantaneous* STLOG Gramian volume
   (full-rank on the orbit) instead of *recursive* information accumulation against process noise.
   Instantaneously observable ≠ recursively informative. *(#6/#7, #14, #18)*

---

## 2. What the current ladder actually proves — and what it does NOT (honest, code-verified)

**It avoids the trap by side-stepping the hard part, not by solving it.** Both halves are true.

**Genuinely a carried loop (the honest part):** rungs 1 and 3a (`simple_robot_…`, `flat_robot_…`,
`flatout_oac_validate.py`) **do** re-plan OA each step on the running estimate `x_hat` while truth evolves
independently and error accumulates (`res = ctrl.solve(x_hat, guess)`). Stable + beneficial from a 2.26 m
tangential error.

**But the differences from the divergent quad are load-bearing — the critics' corrections:**

- **NOT "the same range-only structure."** The flat-output observation measures **leader position (3) AND
  both yaws DIRECTLY** (`flatout_oac_validate.py:72-75`). It shares only the *follower-position-via-range*
  sub-block. The surrounding observation model, the **process model**, and the **noise regime** are
  strictly easier.
- **The stability is a MODELING ASSUMPTION, not a controller property.** The flat-output process model is
  **exactly linear** (`xdot = u`, `flat_robot:89-90`) with low input noise (`INPUT_VAR=1e-2`) and **no
  velocity/attitude blocks**. "No overconfidence" follows from zero process-model error, not from anything
  the controller does. The quad's first added block (velocity) reintroduces a forced/nonlinear process
  model and the overconfidence mechanism returns. Velocity is observed by range only through its integral,
  at **higher Lie order** than position → *more* weakly observable; the forensic pos/rot/vel NEES of
  118/5.8/**0.6** means the 0.6 is a **sleeping trap** (never excited), not a clean block.
- **The wins escape the regime where the floor bites.** Flat-output 0.26 m and bridge 0.33 m both lie
  **below the quad's ~3.6 m recursive CRLB floor** — possible only because they leave the regime that
  produced divergence. **The flat-output recursive floor is never computed** anywhere in the repo, so
  "~9×/17×" is *not* evidence the floor obstruction is survivable.
- **The bridge is NOT a carried-quad loop.** `flatout_bridge.py` loads a **frozen offline** OA reference
  and *tracks* it on the quad via a deterministic geometric cascade — **no `ctrl.solve`/STLOG anywhere**.
  Its EKF is a **separate 3-state relative-position filter** (`_rel_dyn`, P0=diag([2,2,2]), **no
  velocity/attitude states**) run **post-hoc**, predicting with the **reference** velocity
  (`u = ref_vel[i] - leader_vel[i]`, line 108) — so tracking error is **invisible** to it and it
  **structurally cannot exhibit** the failure mode it is used to rule out. "Robust to ~1 m track error" is
  partly a harness artifact (clean reference velocity fed to the filter).

**The two-axis picture (the precise way to see the gap).** The ladder advances along two *independent*
axes, and they have only ever been advanced *separately*:
- **Axis A — closed-loop hardness:** plan-on-truth / full-ahead-of-time open-loop (*already solved*) →
  **carried estimate + receding-horizon OAC** (controller re-plans each step on `x_hat`, truth evolves
  separately). Every ladder rung (1 planar, 2 point-mass, 3a flat-output) is carried + RHOAC — verified:
  `ctrl.solve(x_hat, …)` each step with `x_hat = x_true + offset`, `x_true` advanced independently.
- **Axis B — model fidelity:** kinematic flat-output `[x,y,z,psi]` (`xdot=u`) → the **10-state quad**.
  The bridge pushes Axis B to the real quad, but **predominantly by varying the model** — it tracks a
  *frozen* OA reference (no `ctrl.solve`), i.e. it operates in the *solved* open-loop / full-ahead regime.

So: **carried + RHOAC is proven only at low model fidelity (A high, B low); the quad model is reached only
in the solved open-loop regime (B high, A low).** The bridge is not deficient — being open-loop is *by
design*, since full-ahead planning is already solved; its role is the model bridge + a benefit measurement.

**Demonstrated:** carried + RHOAC is stable+beneficial at *kinematic* fidelity; the OA benefit *transfers*
to the quad model *in open-loop* (measured by a reduced post-hoc filter). **NOT demonstrated — the
diagonal corner:** carried + RHOAC **at quad-model fidelity** — re-planning OA each step on a carried
estimate that carries the quad's nonlinear velocity/attitude dynamics. That corner is exactly where the
old full-quad-first approach *started* and diverged; it is the only place the divergence can reappear, and
no rung occupies it yet (→ **G7**).

---

## 3. Guardrails for the climb

- **G1 — No skipping rungs.** Climb fidelity one block at a time: flat-output `[x,y,z,psi]` → **add a
  velocity block to the carried estimate** → add the attitude/manifold block last. Never jump to the
  10-state manifold ESEKF in the loop.
- **G2 — NEES + CRLB-floor gate (reject overconfident filters).** Before a rung may close the loop, its
  *passive* filter on the truth-planned orbit must pass per-block NEES (E=3, incl. the position block)
  in-band **and** reported 3σ_pos **≥ the recursive CRLB floor** (Riccati-at-truth with real Q/R). Claiming
  below-floor certainty = auto-reject.
- **G3 — ≥20 seeds, fraction-bounded, transient-included.** Report fraction <1.5 m / [1.5,5) limping /
  >5 m diverged, **errmax median**, over the **full** rollout (incl. startup). Never a best seed; never a
  median that hides a diverged tail.
- **G4 — No truth-anchor in the shipped loop.** Re-anchor / anchor-then-release are sim-only scaffolds
  (divergence impossible by construction); use only to confirm an orbit is observable.
- **G5 — Attack the root cause, not the symptom.** No standalone scalar covariance→weight schedule or
  one-step gate. Permitted structural moves: tighten the *estimate* (2nd range anchor / heterogeneous
  sensor — geometry-dependent, needs online tr(P_pos) gating); belief-space / non-Gaussian filter so the
  controller sees the thin-ridge posterior; **keep OA where the filter is well-conditioned (flat-output)
  and track on the quad**; reformulate to the recursive tangential PCRB (~3–4×, necessary-not-sufficient);
  robust/tube MPC tolerating a permanent tangential error set.
- **G6 — Covariance-aware control before in-loop OA.** The lethal mechanism is a covariance-blind
  controller planning on the mean; no rung closes OA on a carried estimate until the controller consumes a
  *calibrated* covariance. (Consistency alone is insufficient — the UKF was consistent and diverged worse.)
- **G7 — Build the diagonal-corner rung.** The two axes (carried+RHOAC, model fidelity) have only been
  advanced separately; build the rung that occupies **both at once** — carried-estimate re-planning (Axis A)
  on a model carrying the quad's nonlinear velocity/attitude dynamics (Axis B). Equivalently: close the
  bridge's loop (re-plan OA on the carried quad estimate instead of tracking a frozen reference). This is
  the only regime where the divergence can reappear; reach it deliberately and instrumented, early — not
  for the first time at the top of the climb. (Plan-on-truth and full-ahead open-loop are already solved
  and need no further rungs.)
- **G8 — The harness must implement G3.** `flatout_oac_validate.py` currently scores "stable" at `<5 m`
  and NEES as `median(neess[20:])` — it **cannot detect the forensic failure signature** (median 3.6 m,
  19% bounded, 31% diverged). Fix the metric before trusting any quad-rung pass.
- **G9 — Faithful recovery measurement.** Measure recovery/consistency with a filter of the **same
  dimension and process-model fidelity** as the in-loop estimator, predicting with the **achieved/estimated**
  velocity (not the reference command). Replace the bridge's 3-state, ref-velocity `ekf_recover` before any
  transfer claim is admissible.
- **G10 — OA-aggressiveness ramped on filter convergence.** Do not plan aggressive maneuvers (or fire the
  symmetry-breaking kick) while the estimate is un-converged; the positive-feedback loop is armed at step 0.
- **G11 — Geometry-degeneracy constraint.** Constrain the controller out of near-collinear / co-altitude
  configurations where the radial-unobservable direction decouples and sensor fixes fail — the closed-loop
  OA objective is *incentivized to seek that edge*. A passive tr(P_pos) warning is not enough (the gauge lies).
- **G12 — Multi-direction + multi-seed transfer.** The bridge dump is **seed 0, tang_horiz only**. Show
  transfer across radial / tang_vert and ≥20 seeds before "the benefit transfers" is a claim, not an anecdote.
- **G13 — Pre-registered thresholds + fragility rule.** Fix seed count and bound threshold before the run;
  primary endpoint = diverged-*tail* fraction (not median); a single diverged seed in the 6e-5 fragility
  neighborhood blocks GO.
- **G14 — Implement the CRLB floor in code.** Go-condition 2 is currently *prose*. Implement the
  Riccati-at-truth PCRB on the candidate orbit and have the go/no-go rule actually read it, in the **same
  standoff configuration the rung flies** (the forensic 3.6 m is at a ~1 m standoff; the dump uses a ~5 m
  standoff — not apples-to-apples).
- **G15 — Feasibility gate.** The OA reference must lie within the vehicle's real accel/jerk envelope
  (a quad does ~1 g lateral; the 6.3× plan rides 2.3 g / 4.6 g) before its benefit is counted.
- **G16 — Abandon criterion.** If the recursive CRLB floor on the orbit cannot be pushed below the standoff
  with credible sensing, **declare the carried-quad OA loop unachievable and stop** — the flat-output-plan +
  track approach is then the *endpoint*, not a stepping stone. Define this before spending more budget.

---

## 4. Go / No-Go decision rule

**GO — attempt to close the OA loop on the full carried quad estimate — only if ALL hold:**
1. Filter **calibrated, not just consistent** (per-block NEES in-band on every seed **and** 3σ_pos ≥ CRLB floor). *(G2/G14)*
2. **CRLB floor below the standoff** on the candidate orbit (today ~3.6 m **above** the ~1 m standoff — a structural ceiling that leader maneuvering does **not** improve: 1.578 vs 1.580 m). *(#18)*
3. Control law is **covariance-aware** (belief-space or robust/tube MPC). *(G6)*
4. **Carried-on-truth passes first** (control on truth, mean carried; ≥20 seeds bounded+consistent) — necessary-not-sufficient (the UKF passed this and still failed the closed loop). *(#13)*
5. **Carried-closed-loop** clears a pre-registered bound bar over ≥20 seeds (target: essentially all bounded to steady state, errmax median below standoff) — far above today's 19% bounded / 31% diverged. *(#15)*

**NO-GO — keep OA at the flat-output level, track on the quad — if ANY GO condition fails.** This is the
situation **today** (conditions 1 and 2 fail): plan OA in flat-output space (~9–17×), transfer via the
geometric tracker (~6×), measure with a *faithful* post-hoc filter (after G9). That is the current answer.

---

## 5. The honest open risk

- **The CRLB ceiling may never drop below the standoff with one scalar range/step.** Reformulation buys
  ~3–4× only; leader maneuvering buys *zero*. If we can't break the single-range bottleneck (credible 2nd
  range anchor / complementary sensor that survives adverse geometry), **no in-loop carried-quad OAC loop
  is safe regardless of filter or controller** — and flat-output-plan + track is the endpoint (→ G16).
- **The cure opens another failure, geometry-dependently, under adversarial control.** The baro+2nd-order
  UKF that cures radial inflation (5.3→0.67 m) *fails* on near-static side-by-side geometry — and a
  closed-loop controller can **steer itself into** exactly that geometry. Evaluate every sensor/filter fix
  **under the closed-loop controller**, not on truth-planned orbits; assume the controller probes the worst
  geometry (→ G11).
- **The ladder defers the hard problem to its last rung.** Every current "win" escapes the regime where the
  floor bites; the velocity/attitude blocks + real Q/R are met only at the final rung, with maximal sunk
  cost pressuring a GO. G7 (build the carried+nonlinear rung early) exists to de-risk this deliberately.

---

## 6. Heterogeneous CLKF × OAC sub-rungs — VERIFIED (2026-06-06)

The "carried + RHOAC at quad fidelity" corner (G7) decomposes into two independent fidelity axes —
**estimator (CLKF)** and **OAC** — two rungs each. The estimator's high-fidelity rung is driven by
**measured IMU specific-force + gyro** (strapdown INS, range-aided), not the thrust/rate command (feeding
the command into a real estimator is rare and risky — and is exactly the command-driven path that fed the
#8–#15 divergence; verified: `error_state_ekf.py:210` integrates the dynamics under `u`, and `u =
plan_u[commit_k]` is the OA command, `quadrotor_…:753`).

|              | **O0 kinematic flat-output STLOG** | **O1 full-quad STLOG** |
|--------------|------------------------------------|------------------------|
| **E0 command/kinematic** | DONE — the flat-output rung (stable) | THE OLD DIVERGENCE (#8–#15) — do NOT rebuild |
| **E1 IMU-driven range-aided full ESEKF** | **build first** — isolate the estimator | the corner — only if both single-axis rungs pass |

**Verified payoff (Lie-rank + CRLB computation, two adversarial critics) — the IMU is a process-noise /
honest-covariance lever, NOT a structural-observability lever:**
- **Structure collapse to flat-output: FAILS.** The IMU is proprioceptive — zero leader information, does
  not observe relative position. Lie-rank (unknown moving leader) = **7/21 observable @ order 6, 14 nulls**
  (all leader states + both bias triads + INS couplings). Strapdown keeps the tangential core *and adds*
  leader + drift + bias unobservables → strictly **harder** than the flat-output rung.
- **Honest covariance: HOLDS but weaker than hoped.** Consistent-but-**large** (tangential 3σ ≈ 2–3.6 m),
  not the *tight calibrated* covariance the post-mortem said was missing. Unblocks G6 only for
  **boundedness** (tube/robust controller backing off on a loose 2–3.6 m set), not tight formation.
- **CRLB floor: UNCHANGED** (−0.2%; follower IMU adds no Jacobian row touching the leader). Stays 2–3.6 m,
  **above** the ~1 m standoff and the 1.10 m range-only geometric ceiling; leader maneuvering buys zero.
- **OA-excitation-helps-bias: INERT here.** Under a single range the biases are unobservable at any
  excitation → bias states are a net **liability** (larger confident-wrong surface).
- **Velocity/attitude "sleeping trap" removed: CONDITIONAL** on retaining a direct attitude aid
  (gravity/mag); strapdown is open-loop dead-reckoning and drifts without it.

**One line:** the decisive lever is **measurement multiplicity** (more range channels / a complementary
sensor) + a **non-Gaussian filter** — neither of which the IMU provides.

**New gates (additive to G1–G16):**
- **G17 — IMU-realism / model-mismatch gate.** The truth sim MUST inject datasheet IMU noise + *drifting*
  bias random-walk + a model mismatch the IMU senses but the filter does not (drag / unmodeled thrust /
  mass error); gate that the filter's reported P **expands to cover** the truth error. Otherwise a "perfect
  IMU = the truth-driving command relabeled" passes consistency trivially while testing none of the insight
  (confident-wrong wearing an IMU label).
- **G18 — Hard direct-attitude-aid gate.** Retain an active attitude/gravity aid; No-Go if attitude is
  propagated open-loop. (Positive control: removing it must demonstrably break attitude/velocity bound.)
- **G19 — Long-horizon end-of-rollout bias-NEES.** Rollout ≫ bias-RW correlation time; per-bias-block NEES
  at the END (not the median) to catch delayed confident-wrong from unobservable bias growth.
- **G20 — Ridge-probe rung before the corner.** Neither single-axis rung surfaces the decisive non-Gaussian
  curved-annulus failure (#14) — Rung 1 uses the benign planner, Rung 2 judges only the planner. Add a rung
  pairing an aggressive worst-geometry maneuver with the squared-range likelihood, checked against a
  particle-filter reference, BEFORE the corner.
- **G21 — Net-OA-benefit gate.** Require closed-loop steady tangential error strictly BELOW the
  flat-output-plan+track (G16) baseline; else the rung is a no-value pass (a bounded honest-P loop that
  backs off all OA excitation is dominated by the simpler baseline that needs no IMU/CLKF).
- **G22 — Corner-scale real-time gate.** Validate solver runtime at the corner's actual size (full-quad
  STLOG, 15×15 tangent Gramian, 16-state-per-agent strapdown) — the binding constraint per memory, untested
  by either single-axis rung.
- **G23 — Gauge / observability-rank gate.** Before closing the loop, numerically check the augmented
  filter's information-matrix null space (promoting the leader to a state under a single range leaves the
  entire leader sub-state + biases unobservable) and confirm `proc_cov` is not manufacturing pseudo-
  observability (confident-wrong by the back door).

**The strategic fork:**
- **Path A — accept the floor:** flat-output plan + track (the G16 endpoint), validated and deployable now.
  Even three clean sub-rungs land here; the carried-quad corner is not worth chasing with single-range
  sensing.
- **Path B — move the floor:** measurement multiplicity + a non-Gaussian filter. The range-only-faithful
  multiplicity (no bearing confound) is **more agents** — a 3rd agent adds two range channels and makes the
  tangential directly observable. Build E1 along the way for an honest covariance + deployment realism, but
  the floor only moves with more channels.

**Value of `(E1,O0)`:** diagnostic (honest covariance; empirically pin the floor 2 vs 3.6 m; test the
attitude-aid conditional) + deployment realism — NOT a path to the corner on its own.

### §6.1 Phase 1 EMPIRICAL RESULT — `experiments/clins_bridge.py` (2026-06-06)

Built the collaborative INS (lean translation double-integrator `[r,v]`, IMU-driven, range-aided, AHRS
attitude stand-in, no bias states) and ran it CARRIED on the quad's actual trajectory (control-on-truth-
tracking of the frozen OA reference — the estimator port isolated, G1). The truth carries **drag** (a force
the IMU senses, the command does not) + IMU noise + accel bias, so the sim is honest (G17). Results
(10-seed medians, OA trajectory) — they confirm the §6 predictions:

- **G17 gate passes:** IMU-driven pos-NEES **13** vs command-driven **234** (18×) — the IMU captures the
  drag the command is blind to, so the sim is NOT the trivial "perfect-IMU = relabeled command".
- **The bridge's 0.33 m recovery was a harness artifact** (it predicted with the clean reference velocity).
  The honest collaborative INS gives **~0.78 m** recovery. OA still beats no-OAC (0.78 vs 3.83 m, ~5×).
- **The range-only EKF cannot be made TIGHT-consistent by process-noise tuning.** Sweeping the floor: NEES
  470 (q=0.16) → 277 (0.3) → 120 (0.6) → **52→13 (q≈1.0)** → 88 (2.0). It bottoms at NEES ~13 with an
  honest-but-**loose** tangential 3σ ≈ 1.1 m, and an unreasonably large floor (q≈1 m/s² ≈ 0.1 g). Below the
  floor the tangential **collapses** (NEES ≫ 100 — the #14 ridge); no floor restores NEES ~3. **Tight
  consistency needs a non-Gaussian filter, not the IMU** — exactly the §6 "decisive non-Gaussian ridge".
- **OA helps consistency** (excitation): NEES 13 (OA) vs 193 (no-OAC).

Net: the estimator port is structurally easy (the double-integrator works; the user's call) and yields an
**honest-but-loose** covariance (tangential 3σ ~1.1 m, at the standoff) — confirming the IMU is a
process-noise/honest-covariance lever, not an observability lever. The residual overconfidence (NEES 13,
startup-dominated) is the non-Gaussian ridge → a PF/RBPF is the lever for tightness, and measurement
multiplicity for the floor. Reproduce: `experiments/clins_bridge.py --ref /tmp/oa.npz --seeds 10`
(sweep `--q-accel`, isolate with `--accel-bias 0 --att-err 0`).

### §6.2 Phase 2 EMPIRICAL RESULT — `experiments/clins_closed_loop.py` (2026-06-06)

CLOSED the loop: flat OAC re-plans each step on the carried collaborative-INS estimate (RHOAC) → the
geometric tracker realizes it *using that estimate* → the full-fidelity quad (with drag) flies → IMU+range
drive the INS. The full (E1,O0) cell. Result (20-seed; the decisive test of whether an honest covariance
keeps the loop bounded where the old command-driven one diverged):

| config | recovery | NEES | %bounded | formation |
|---|---|---|---|---|
| OAC, plan-on-truth | 0.00 m | 0 | **100%** | 5.4 m |
| OAC, **IMU-driven (E1,O0)** | 2.9 m | 135 | **25%** | 5.5 m |
| no-OAC, IMU-driven | 13.6 m | 332 | 0% | 5.3 m |
| OAC, command-driven (**#8 repro**) | 4.4 m | 317 | 5% | 5.1 m |

- **The honest INS converts the CATASTROPHIC command-driven divergence (5%) into a MARGINALLY-bounded loop
  (25%)** — necessary, not sufficient, exactly as the verification predicted. The formation HOLDS (~5.4 m,
  the distance constraint + tracker work); the divergence is entirely in the estimate's tangential, and
  plan-on-truth's 100% shows the ceiling exists. This matches the #15 verdict (no deployable carried loop):
  the IMU helps but does not cross the threshold (cf. the old EKF-hybrid's 19% bounded).
- **Re-confirms the derivative-bounded finding from the control side:** the OA plan's acceleration is
  infeasible (~2.3 g), so it must NOT be fed forward — doing so overshoots the tracker and drives the
  formation to ~48 m *even plan-on-truth*. Pos+vel feedback alone holds formation. (The accel-feedforward
  bug was caught and removed via the plan-on-truth isolation — a wiring check, not an estimator finding.)

**Conclusion for (E1,O0):** built and validated end to end. The IMU/collaborative-INS port is structurally
easy (the user's call) and yields an honest-but-loose covariance that *removes the divergence mechanism* —
5% → 25% bounded — but the residual #14 ridge (loose Gaussian filter) still diverges the tangential on
~3/4 of seeds. **The next lever is NOT more estimator process-model work** — it is a **non-Gaussian filter**
(PF/RBPF, for robustness on the curved range ridge) and/or **measurement multiplicity / more agents** (to
move the floor below the standoff). Reproduce: `experiments/clins_closed_loop.py --seeds 20`.

### §6.3 Is tracking error a contributor to the marginality? NO (intervention sweep, 16-seed)

The OA commands strong motion (~3.1 m/s relative) the tracker falls ~15% short of (achieved/commanded
maneuver speed ach/cmd ~0.85–0.90, vtrack ~1.4 m/s). Tested whether that shortfall drives the ~25–31%
marginality, via `clins_closed_loop.py` knobs (OAC IMU-driven, 16 seeds):

| intervention | vtrack | ach/cmd | NEES | %bounded |
|---|---|---|---|---|
| baseline (n10) | 1.48 | 0.90 | 173 | **31%** |
| faster tracker n20 | 1.37 | 0.86 | **74** | **31%** (flat) |
| faster tracker n40 | 1.48 | 0.85 | 166 | 25% (down) |
| gentler OA vscale0.7 | 1.34 | **0.99** | 307 | **6%** |
| gentler OA vscale0.5 | 1.09 | 0.91 | 173 | 6% |
| feasible FF (1 g) | 4.55 | 0.92 | 78 | **0%** |

- **Clean test (faster tracker, same OA commands):** n20 tightens tracking and *halves* the estimator
  NEES (173→74) — yet %bounded does not move (31%→31%); n40 nudges it *down*. Improving tracking buys ≈0 pp.
- **Gentler OA eliminates the shortfall (ach/cmd→0.99) but COLLAPSES %bounded to 6%** — the boundedness
  comes from the aggressive command's *observability content*, not from executing it faithfully. Gentling
  to fix tracking is strongly counterproductive (−25 pp).
- **Feasible feedforward hurts** (0%, formation 12.5 m) — the OA accel shape is observability-bearing, not
  executable; confirms it must not be fed forward.
- **Reconciliation:** plan-on-truth has the *same* shortfall (ach/cmd 0.82) yet is 100% bounded.

**Quantified:** tracking-shortfall contribution ≈ **0 pp (±6)**; estimate / #14-ridge contribution ≈
**69 pp** (the entire plan-on-truth → IMU-driven gap, at matched tracking). The user's premise (the tracker
falls short of the strong OA command) is factually true, but the shortfall is **benign** — the aggressive,
"infeasible" command is *load-bearing* for observability. Spend effort on the estimate (non-Gaussian filter
/ regularize the singular direction / measurement multiplicity), NOT the tracker.

### §6.4 The 3-agent floor check — FLIPS the recommendation (`experiments/multi_agent_floor.py`)

Cheap PCRB/FIM check before building: does measurement multiplicity (a follower ranging to N-1 agents)
remove the #14 ridge by geometry? Position FIM `F = Σ uᵢuᵢᵀ/σ²` (uᵢ = LOS unit vector), σ_r = 0.1 m,
5 m standoff:

| N agents | range channels | static rank/3 | static worst 1σ | recursive worst 1σ (w/ motion) |
|---|---|---|---|---|
| 2 | 1 | 1/3 | ∞ (motion-only) | 0.55 m |
| 3 | 2 | 2/3 | ∞ (motion-only) | 0.51 m |
| 4 (non-coplanar) | 3 | **3/3** | **0.19 m** | **0.09 m** |
| 4 (coplanar) | 3 | 2/3 | ∞ | — |

Two findings that change the plan:

1. **The floor is NOT the binding constraint at this standoff.** Even N=2 has a (position-only,
   known-leader) recursive floor of **0.55 m ≪ 5 m standoff**. Yet the Phase-2 Gaussian filter achieves
   only ~2.7 m — ~5× *above* the floor. So the binding constraint is the **filter's inability to reach the
   floor (the #14 ridge)**, not the floor itself. (Caveat: 0.55 m is the optimistic position-only floor;
   the realistic full nav-state floor is higher, ~2–3.6 m per §6 — still below 5 m, but a tight ~1 m
   formation *would* be floor-bound. The flat/clins setup is not.)
2. **Multiplicity's ridge-removal is governed by the WORST direction, so it's an N=4 (3D) requirement, not
   N=3.** Each range adds one LOS direction; the recursive floor tracks the least-observable direction.
   N=3 (2 ranges) makes 2 of 3 directions instantaneous but leaves the out-of-plane one motion-only →
   barely moves the 3D recursive floor (0.51 vs 0.55). **Full instantaneous observability (ridge removed on
   every axis) needs N=4 / 3 non-coplanar ranges** (floor 0.19 m static). **Coplanar geometry fails**
   (N=4 coplanar stays rank 2) — a closed-loop controller must be *constrained to keep the formation
   non-coplanar* (a new geometry guardrail). For a strictly PLANAR formation, N=3 (2 non-collinear ranges)
   suffices (rank 2 = full in 2D).

**Decision (flips §6's tentative "3-agent first"):** measurement multiplicity is NOT the cheap N=3 win —
removing the ridge by geometry needs N=4 + non-coplanar maintenance (3D), heavier than hoped. And the floor
isn't the binding constraint here anyway — the **filter** is. So the **non-Gaussian filter (PF/RBPF) is the
lighter, more direct next lever**: it lets the *existing* N=2 floor (~0.5 m) be achieved, closing the ~5×
gap between the 2.7 m the Gaussian filter delivers and the floor. Measurement multiplicity (N=4 non-coplanar,
or N=3 planar) is the heavier geometric alternative, warranted only for a *tight* (≲1 m) formation where the
floor itself binds.

### §6.5 TWO LEADERS empirically FIX the carried loop — 25% → 90–95% bounded (`clins_two_leaders.py`)

The user's idea: give the follower a 2nd KNOWN leader and fuse ranges to both, so the tangential is pinned
by a 2nd anchor instead of riding the single-range #14 ridge. Tested in the full closed loop (the INS fuses
both ranges; the OA still plans on leader 1 only, isolating the FILTER benefit), **20-seed** (16-seed gave
within-noise identical: 94/88/56):

| placement of 2nd leader | OAC %bnd | OAC rec | OAC NEES | no-OAC %bnd |
|---|---|---|---|---|
| (1 leader, baseline) | 25% | 2.89 m | 135 | 0% |
| lateral flanks (collinear) | **95%** | 0.49 m | 4.8 | 15% |
| one ahead (in-plane tangential) | **90%** | 0.36 m | 4.6 | 10% |
| ahead + above (3D) | 55% | 0.82 m | 18.0 | 20% |

**This is the first thing in the whole climb that fixes the loop.** A 2nd known leader takes the carried
loop from marginal (31%) to robustly bounded (**88–94%**), recovery 2.74→0.4 m, NEES 173→~5 (consistent).

- **It's a SYNERGY, not the anchor alone.** 2 leaders alone (no-OAC) is still mostly unbounded (6–19%):
  2 ranges = rank 2, so one direction stays motion-only and (for the symmetric lateral case) the no-OAC
  NEES is catastrophic (3919). The 2nd anchor removes the ridge on the directions its LOS spans; the OA
  maneuver resolves the remaining one and breaks the symmetry. **Both are needed.**
- **Placement matters, but in-plane wins (and the static-FIM intuition was WRONG for the closed loop).**
  I predicted the lateral/collinear placement (statically rank-1 degenerate — both LOS along the
  leader-follower axis) would barely help. In the *closed loop with an in-plane offset error + an OA
  maneuver*, it did **best** (94%): the maneuver breaks the static symmetry, and an in-plane 2nd anchor
  aligns its information with where the uncertainty and the maneuver actually live. The out-of-plane (3D,
  +z) placement did *worst* (56%) — it "spends" range information in z, where there is little error and
  little maneuvering. Lesson: place the 2nd anchor in the plane of the error/maneuver, not orthogonal to it.
- **Reconciles with §6.4 and confirms its thesis.** The recursive *floor* barely moved with the 2nd range
  (§6.4: 0.51 vs 0.55 m) — yet %bounded jumped 31→90%. Exactly as §6.4 argued: the binding constraint was
  never the floor, it was the **filter's inability to reach it (the #14 ridge)**. A 2nd anchor makes the
  weak directions *instantaneously* observable, so the Gaussian filter stops collapsing — the ridge is
  removed at its geometric source.

**LEVER UPDATE (supersedes §6.4's tentative non-Gaussian-filter recommendation):** measurement multiplicity
— a 2nd known anchor — is the EMPIRICALLY VALIDATED fix for the (E1,O0) marginality (90% bounded), and it
beats a fancier filter: it removes the ridge by geometry rather than trying to model it. Caveats for the
climb: (i) it requires a 2nd GNSS-equipped leader / known beacon (a real operational cost); (ii) the OA
maneuver is still needed (the anchor alone gives 6–19%); (iii) the placement must be in the error/maneuver
plane and the controller kept out of the degenerate symmetric configuration (no-OAC NEES 3919 is the
warning). Next refinement: a 2-range OA objective (let the planner exploit both anchors — it currently
plans on leader 1 only) and a 20-seed confirmation. Reproduce: `experiments/clins_two_leaders.py --seeds 16`.

### §6.6 The 2-RANGE OA objective — a strict win, and it SOLVES (E1,O0) (`clins_two_leaders.py`, 20-seed)

§6.5 fixed the loop with 2 leaders while the OA still planned on leader 1 only (isolating the filter
benefit). §6.6 lets the planner exploit BOTH anchors: a 12-state STLOG over [leader1, follower, leader2]
(both leaders directly observed, follower fused by both ranges; `build_oac2`). 20-seed, OA-1rng vs OA-2rng:

| placement | %bnd 1→2rng | recovery | NEES | cmd_spd (maneuver) | formation |
|---|---|---|---|---|---|
| lateral flanks | 95 → **95** | 0.49 → **0.26** | 4.8 → 4.0 | 3.14 → **2.86** | 5.34 → 4.85 |
| one ahead | 90 → **100** | 0.36 → **0.25** | 4.6 → 4.7 | 3.16 → **2.84** | 5.29 → 4.89 |
| ahead + above (3D) | 55 → **100** | 0.82 → **0.24** | 18.0 → **7.9** | 3.18 → **3.05** | 5.31 → 4.90 |
| (1 leader baseline) | 25% | 2.89 | 135 | 3.16 | 5.47 |

**Strict win on every axis:** %bounded ties or improves (the weak 3D placement is RESCUED 55→100, ahead
90→100, lateral holds 95); recovery ~2–3× better (0.24–0.26 m); NEES lower (3D 18→7.9, all consistent ~5);
maneuvering gentler (cmd_spd ~7–10% lower — the predicted "two anchors localize, so maneuver less"); and
the formation holds tighter (5.3→4.9 m). The biggest gain is the 3D rescue: a 2-anchor-aware planner makes
the loop **robust to 2nd-leader placement** (an operator no longer has to place the anchor perfectly).

**(E1,O0) is SOLVED for the 2-leader configuration:** 2 known leaders + range fusion + a 2-range OA
objective → **95–100% bounded, recovery ~0.25 m, NEES ~5 (consistent)**, from the carried collaborative
INS, no truth in the loop — vs the single-leader 25% / 2.9 m / NEES 135. The full chain that works:
honest IMU-driven INS (removes the divergence mechanism) + a 2nd known anchor (removes the #14 ridge by
geometry) + a 2-range OA objective (exploits both anchors → robust + gentle). The remaining caveats are
operational (a 2nd GNSS leader is required) and the climb's untouched axes (the OA is still flat-output
O0; lifting to a full-quad OA objective O1 is separate, and the floor still binds for a TIGHT ≲1 m
formation — see §6.4). Reproduce: `experiments/clins_two_leaders.py --seeds 20`.

### §6.7 Does HIGHER-ORDER observability reveal the invisible coordinate? NO (`higher_order_obs.py`)

Tested (before climbing to O1) whether the tangential's invisibility is an *order* phenomenon — i.e. a
2nd-order O0 model (double-integrator, velocity a STATE) + a higher-order STLOG (more Lie derivatives)
makes it observable. Probe = accumulated STLOG over a tangential maneuver → tangential follower-position
1σ vs STLOG order, swept over dt:

| dt | O0-1st tang 1σ, order 1→5 | O0-2nd tang 1σ, order 1→5 |
|---|---|---|
| 0.2 (operating) | 1.78 → 1.78 (FLAT) | 3.58 → 3.55 (FLAT, worse) |
| 1.0 | 0.24 → 0.24 (FLAT) | 0.41 → 0.41 (FLAT, worse) |
| 3.0 | 0.105 → 0.105 (FLAT) | 0.15 → 0.15 (FLAT, worse) |

**REFUTED — higher STLOG order does nothing.** The tangential 1σ is flat across orders 1–5 at *every* dt,
for both models. Mechanism: each order-k STLOG term carries a `dt^(2k+1)` weight, so at the operating dt
the higher-order terms are ~10²–10³× suppressed and contribute ≈0. What *does* resolve the tangential is
the **accumulated motion / time window** (1σ 1.78→0.24→0.105 as dt 0.2→1.0→3.0) — information-over-time,
not Lie-derivative order. And the **2nd-order model is consistently WORSE** (more weakly-observable states
spread the information thinner). Closed-loop confirms: STLOG order 2/3/4 on the single-leader loop give
*identical* %bounded (38/38/38, 8-seed), recovery, NEES — the order-invariant STLOG → an order-invariant
OA objective → an identical loop.

**Conclusion:** the tangential is **information-limited, not order-limited.** Higher-order observability
cannot conjure information the measurements don't contain; only more measurements (a 2nd anchor, §6.5/6.6)
or more accumulated motion add it. (Compounding this: even if the STLOG *saw* high-order observability, the
1st-order EKF can't exploit it — only a 2nd-order filter / UKF recovers the high-order tangential, the §6
UKF findings.) So: do NOT pursue higher-order O0/STLOG; the validated lever is measurement multiplicity.
Reproduce: `experiments/higher_order_obs.py --dt 0.2 1.0 3.0`.

### §6.8 O1 observability — higher order helps the RANK (structural), 2 leaders give the MAGNITUDE

Before building the O1 corner, probed whether the full-quad relative-pose model (`inter_quadrotor_pose`,
state [r_lf, q_fl, v_lf]; obs [range², q_fl] — relative attitude measured, velocity + tangential position
observed only via the range) justifies higher-order STLOG, in contrast to O0 (§6.7, where it was flat).
Manifold STLOG, orders 1–5, dt=0.2 (`experiments/o1_obs_probe.py`):

| config | tang-pos 1σ (ord 1→5) | vel 1σ | rank/9 (ord 1→5) |
|---|---|---|---|
| 1 leader (range + attitude) | 2.94 → 2.94 (≈prior, unobservable) | 2.48 (weak) | **5 → 6** |
| 2 leaders (range1+range2+att) | **0.04** → 0.04 (observable) | 1.78 (weak) | 7 → 7 |

**The O0/O1 distinction is real, and your high-order justification holds — at the RANK level:**
- **Higher order lifts the 1-leader rank 5→6 at order 2** — one more direction becomes *structurally*
  observable through the coupled dynamics chain (pos←vel←accel←attitude). O0 had NO such rank lift (§6.7).
  So O1 genuinely needs order ≥2 for structural observability — velocity here is a *state the dynamics
  integrate*, not the droppable input it was in O0.
- **But the MAGNITUDE is still dt-suppressed** (1σ flat across orders): the structurally-observable
  tangential sits at the prior (2.94 m ≈ unobservable) for one leader at *every* order. Higher order makes
  it *rank*-observable, not *accurately* observable.
- **The 2nd leader supplies the accuracy** the order can't: tangential 2.94→0.04 m (75×), rank 6→7. The
  velocity stays weak (1.78 m) even with 2 leaders — but in the loop the **IMU dead-reckons velocity** (E1),
  so the STLOG's velocity weakness doesn't bite; the OA only needs to observe the tangential *position*.

**Synthesis — higher order and 2 leaders are COMPLEMENTARY for O1:** higher order gives the OA objective
*structural completeness* (it sees the full coupled observability — the paper's point), the 2nd leader
gives *accuracy* (the magnitude), and the IMU handles velocity/attitude. This is exactly why "help O1 with
two leaders" is the right framing: O1 alone is structurally-but-not-accurately observable; the 2 leaders
close the magnitude gap. Reproduce: `experiments/o1_obs_probe.py`.

### §6.9 O1 corner — step 1: the 2-range full-quad OA works in plan-on-truth (`o1_corner.py`)

User chose the faithful corner: the OA plans the follower's thrust + body-rates DIRECTLY by maximizing the
full relative-pose STLOG (`inter_quadrotor_pose`, order 5, manifold) over TWO ranges, observability target =
position + velocity [0,1,2,7,8,9] (attitude measured). Step 1 validates the OA machinery in plan-on-truth
(perfect feedback, relative-pose rollout):

| config | dist (start→end, min/max) | ms/solve |
|---|---|---|
| no-OAC (hover-hold) | 2.06 → 2.06 (flat) | — |
| O1 OA, 2-range, maxiter 30 | 2.06 → 1.05 (1.02 / 2.92) | ~5000 |
| O1 OA, 2-range, maxiter 6 (early-stop) | → 1.44 (1.35 / 2.74) | ~640 |

**The corner's planner is sound:** the order-5, 2-range, manifold O1 OA solves, plans a real observability
maneuver (the relative distance explores the [1,3] m band — the dist constraint binds), and holds the
formation; no-OAC just hovers. The frontier early-stop (maxiter 6) keeps the maneuver and cuts the solve to
~640 ms. **Real-time caveat:** ~640 ms is ~6× the single-leader frontier's ~115 ms — the 2nd range + the
custom 2-range observation add cost; real-time tuning (JIT the obs, smaller window, the penalty-form solver)
is a known follow-on, not a blocker for the corner's validation.

**Remaining (step 2, the closed corner):** the carried IMU-driven relative-pose ESEKF (E1) — drive
`ErrorStateEKF.predict` with the measured follower IMU (not the command) + the leader's known cruise, fuse
the 2 ranges + the measured relative attitude, close the loop on the carried estimate. The pieces are all
in hand (the O1 OA here; the IMU-INS + 2-range fusion pattern from §6.5/6.6; the ESEKF on the quad-baro
branch). Reproduce step 1: `experiments/o1_corner.py [--order 5 --maxiter 6]`.

### §6.10 O1 corner step 2 (closed) — the lean estimator confirmed; direct thrust+rates caps at ~50%

Closed the (E1,O1) corner (O1 OA re-planned on a carried relative-pose ESEKF, 2 leaders, direct
thrust+rates). 8-seed (`o1_corner.py --closed`):

| estimator / control | %bounded | recovery | NEES |
|---|---|---|---|
| full range-only ESEKF | **12%** | 7.53 m | 692 |
| full ESEKF, larger proc floor (×30 / ×100) | 0% / 0% | 23 / 30 m | worse |
| **LEAN est** (velocity+attitude IMU/AHRS-measured) | **50%** | 1.90 m | 1128 |
| lean + gentle control (rate ×0.4 / ×0.2) | 50% / 25% | 1.5 / 0.6 m | 404 / 210 |

**Findings (single-seed close was luck — the #6 fragility lesson, again):**
- **The full-fidelity estimator is the wrong call (user-confirmed):** the full relative-pose ESEKF is
  ridge-fragile (12%), and the O0 honest-floor lever makes it *worse* (a looser floor → the OA plans on a
  wronger velocity/attitude). It cannot be tuned to robustness.
- **The lean IMU-driven estimator is the right call (user's redirect, confirmed):** velocity+attitude
  measured (IMU/AHRS), only the tangential position range-aided → 12% → **50%**, recovery 7.5 → 1.9 m. The
  full estimator was *half* the problem. O1 (the objective) IS a value adder (§6.8/6.9); the full
  *estimator* is not.
- **The other half is the direct thrust+rates control:** it destabilizes the carried estimate on ~half the
  seeds; gentling it trades *estimate for formation* (rate ×0.2: recovery 0.6 m but %bnd 25% — too gentle
  to hold the band), so there is no net-win operating point. The direct corner caps at ~50%.

**The robust realization of (E1-lean, O1):** put the O1 *objective* on the architecture that already hits
95% — the (E1,O0,2-leader) **flat-plan + geometric-tracker + lean translation-INS** loop (§6.5/6.6) — not
the direct thrust+rates relative-pose corner. The geometric tracker smooths the control (removing the
~50%-capping instability) and the translation INS is robust. The remaining build is to score the OA with
the full-quad O1 STLOG via the flatness map (flat outputs+derivatives → quad state), keeping the validated
control + estimator. Synthesis: **O1 objective (value) + lean estimator (robust) + flat+track control
(robust) = the robust (E1-lean, O1)**; the direct-thrust-rates corner is the wrong control architecture for
the carried loop. Reproduce: `o1_corner.py --closed [--rate-scale R]`.

### §6.11 Plan-flat / score-O1 machinery works (`o1_flat.py`; flatness map from minsnap_trajectories)

The robust (E1-lean, O1) needs the O1 *objective* on the flat-plan+track control (not direct thrust+rates,
§6.10). The link is a differentiable flatness map: the OA's decision variables are the follower's
flat-output velocities → (Mellinger-Kumar) → quad inputs → the order-5 relative-pose STLOG. My hand-rolled
map gave NaN gradients (tilt-quaternion singularity at hover) + a non-PSD STLOG (noisy finite-diff body
rates). **Ported the vetted map from `minsnap_trajectories.flat_output_to_quadrotor_trajectory`** (JAX,
differentiable): half-vector tilt quaternion `tilt_den=√(2(1+z_z))` (smooth except z_z=-1) + analytic body
rates from `dz = -z×(z×jerk)/|z|` (from the flat-output jerk, not a finite-diff).

Result: the machinery works. Differentiable (no NaN); a bounded gradient ascent (clip |vel| ≤ 3 m/s) grows
a bounded observability maneuver, driving the O1 objective from 9.84 (hover) to 6.77 (mean |vel| 3.8 m/s).
(Positive `neg_softmin_eig` values are expected: the truncated order-5 STLOG is non-PSD at degeneracy —
the very reason softmin-eig is used; the OA minimizes it. Unconstrained ascent explodes the velocity, as it
should — the OA's velocity bound + distance constraint cap it.)

**Next:** drop the O1-on-flat objective into the clins flat+track+lean-INS closed loop (a custom RTController
cost) and multi-seed it -- the real test of whether the O1 objective beats O0 *with the lean estimator*.
Open tension to settle there: with the lean estimator measuring velocity+attitude, the O1 objective's
velocity/attitude-observability advantage over O0 may be partly moot (its §6.8 value was for the full
range-only estimator); the residual O1 value is dynamically-faithful position-observability maneuvers.
Reproduce: `experiments/o1_flat.py`.

### §6.12 (E1-lean, O1) CLOSED LOOP: O0 dominates O1 at the robust operating point (`o1_flat_closed.py`)

The capstone of the (E1-lean, O1) corner. Built the robust realization promised in §6.10/6.11: the O1
*objective* (full-quad relative-pose STLOG via the flatness map, §6.11) driving the VALIDATED control+
estimation architecture -- the clins lean translation-INS + geometric tracker + 2-range fusion that hits
95% at O0 (§6.6). Wrapped the O1-on-flat objective in an `ObservabilityCost`-compatible cost
(`O1FlatCost`: `__call__`->flatness-map O1 STLOG, `eval_integrator`->flat rollout for the distance
constraint) and dropped it into the SAME `RTController` as `build_oac` (SLSQP, maxiter 6, hard distance
constraint). So the only thing differing from the O0 baseline is the objective. 20 seeds, 2 lateral leaders.

**The optimizer is a giant confound -- always match it.** First pass used a crude projected-gradient
solver (8 fixed steps + penalties). It collapsed BOTH objectives: O0-via-PG 15%, O1-via-PG 35% (vs
O0-via-SLSQP 95%). The disentangler -- run *my* solver with the *O0* objective -- proved the solver, not
the objective, was the killer. NEVER compare objectives across different solvers; the earlier "O1 degrades
the loop" was a pure solver artifact and is void.

**Two real bugs in the O1 objective, both found + fixed:**
- Flatness-map tilt quaternion `tilt_den=sqrt(2(1+z_z))` went NaN when a jagged plan implied an inverted
  thrust (z_z<-1). Guard: `z_z = max(z_z, 2)` (thrust can't invert). Without it the quad quaternion -> 0.
- **Plan-dt:** the flatness finite-diff used the control DT=0.1, but the STLOG integrates the inputs at
  STLOG_DT=0.2 (the plan/lookahead spacing). The plan points are 0.2 s apart, so accel was overestimated
  2x -> O1 scored physically inconsistent, over-aggressive maneuvers. Fix: finite-diff at STLOG_DT.

**Fair result (matched SLSQP + hard constraint, 20 seeds):**

| objective | %bnd | rec_med | rec_p90 | NEES |
|-----------|------|---------|---------|------|
| O0 flat (m6)                  | **95%** | 0.49 | 0.97  | 4.8 |
| O1 full-quad, PG (crude)      | 35%  | 3.81 | 47.0  | 43.6 |
| O1 full-quad, SLSQP m6 (dt-bug)| 55%  | 1.35 | 11.4  | 13.2 |
| O1 full-quad, SLSQP m6 (FIXED)| **75%** | 0.60 | 3.25 | 5.2 |

Each confound removed pulled O1 toward O0: optimizer 35->55, plan-dt 55->75. At its best (m6, fixed) O1 is
VIABLE -- bounded + NEES-consistent (5.2 vs O0's 4.8), recovering the tangential error (0.60 m vs 0.49 m).
But it caps at 75%, below O0's 95%.

**The decisive tell -- O1 is non-monotonic in solver effort (dt-fixed maxiter sweep):** m3->70%, m6->75%,
m15->55%. Optimizing the full-quad objective HARDER makes it WORSE: it drives more aggressive observability
maneuvers that destabilize the lean INS -- the §6.10 carried-loop tension, resurfacing even on the robust
architecture. O0 has no such peak; it is flat-robust at 95% (its flat-kinematic maneuvers are benign).

**VERDICT.** At the robust lean-INS operating point (the estimator MEASURES velocity + attitude), the simple
O0 flat objective DOMINATES the full-quad O1 objective: O1 is viable but caps at 75% and is fragile to
over-optimization, while O0 is 95% and monotone-robust. This empirically settles the §6.11 tension: O1's
structural value (§6.8 -- making velocity/tangential-position observable) is MOOT when the estimator is
lean, and its aggressive maneuvers then only add fragility. **Recommendation: at the lean operating point,
ship O0; O1's flatness-map + order-5 manifold machinery is not worth it here.**

**This resolves the two-rung climb.** O1's payoff regime is NOT (E1-lean, O1) -- it is (E0, O1): the FULL
range-only estimator where velocity/attitude are unobserved (§6.8). But that is exactly the carried,
range-only corner that diverges (#8-#15, the §4 NO-GO). So the climb closes with a clean, honest map: the
robust, deployable corner is (E1-lean, O0) at 95%; the O1 objective adds value only in the corner we've
shown is not safely closable on a carried estimate. Reproduce: `experiments/o1_flat_closed.py --only all`.

### §6.13 Belief-space dual-control is NOT the lever: the carried-O1 corner is CONSISTENCY-limited (`belief_space_oac.py`)

The decisive test of the HO-observability memo's next-step #1 (does planning on the covariance rescue the
diverging carried-O1 corner?). Replaced the deterministic "maximize the STLOG" objective with a belief-space
one -- "minimize log-det of the predicted POSTERIOR covariance" over the horizon (position+velocity
sub-block), seeded each solve with the carried filter covariance P0 and propagating the ESEKF Riccati along
the planned trajectory with the FILTER'S OWN F/G/H. Same RTController, model, noise, horizon, hard
constraint, truth sim as the diverging deterministic O1 runs -- only the objective differs. Unit-checked:
the planner's covariance propagation reproduces the ESEKF's actual predict()/update() to MACHINE PRECISION
(3e-18 / 3e-17), so results are not a propagation bug.

Result (20 seeds, full range-only ESEKF): **NO-GO.** Belief-space does not rescue the corner -- ~25% bounded
peak (vs O0's 95%), NEES catastrophically out of band everywhere (1e3-1e6 vs ~3 expected), and it INVERTS
with solver effort (m6 ~25% -> m15 ~0%; more optimization is WORSE). Mechanism, instrumented directly within
one run: the planner drove its planned log-det(P) DOWN ~7 nats while the realized NEES climbed 1,649 ->
66,556 and true error grew 1.4 -> 4.6 m. Minimizing a confident-wrong covariance just rewards the spurious
over-confident directions. (The 3-skeptic verification also confirmed: indexing correct; m15 blow-up is real
over-excitation not a slogdet artifact; the distance constraint never binds; the information-form functional
reproduces the same signature -- every covariance-shrinking functional fails identically.)

**VERIFIED VERDICT (CONFIRMED, with a REFINE):** the carried full-quad range-only O1 corner is
CONSISTENCY-limited, not planning-representation-limited. State the numbers QUALITATIVELY (n=20 medians are
noisy; the robust signal is "belief-space does not improve with effort, it degrades"). The precise claim is
NOT "belief-space fails" but: **belief-space planning on an INCONSISTENT filter's covariance fails -- and is
untestable as a self-limiter until the covariance is made consistent.** The lever is a CONSISTENT /
non-Gaussian filter (FEJ / observability-constrained / iterated / invariant EKF -- `predict_fej`/`update_fej`/
`update_iterated` already ship in `error_state_ekf.py`, UNUSED here) and/or RICHER measurements (the X-IO
LIDAR+VIO port), NOT a better planning objective.

**The one clean follow-up (the consistency-vs-information separator):** re-run the IDENTICAL belief loop with
`predict_fej`/`update_fej` in BOTH the filter and the Riccati seed. If NEES then falls into the chi^2 band and
the m6->m15 inversion disappears -> the corner was estimator-limited (rescuable, belief-space lives). If
recovery stays flat with an honestly LARGE Sigma -> the corner is genuinely information-limited, and only the
X-IO measurement-rich port (or more anchors) moves it. Reproduce: `experiments/belief_space_oac.py --seeds 20`.

### §6.14 (E1,O1) VARIATION: a barometer substitutes for the 2nd leader (`baro_o1.py`)

Tested whether a cheap onboard BAROMETER can replace the 2nd known leader in the (E1,O1) corner (a 2nd GNSS
anchor is expensive; a baro is standard kit). The barometer measures the follower's absolute altitude --
exact + differentiable from the relative state, `baro(x) = to_absolute_state(x_l1, x)[12]` -- added to BOTH
the carried ESEKF and the O1 OA's STLOG. Geometric prediction: 1 range leaves the TWO tangential directions
weak; a 2nd range removes one (a horizontal), a barometer removes a DIFFERENT one (the vertical); both leave
ONE weak horizontal direction for the O1 maneuver. Held identical to o1_corner's closed corner (lean
IMU-driven relative-pose ESEKF, direct thrust+rates O1 OA, quad truth+drag); only the OBSERVATION changes.

Result (lean estimator, 20 seeds, baro_std 0.3 m): the barometer MATCHES the 2nd leader.

| observation     | %bnd | rec_med | rec_p90 | NEES |
|-----------------|------|---------|---------|------|
| 1 range (floor) | 20%  | 3.45    | 19.9    | 8403 |
| 1 range + baro  | 45%  | 1.59    | 6.0     | 627  |
| 2 ranges (ref)  | 45%  | 1.72    | 4.5     | 1128 |

**VERDICT: a barometer substitutes for the 2nd known anchor at the observation level** -- same lift off the
range-only floor (20% -> 45%), comparable recovery + NEES, at near-zero deployment cost (no 2nd GNSS leader).
The geometric "each sensor kills one weak tangential" picture holds.

Caveats (don't oversell): (i) BOTH cap at 45% -- the ceiling here is the direct thrust+rates control
destabilizing the carried estimate (the section 6.10 cap), NOT the observation; to lift further, put
1range+baro on the flat+track architecture (where O0/O1 reach 75-95%). (ii) NEES stays ~1e3 -- the baro adds
INFORMATION, not CONSISTENCY; the carried ESEKF is still loose (the section 6.13 consistency limit is
untouched). (iii) Idealized baro (direct altitude + 0.3 m Gaussian noise); a real barometer carries a
slowly-varying pressure (QNH) bias that would need a bias state and could erode the win. Next: 1range+baro on
the flat+track loop, and a baro-noise sweep. Reproduce: `experiments/baro_o1.py --seeds 20`.

### §6.15 (E1,O0) DEPLOYABLE corner: barometer gets 85%, NOT the 2-leader 95% -- a FUNDAMENTAL gap (`baro_o0.py`)

§6.14's barometer win was on the O1 corner, which is control-capped at 45% -- so it did not answer the real
question: does a barometer reach high boundedness on the DEPLOYABLE (E1,O0) corner (clins flat-plan + tracker
+ carried IMU-driven translation INS, ~95% with 2 leaders)? Added a barometer mode to clins_closed_loop
(`build_oac_baro`: the follower altitude x[6] in the flat STLOG; a direct r_z pseudo-measurement in the
translation INS). 20 seeds:

| observation          | %bnd | rec_med | NEES |
|----------------------|------|---------|------|
| 1 range (floor)      | 25%  | 2.89    | 135  |
| 1 range + baro 0.1 m | 85%  | 0.54    | 18.8 |
| 1 range + baro 0.3 m | 85%  | 0.54    | 18.4 |
| 1 range + baro 1.0 m | 85%  | 0.81    | 32.0 |
| 2 ranges (reference) | 95%  | 0.26    | 4.0  |

**VERDICT: a barometer lifts the (E1,O0) loop to HIGH boundedness (25% -> 85%) at near-zero deployment cost --
a strong, deployable result -- but it does NOT reach the 2-leader 95%, and that gap is FUNDAMENTAL, not a
sensor-quality issue.** The 85% ceiling is INSENSITIVE to barometer precision (0.1 m -> 1.0 m all give 85%;
only recovery/NEES degrade gracefully). The barometer observes radial + vertical and leaves the along-track
horizontal tangential to the OA maneuver ALONE -- that residual is the ~15% that occasionally fails. The 2nd
range, with the maneuver, resolves BOTH tangentials -> 95% and far tighter consistency (NEES 4 vs ~19). So the
honest answer to "does (E1,O0) + 1range+baro get high boundedness?": yes (85%) but not parity with a 2nd
anchor. This CORRECTS the optimistic aside in §6.14 ("flat+track would lift it to ~95%"): flat+track lifts
boundedness a lot (the O1 45% -> the O0 85%), but the barometer remains a genuinely weaker (cheap, deployable)
substitute for a 2nd known anchor. Reproduce: `experiments/baro_o0.py --seeds 20`.
