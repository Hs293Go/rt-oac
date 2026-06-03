# Phase 0 findings — where the OAC solve time actually goes

All numbers measured on this machine (16 CPU, JAX x64, `JAX_PLATFORMS=cpu`), order-5 STLOG,
N=20, the quadrotor `inter_quadrotor_pose` scenario, persistent XLA cache enabled. Scripts
in `benchmarks/`.

## 1. The observability computation is cheap; the optimizer is the cost

JIT-compiled component costs (steady state, `block_until_ready`):

| component | cost |
|---|---|
| STLOG eval (one node) | 52 µs |
| objective (forward rollout + 20× STLOG + metric) | 0.5 ms |
| gradient (reverse pass through order-5) | 4.0 ms |
| constraint (distance over horizon) | 0.035 ms |
| constraint Jacobian | 0.31 ms |

Full `trust-constr` solve (40 iters): **0.68 s inside our JAX callables (6%)**, **10.6 s
inside scipy's own code (94%)** (`benchmarks/diagnose_scipy.py`). Confirmed: no per-solve
recompilation (`new_compiles` = 0 after warmup), and **no numpy↔JAX boundary penalty**
(controller methods cost the same whether handed numpy or jax arrays). The often-cited
"STLOG is expensive" framing is wrong for this code — eager (un-jitted) STLOG is ~440 ms,
but jitted it is 52 µs; everything runs jitted inside the controller.

→ **Speeding up the STLOG/gradient is not the path. The optimizer is.**

## 2. Every solve exhausts the iteration cap

Companion `optimization_results_stlog.npz`: `nit ≡ 40` (min=max=mean=40) across all 2399
solves; optimality never reaches `gtol`. So the solver always burns its full budget — the
iteration count is the lever, which makes warm-starting (currently discarded each step) and
a non-thrashing objective high-value.

## CORRECTION (post-Phase-0): the degeneracy was a pathological test config

Findings §3–§5 below were measured at the default `x0 = [0,-1,1, 0,0,0,1, 0,0,0]` with the
follower hovering — which is a **Manifestly Unobservable Configuration (MUC)**. The thesis
(`include/ch4_cl_for_quads.tex`) enumerates four MUCs for this range+attitude system:
(1) zero relative velocity / co-hover, (2) radial co-motion `v ∥ r`, (3) perfect orbit
`r₁v₂−r₂v₁=0`, (4) identical attitudes `q_fl = identity`. The default config hits (1) and
(4), so min-eig ≈ 0 there is **expected and correct**, not a defect. Escaping MUCs is out of
scope.

Two consequences:
- The "objective is degenerate / zero gradient" and "10 s solve" results below are the
  **MUC-config** behavior. The companion's 0.4 s average comes from **non-MUC** states the
  follower visits mid-flight. There is likely **no real speed regression** — §3/§5 simply
  benchmarked the worst (pathological) point.
- A genuine latent bug exists in the **uncommitted manifold port** of the companion repo:
  with `manifold=` set the STLOG is 9×9 (tangent), but the config slices with ambient
  indices `[0,1,2,7,8,9]`; index 9 is out of bounds and JAX **silently clamps 9→8**,
  duplicating a row/col → an *artificially* singular Gramian. rt-oac defaults to the ambient
  10×10 Gramian and is unaffected, but Phase 1d (manifold) must use tangent indices
  `[0,1,2,6,7,8]`. **Flagged for the companion repo (the X-IO branch).**

The measurements are re-run at a **non-pathological** config in §6.

## 3. (MUC config) The objective is degenerate there (zero gradient)

At the operating points, the sliced (position+velocity) 6×6 Gramian is **numerically
singular**: min-eigenvalue ≈ 0 across all 20 horizon nodes (`sum(min_eig) ≈ -1.3e-13`).
Consequently the baseline `eps/(Σσ_min+eps)` objective has an **autodiff gradient norm of
~1e-13 (i.e. zero)** at the reference. The result:

| solver on baseline σ_min | outcome |
|---|---|
| trust-constr | 40 iters, 10.4 s, ends at obj=1 — *no improvement* (thrashes on the flat/degenerate landscape) |
| SLSQP | 1 iter, 0.013 s — correctly detects ~0 gradient and quits, obj=1 |

Both end at the worst objective value. The companion's historical `fun_hist` is likewise
flat. **This — not raw solver speed — is why the OAC solve is ineffective: there is no
usable gradient to follow.**

## 4. Smooth surrogates restore the gradient and let a lean solver climb

Replacing the non-smooth min-singular-value with a smooth surrogate (injected via
`ObservabilityCost(gramian_metric=...)`, no core fork — see `src/rt_oac/metrics.py`):

| metric | grad-norm @ ref | SLSQP | true Σ(min-eig): ref → after |
|---|---|---|---|
| baseline σ_min | 1.0e-13 | 1 iter | ~0 → ~0 |
| **neg_logdet** | **7.77** | 60 iter, 1.2 s | ~0 → **7.7e-9** |
| neg_softmin_eig | 0.26 | 60 iter, 1.1 s | ~0 → 3.0e-9 |

`log-det = Σ log(λ_i+ε)` has gradient `1/(λ_i+ε)`, which is *largest* exactly where an
eigenvalue is near zero — so it actively lifts the degenerate direction, which the σ_min
objective cannot. SLSQP (~13 ms/iter vs trust-constr's ~265 ms/iter) follows it and raises
the true min-eigenvalue off zero in ~1.2 s.

Caveat: on the *sliced (r,v)* objective the achievable min-eig is only ~1e-9 vs max-eig
~162 — still extremely ill-conditioned, suggesting that objective is partly *structurally*
degenerate near hover. The principled alternative (full tangent-space manifold Gramian) is
tested in §5.

## 5. Manifold (tangent-space 9×9) Gramian objective

The principled objective (full tangent-space Gramian, quaternion gauge removed, no slicing)
behaves **identically** to the sliced one:

| metric (manifold 9×9) | grad-norm @ ref | SLSQP | true Σ(min-eig): ref → after |
|---|---|---|---|
| baseline σ_min | 0.0 (exactly) | 1 iter | ~0 → ~0 |
| neg_logdet | 7.81 | 60 iter, 1.6 s | ~0 → 2.0e-8 (per-node min-eig ~6e-11…3e-9) |

Reference min-eig ≈ 0, max-eig ≈ 162 — the full observable-state Gramian is **also**
near-singular at the operating point. So the degeneracy is **structural / physical**, not an
artifact of slicing to (r,v): at near-hover, a relative-state direction is genuinely
(near-)unobservable from range+attitude over the 0.2 s STLOG window. Surrogates lift it the
same tiny amount (~1e-9) either way.

→ The objective-formulation choice (sliced vs manifold) does **not** resolve the
degeneracy. What may resolve it: (a) operating from non-degenerate configurations (once the
follower is maneuvering), (b) a longer observation horizon to excite slow modes, or
(c) accepting the small lift and verifying via closed-loop EKF whether it suffices. The
arbiter is the closed-loop estimation experiment, not the instantaneous Gramian.

## 6. Re-measurement at NON-pathological configs (the decisive result)

At non-MUC initial states (tangential relative velocity + non-identity attitude), scoring
the **accumulated** Gramian `Σ_k W_k` (`#observable directions` out of 6, accumulated
min-eig, accumulated log-det):

| config | objective | grad-norm | SLSQP | #obs (acc) | acc-min-eig | acc-logdet |
|---|---|---|---|---|---|---|
| tangential+att | paper min-eig | 3e-15 | 1 iter | 4→4 | ~0 | 2.1→2.1 |
| tangential+att | **neg_logdet** | **14.9** | 1.1 s / 58 | **4→6** | **1.70** | **2.1→29.5** |
| tangential+att | neg_softmin | 0.40 | 1.3 s / 80 | 4→6 | 2.45 | 2.1→28.7 |
| thesis-rec | paper min-eig | 0 | 0.75 s / 27 | 4→4 | 5e-5 | 5.9→8.7 |
| thesis-rec | **neg_logdet** | **12.2** | 2.4 s / 79 | **4→6** | **10.6** | **5.9→37.3** |
| thesis-rec | neg_softmin | 6e-3 | 2.4 s / 80 | 4→6 | 3.90 | 5.9→33.9 |

**`log-det` drives the trajectory to FULL observability (4→6 directions) in ~1–2 s; the
paper's per-node min-eigenvalue objective gains nothing (4→4, gradient ~0).** The min-eig
objective is floor-limited by the `T^(2r*+1)=T^11` short-time scaling and is therefore a
poor numerical target; log-det is smooth, not floor-limited, and rewards lifting every
near-zero eigenvalue. Per-node log-det (summed over the horizon) also improves the
*accumulated* trajectory observability, so it is the right objective for the real-time OAC.

**Decision (data-driven):** real-time OAC = **non-MUC initial config + per-node log-det
objective + SLSQP + warm-start**, validated by closed-loop EKF accuracy. Cold-start SLSQP is
~1–2 s for ~60–80 iters (~19 ms/iter); warm-starting to a handful of iters targets ≤100 ms.

## 7. Real-time achieved (closed-loop, true-state rollout)

Receding-horizon rollout from a non-MUC initial state, log-det objective + SLSQP, applying
the first control and advancing the true relative dynamics each step
(`experiments/closed_loop_rollout.py`). Scoring observability by #directions of the
accumulated `Σ_k W_k` above a relative threshold:

| maxiter | median wall (cold / warm) | observability over run |
|---|---|---|
| 80 | 1501 ms / 1004 ms | 6/6 (full) |
| 12 | 237 ms / 218 ms | 6/6 (full) |
| **6** | **103 ms / 124 ms** | **6/6 (full)** |

**The 10 Hz / 100 ms target is reached.** Critically, observability stays **6/6** even at
maxiter=6 — the iterations beyond ~6 only refine eigenvalue *magnitudes*, not the set of
observable directions. Versus the original ~10 s degenerate solve, this is a ~100× speedup
to a *functional* (full-observability) controller.

**Honest caveat:** "#observable directions" is a coarse proxy (eigenvalue count above a
relative threshold), not estimation precision. Early stopping leaves log-det magnitude on
the table; whether that degrades actual **EKF estimation accuracy** is unverified — that is
the rigorous arbiter and requires the closed-loop EKF experiment (the quadrotor needs a
quaternion/error-state EKF; the planar unicycle EKF is already wired in the companion).

**Remaining headroom to comfortably clear 100 ms:** per-iter is ~17 ms, but the JAX gradient
is only ~4 ms, so ~13 ms/iter is scipy SLSQP overhead. A JAX-native solver (or a learned
warm-start giving a near-optimal first guess) would drop well under 100 ms with margin.

## Conclusion / revised priorities

The plan's ROI ordering assumed STLOG/gradient cost dominated. It does not. The evidence
re-prioritizes:

1. **Reframe the objective (was Phase 2) → now primary.** A smooth, non-degenerate
   surrogate is the prerequisite for *any* solver to make progress. Choose the objective
   formulation (sliced σ_min surrogate vs. full manifold Gramian) — see §5.
2. **Lean solver + warm-start (Phase 1/2).** SLSQP already ~20× leaner per-iteration than
   trust-constr; warm-starting cuts the always-maxed iteration count further.
3. **STLOG acceleration / structure (was Phase 1d/3): deprioritized** — compute is only 6%.
4. **Learning / transcription:** unchanged ordering, gated on the above.

Target remains 10 Hz / ≤100 ms on CPU. With surrogate + SLSQP at ~1.2 s for 60 cold-start
iters, warm-starting to a handful of iters should approach the target; the actual compute
floor for a converged solve is ~0.7 s of JAX today, mostly cuttable via fewer iters.
