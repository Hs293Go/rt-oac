# RT-OAC: real-time observability-aware control — results

Answering the JGCD "NEW" open problem: solving the observability-aware control (OAC) problem
in real time. Full method/measurement detail in `phase0_findings.md`; this is the summary.

## Outcome

**The OAC solve went from ~10 s (and not even functional) to ~100 ms while *improving*
closed-loop estimation accuracy.** Three evidence-driven changes did it — none of which was
"make the STLOG faster," the assumption the original plan started from.

## What the profiling proved (Phase 0)

1. **The bottleneck is the optimizer, not the observability math.** Per solve, the STLOG /
   gradient / constraint cost is only ~6%; scipy `trust-constr`'s own internals were ~94%.
   STLOG eval is 52 µs; the gradient 4 ms. → accelerating the STLOG is pointless.
2. **The paper's per-node minimum-eigenvalue objective is numerically degenerate.** Its
   value is floored by the short-time scaling `λ_min ~ T^(2r*+1)` (`T^11` at order 5), so its
   autodiff gradient is ~0 and *no* gradient solver can make progress on it.
3. **The apparent "10 s / flat objective" was the co-hovering MUC**, a manifestly
   unobservable configuration (thesis ch.4), not a regression. (A separate latent bug was
   found and flagged in the companion's uncommitted manifold port: a 9×9 tangent Gramian
   sliced with ambient indices `[…,9]` → JAX clamps the OOB index → artificially singular.)

## The fix

| change | effect |
|---|---|
| objective: per-node **log-det** instead of min-eig | smooth, not `T^11`-floored; gradient is *largest* where eigenvalues are near zero, so it actively lifts weak directions. Drives a non-MUC trajectory to **full observability (4→6 directions)**. |
| solver: **SLSQP** instead of trust-constr | ~18 ms/iter vs ~265 ms/iter (trust-constr thrashed on the degenerate landscape). |
| **early stopping** (maxiter ≈ 6) | iterations past ~6 only refine eigenvalue magnitudes, not the set of observable directions. |
| operate from a **non-MUC** initial config | tangential relative velocity + non-identity attitude (avoids the 4 thesis MUCs). |

## Real-time (quadrotor, order-5, closed-loop true-state rollout)

| maxiter | wall/solve | observability |
|---|---|---|
| 80 | 1.0–1.5 s | 6/6 |
| 12 | ~0.22 s | 6/6 |
| **6** | **~0.10 s** | **6/6** |

→ the 10 Hz / 100 ms guidance target is met while maintaining full observability.

## Estimation-accuracy validation (planar leader-follower EKF, 8 seeds)

Closed-loop EKF that *carries its own estimate* (controller sees only the estimate; truth
evolves separately; error accumulates). Follower position is observable only via the
inter-robot range. Large initial follower-position error (std 2 m).

| condition | foll-pos RMSE | final err | terminal cov(foll) | solve |
|---|---|---|---|---|
| no-OAC (drive straight) | 1.53 ± 0.87 | 1.52 | 1.5e-1 | — |
| OAC-full (maxiter 60) | 0.64 ± 0.59 | 0.12 | 7.6e-3 | 4 ms / 10 it |
| **OAC-fast (maxiter 6)** | **0.68 ± 0.68** | **0.11** | **8.0e-3** | **3 ms / 6 it** |

* **OAC works:** no-OAC cannot correct the unobservable initial error (final ≈ initial);
  OAC drives final error down ~13× and RMSE ~2.4×, with ~20× tighter covariance.
* **The fast real-time solve preserves the full benefit:** OAC-fast ≈ OAC-full on every
  metric. Early stopping is safe — the key real-time result, confirmed on estimation accuracy.

## Status & what's left

Done: repo scaffold; Phase 0 profiling; the reframed objective (Phase 2); the lean solver +
warm-start + early-stop (Phase 1); planar closed-loop EKF validation (Phase 7, planar).

Open / optional: quadrotor quaternion (error-state) EKF to validate estimation accuracy on
the real target (deferred — planar validated the methodology); a JAX-native solver and/or
learned warm-start to push under 100 ms with margin / reach the 50 Hz stretch (not needed for
the 10 Hz target); multiple-shooting transcription (Phase 3, deprioritized — compute is 6%).
