# Toward Real-Time Observability-Aware Control: A Diagnostic Study and a Reformulated Objective

**Working report — RT-OAC investigation.** Companion code: `~/python-scripts/rt-oac`.
All measurements were taken on a 16-core CPU host (`JAX_PLATFORMS=cpu`, float64), reusing
the verified STLOG / Lie-derivative / integrator / model implementations from the
`observability_aware_control` companion repository. Raw data and scripts are in
`benchmarks/`, `experiments/`, and `results/`.

---

## Abstract

We study the open problem of solving the observability-aware control (OAC) problem fast
enough for real-time, receding-horizon execution, which the parent work currently performs
only offline. Through systematic profiling we establish that, contrary to the prevailing
assumption, the computational bottleneck is **not** the evaluation of the Short-Time Local
Observability Gramian (STLOG): the entire observability computation accounts for ~6% of
solve time, while the general-purpose nonlinear solver accounts for ~94%. We further find
that the objective used in the parent work — the sum over the horizon of the minimum
eigenvalue of the per-node Gramian — is *numerically degenerate*: its value is floored by
the short-time scaling \(\lambda_{\min}\sim T^{2r_*+1}\), and its gradient is effectively
zero, so no gradient-based solver can make progress. Replacing it with a smooth **log-det**
surrogate restores a strong gradient and drives trajectories to full observability. Combined
with a lightweight solver (SLSQP) and early termination, the per-solve wall time on the
range-only quadrotor problem drops from ~10 s to ~100 ms (a 10 Hz guidance rate) while
maintaining full observability. A closed-loop Extended Kalman Filter (EKF) evaluation on a
planar leader-follower analogue confirms that the fast, early-terminated controller reduces
follower-position estimation error by ~13× (final) and posterior covariance by ~20× relative
to a non-OAC baseline, and that early termination preserves the full estimation benefit. The
controller reproduces the theoretically expected observability-optimal *orbiting* behavior.

---

## 1. Problem statement

The parent work designs control inputs that improve the observability of a leader-follower
quadrotor cooperative-localization system from range and relative-attitude measurements. The
relative state is
\[
\mathbf{x} = [\,\mathbf{r}^\top,\ \mathbf{q}^\top,\ \mathbf{v}^\top\,]^\top \in \mathbb{R}^{10},
\quad \mathbf{r}\in\mathbb{R}^3,\ \mathbf{q}\in\mathcal{S}^3,\ \mathbf{v}\in\mathbb{R}^3,
\]
with input \(\mathbf{u}\in\mathbb{R}^8\) (leader and follower thrust + body rates) and output
\(\mathbf{y}=[\tfrac12 d^2,\ \mathbf{q}^\top]^\top\in\mathbb{R}^5\), \(d=\lVert\mathbf{r}\rVert\).
Observability over a short horizon \(T\) is quantified by the \(r\)-th order STLOG
\[
\mathbf{W}^{(r)} = \sum_{i,j=0}^{r}\frac{T^{\,i+j+1}}{(i+j+1)\,i!\,j!}
\, D(L_\mathbf{f}^i \mathbf{h})^\top \boldsymbol{\Sigma}^{-1} D(L_\mathbf{f}^j \mathbf{h}),
\]
evaluated at each node of a predicted trajectory. The observability-predictive controller
(OPC) maximizes a metric of \(\{\mathbf{W}^{(r)}_k\}_{k=0}^{N-1}\) over a receding horizon
subject to input and inter-vehicle-distance constraints. With \(r=5\), \(T=0.2\,\mathrm{s}\),
\(N=20\), the OPC solve is too slow for online thrust/rate generation and is used offline.

**Objective of this study:** identify what actually makes the solve slow and ineffective,
and make it real-time.

---

## 2. Method

We constructed a focused testbed (`rt-oac`) that imports the companion's verified numerics
unchanged and adds: (i) a profiling harness that attributes solve time across components;
(ii) a library of smooth surrogate objectives injected through the cost's existing
`gramian_metric` hook; (iii) a controller supporting a selectable solver, warm-starting, and
early termination; and (iv) a closed-loop EKF evaluation. All timed quantities are
JIT-compiled and synchronized with `block_until_ready`; reported wall times are steady-state
(post-compilation). A persistent XLA cache amortizes the one-time ~200 s compilation of the
order-5 graph.

---

## 3. Results

### 3.1 The optimizer, not the observability computation, is the bottleneck

Component micro-benchmarks (Table 1) show the observability machinery is cheap once
compiled. A single STLOG evaluation costs **52 µs** (vs. ~441 ms *un-compiled* — a ~6500×
penalty that explains earlier folklore that "the STLOG is expensive"). A full objective
evaluation over the \(N=20\) horizon is 0.5 ms; its reverse-mode gradient 4 ms.

**Table 1 — per-component cost (order 5, \(N=20\)).**

| component | cost |
|---|---|
| STLOG evaluation (one node, JIT) | 52 µs |
| objective (rollout + 20× STLOG + metric) | 0.5 ms |
| gradient (reverse mode) | 4.0 ms |
| distance constraint | 0.035 ms |
| constraint Jacobian | 0.31 ms |

Attributing one full `trust-constr` solve (40 iterations) shows **0.68 s (6%) inside the
JAX callables and 10.6 s (94%) inside the solver's own Python/linear-algebra internals**. We
verified there is no recompilation across solves and no measurable NumPy↔JAX boundary
penalty (the controller's methods cost the same on NumPy or JAX inputs). The implication is
that accelerating the STLOG cannot materially help; the solver must change.

### 3.2 The minimum-eigenvalue objective is numerically degenerate

At the operating points of interest the per-node Gramian is numerically singular: the
minimum eigenvalue of the sliced (position, velocity) \(6\times6\) Gramian is
\(\sim 10^{-13}\) across all horizon nodes, and the same holds for the full tangent-space
\(9\times9\) Gramian (quaternion gauge removed). Consequently the parent objective
\(\sum_k \lambda_{\min}(\mathbf{W}_k)\) has an autodiff **gradient norm of \(\sim10^{-13}\)
(effectively zero)**. The cause is intrinsic: the worst-observed direction enters only
through high-order Lie derivatives, whose STLOG weight scales as \(T^{2r_*+1}=T^{11}\) at
order 5, i.e. \(\sim2\times10^{-8}\) — below the noise floor.

This degeneracy, not solver speed alone, is why the solve is ineffective. Given a zero
gradient, `trust-constr` exhausts its 40-iteration budget without improving the objective,
while `SLSQP` correctly detects the flat landscape and terminates in one iteration; **both
end at the worst objective value.** This also reconciles an apparent ~25× "regression"
(0.4 s → 10 s): the slow case is simply the controller thrashing on the degenerate
landscape of a *manifestly unobservable configuration* (MUC) — here co-hovering / straight
co-motion — which the parent thesis enumerates explicitly. The MUC is expected behavior, not
a fault. (A separate latent defect was found in an in-progress manifold port: a \(9\times9\)
tangent Gramian sliced with ambient indices clamps an out-of-bounds index, fabricating a
singular Gramian; flagged for repair.)

### 3.3 A smooth surrogate restores the gradient and full observability

We replace the metric with a per-node negative log-determinant,
\(\;J = -\sum_k \sum_i \log(\lambda_i(\mathbf{W}_k)+\varepsilon)\). Because
\(\partial \log(\lambda+\varepsilon)/\partial\lambda = (\lambda+\varepsilon)^{-1}\) is
*largest* exactly where \(\lambda\to0\), log-det actively lifts the weak directions that the
min-eigenvalue objective cannot. At a non-MUC configuration (tangential relative velocity,
non-identity relative attitude), measuring observability by the number of well-conditioned
directions of the *accumulated* Gramian \(\sum_k \mathbf{W}_k\) (Table 2):

**Table 2 — objective comparison at a non-MUC config (SLSQP).**

| objective | grad-norm | observable directions (of 6) | accumulated min-eig |
|---|---|---|---|
| min-eigenvalue (parent) | \(\sim10^{-15}\) | 4 → 4 (no change) | \(\sim0\) |
| **negative log-det** | **12–15** | **4 → 6 (full)** | **1.7–10.6** |
| negative soft-min | 0.4 | 4 → 6 | 2.4–3.9 |

Log-det converts a flat, intractable problem into one a lightweight solver drives to full
observability.

### 3.4 Real-time solve via a lightweight solver and early termination

Replacing `trust-constr` (≈265 ms/iteration on this problem) with `SLSQP` (≈18 ms/iteration)
and observing that **iterations beyond ~6 only refine eigenvalue magnitudes, not the set of
observable directions**, we early-terminate. In a true-state receding-horizon rollout of the
order-5 quadrotor problem from a non-MUC state (Table 3):

**Table 3 — quadrotor closed-loop solve time vs. iteration cap.**

| iteration cap | median wall / solve | observable directions |
|---|---|---|
| 80 | 1.0–1.5 s | 6/6 |
| 12 | 0.22 s | 6/6 |
| **6** | **0.10 s** | **6/6** |

The 10 Hz guidance target (≤100 ms) is met while maintaining full observability — a ~100×
improvement over the original degenerate ~10 s solve.

### 3.5 Closed-loop estimation accuracy (planar EKF)

To test the decisive question — whether the fast, early-terminated controller preserves
*estimation quality*, not merely a coarse observability count — we used a planar
leader-follower analogue for which a quaternion-free EKF is available. The leader drives
straight; the follower's position is observable only through the inter-vehicle range. We run
a closed loop in which the EKF carries its own estimate (the controller sees only the
estimate; ground truth evolves independently; error accumulates), with a large initial
follower-position error (2 m std), averaged over 8 seeds (Table 4).

**Table 4 — planar closed-loop EKF (8 seeds; follower-position error).**

| condition | RMSE (mean ± std) | final error | terminal cov. | solve |
|---|---|---|---|---|
| no-OAC (drive straight) | 1.53 ± 0.87 | 1.52 | 1.5×10⁻¹ | — |
| OAC, full (cap 60) | 0.64 ± 0.59 | 0.12 | 7.6×10⁻³ | 4 ms / 10 it |
| **OAC, fast (cap 6)** | **0.68 ± 0.68** | **0.11** | **8.0×10⁻³** | **3 ms / 6 it** |

Without OAC the estimator cannot reduce the unobservable initial error (final ≈ initial);
with OAC the final error falls ~13×, RMSE ~2.4×, and posterior covariance ~20×. Crucially,
**the fast controller (cap 6) matches the fully converged one on every metric** — early
termination is safe with respect to estimation accuracy.

### 3.6 Emergent observability-optimal behavior

The log-det controller reproduces the textbook observability-optimal maneuver. In a planar
rollout the follower sweeps a cumulative relative bearing of **−1613° (≈4.5 revolutions)**
around the leader — i.e. it **orbits** — pressing against the minimum allowed separation
(distance bottoms at the 0.20 m bound) to maximize bearing rate and hence information, while
the non-OAC follower sweeps 0° (Figure: `results/planar_trajectory.png`). This is consistent
with the parent theory and with the known caveat that pure observability objectives, absent a
tracking term, produce operationally awkward (orbiting) trajectories.

---

## 4. Discussion

The study reframes the path to real-time OAC. The dominant levers are, in order:
(1) **objective reformulation** — log-det restores a usable gradient that the parent
min-eigenvalue objective lacks at the relevant operating regime; (2) **a lightweight
solver** — most of the original solve time was general-purpose nonlinear-programming
overhead, not problem-specific computation; (3) **early termination** — the observability
content of a solution saturates within a handful of iterations. STLOG acceleration and
problem transcription, prominent in the initial plan, proved largely irrelevant because the
observability computation is only ~6% of the cost.

**Limitations.** (i) The real-time quadrotor result is validated against an
observable-direction count, a coarse proxy; the rigorous estimation-accuracy validation was
performed on the planar analogue, not the quaternion quadrotor (an error-state EKF for the
latter is future work). (ii) Measurements are single-host, CPU-only. (iii) "Observable
directions" thresholds eigenvalues relative to the largest; absolute conditioning remains
poor (the \(T^{11}\) floor), which is why log-det rather than min-eigenvalue is the
appropriate target. (iv) Operating away from MUCs is assumed; escaping MUCs is out of scope.

---

## 5. Conclusion and future work

Real-time observability-aware control is achievable for the studied system: a smooth log-det
objective, a lightweight early-terminated solver, and operation away from manifestly
unobservable configurations reduce the solve from ~10 s to ~100 ms (10 Hz) while a
closed-loop EKF confirms the fast controller retains the full estimation benefit (~13×
lower final error than no OAC). The headline scientific finding is that the parent
min-eigenvalue objective is numerically degenerate at the relevant regime and should be
replaced by log-det for any gradient-based real-time use.

Future work: (1) an error-state/manifold EKF to validate estimation accuracy on the
quaternion quadrotor; (2) a JAX-native solver to remove the residual ~13 ms/iteration of
solver overhead (the gradient is only 4 ms) and target ≥50 Hz; (3) a learned warm-start to
approach single-iteration solves; (4) a hybrid tracking + observability objective to temper
the orbiting behavior.
