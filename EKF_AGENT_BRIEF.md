# Agent Brief — Quaternion / Manifold-Aware Error-State EKF (RT-OAC task #8)

## Goal
Build a proper **error-state EKF** for the 10-state relative-pose drone system and show it
makes the drone closed-loop **consistent** (estimation error stays within the 3σ envelope,
no divergence). This replaces the placeholder Euclidean `SimpleEKF`, which diverges on the
quaternion state (we observed final position error ~7 m while its 3σ was only ~0.6 m —
wildly overconfident — and the follower flew to 16 m, violating the distance bound).

## Where / how to work
- Repo: `~/python-scripts/rt-oac`. Run everything with `JAX_PLATFORMS=cpu uv run python ...`.
- `import rt_oac` first in any script — it puts the companion source on the path and enables
  JAX x64 (mandatory for this work).
- The companion repo `observability_aware_control` is an **editable dependency**; its modules
  import normally. **Do NOT modify the companion repo's working tree** (it holds the user's
  WIP). Put all new code in `rt-oac` (`src/rt_oac/` and `experiments/`).
- Comply with pre-commit (ruff v0.9.0 lint+format): keep `src/` lines ≤88, full docstrings;
  `experiments/` has relaxed per-file ignores. Run `uvx ruff@0.9.0 check . && uvx ruff@0.9.0 format .`.

## System
- State `x = [r(3), q(4 xyzw), v(3)] ∈ ℝ³ × S³ × ℝ³` (ambient dim 10). Input `u ∈ ℝ⁸`.
- Observation `h(x) = [½‖r‖², q] ∈ ℝ⁵` (squared range + relative quaternion).
- Model: `example_lib.models.inter_quadrotor_pose` — `dynamics(x,u)`, `observation(x)`, and
  `MANIFOLD` = `manifold.Manifold([EuclideanBlock(3), SO3Block(), EuclideanBlock(3)])`,
  **tangent_dim = 9**.
- Manifold API (read `example_lib/manifold.py`): `boxplus(x, ξ)` retracts a 9-vector tangent
  increment (`q ← q ⊗ Exp(δθ)`); `plus_jacobian(x)` is the 10×9 ambient←tangent differential;
  `to_tangent(x, ẋ_ambient)` maps an ambient rate to the body/tangent rate. Reuse the SO(3)
  `Exp`/`Log`/quaternion-multiply helpers in `example_lib/math/` and `manifold.py` — don't
  hand-roll them.
- Integrator: `observability_aware_control.integrator.Integrator(..., manifold=MANIFOLD)`
  advances attitude by the exponential map.
- Placeholder to replace/compare against: `example_lib.misc.simple_ekf.SimpleEKF`
  (Euclidean `operator.sub`/`add`). The diverging closed-loop is `experiments/drone_ekf_plot.py`.

## Deliverable 1 — `src/rt_oac/error_state_ekf.py`
`class ErrorStateEKF` carrying ambient state `x∈ℝ¹⁰` and tangent covariance `P∈ℝ⁹ˣ⁹`:

- **predict(x, P, u, dt) → (x⁺, P⁺):**
  - `x⁺` = one manifold integrator step (or `boxplus(x, dt·to_tangent(x, f(x,u)))`).
  - Error-transition `F (9×9)` = `jacobian_{δx} [ boxminus(step(boxplus(x,δx),u,dt), x⁺) ]` at
    `δx=0` (use `jax.jacobian`; you'll need a `boxminus` = inverse of `boxplus`, i.e. tangent
    difference; SO(3) part is `Log(q_a⁻¹ ⊗ q_b)`).
  - `P⁺ = F P Fᵀ + G Q Gᵀ` with input/process noise `Q` (8×8 from input variances) mapped
    through `G = ∂δx⁺/∂u`, or an additive tangent process noise — your call, document it.
- **update(x, P, y) → (x⁺, P⁺):**
  - Innovation `ν = [ y_range − ½‖r̂‖² ; Log(q̂⁻¹ ⊗ q_meas) ] ∈ ℝ⁴` (range residual + 3-vector
    attitude residual via the manifold log — NOT a raw quaternion subtraction).
  - `H (4×9)` = `jacobian_{δx}` of that residual map at `δx=0`.
  - `R = diag(range_var, att_var, att_var, att_var)` (4×4); use the config noise
    (`range_var≈1e-2`, `att_var≈1e-2`).
  - `K = P Hᵀ (H P Hᵀ + R)⁻¹` (9×4); `x⁺ = boxplus(x, K ν)`; `P⁺` = Joseph form
    `(I−KH)P(I−KH)ᵀ + K R Kᵀ`.
- Keep the quaternion unit-norm by construction (updates go through `boxplus`, never Euclidean
  add). `jax.jit` the predict/update where reasonable.

## Deliverable 2 — `experiments/drone_ekf_eval_esekf.py`
Re-run the drone closed loop with the ErrorStateEKF in place of `SimpleEKF` (controller acts
on the **estimate**, frontier method = log-det + SLSQP@6 + distance constraint, non-MUC start
`[2,0,0, quat_z(20°), 0,1,0]`, ~60–100 steps). Plot: relative trajectory; relative-position
error vs 3σ; inter-drone distance vs bounds. Print final error, final 3σ, and a NEES-style
consistency number over the run. Mirror `experiments/drone_ekf_plot.py` for structure.

## Acceptance criteria
1. **No divergence:** relative-position error stays bounded over the run (contrast: placeholder
   hit ~7 m).
2. **Consistency:** the error stays within (or near) the 3σ envelope — the covariance is NOT
   collapsed to ~0 while the error grows. Report NEES; it should sit roughly in the
   chi-square band, not blow up.
3. **Unit quaternion:** `‖q̂‖` stays 1 to machine precision throughout.
4. **Distance respected** once the estimate is good (no 16 m fly-away like the placeholder).

## Pitfalls
- Don't slice a 9×9 tangent Gramian with ambient indices `[…,9]` — JAX silently clamps the
  OOB index (a known bug elsewhere in this codebase). The EKF is separate, but stay consistent
  in tangent (9-d) coordinates throughout.
- Validate `Exp/Log/⊗` conventions (xyzw vs wxyz; active vs passive) against the model's own
  `observation`/`dynamics` so the residual sign/frame is correct — a wrong convention silently
  ruins the update.
- Start small: unit-test predict/update on a 1-step toy before the full closed loop.
