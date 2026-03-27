# Context: Fish/Spring-Mass Jerk-GNN — Model Specification & Loss Design

## Background
Building a GNN that predicts jerk (d³x/dt³) as the sole learned quantity, with hard consistency integration for a, v, x. Synthetic testbed: 6-body spring-mass. Real target: collective fish dynamics (SLEAP tracking, 4 stickleback, 25fps). Only position x is observed; v, a, j estimated via finite differences.

---

## 1. What happens when you underspecify derivative order

### Missing terms get absorbed into lower-order coefficients
- Network minimizes loss regardless — missing jerk term gets folded into biased velocity/acceleration estimates
- Learned "acceleration" is no longer acceleration — it's acceleration + some jerk-like compensation
- **Destroys interpretability**, which is a primary goal

### Structured residuals
- Errors are not random — they correlate with the missing derivative order
- Diagnostic: if residuals correlate with jerk, model is underspecified
- Plot rollout error vs instantaneous speed/turn rate — should spike during high-jerk events

### Rollout drift
- Lower-order terms are biased to compensate, so errors accumulate directionally rather than cancelling
- Heavy-tailed, non-stationary error distribution

### Coefficient instability across dt
- Strong diagnostic: train at two different dt values — if learned coefficients change substantially, model is absorbing missing terms
- Correctly specified model's coefficients should be dt-invariant

---

## 2. Empirical findings on the fish data

- **Changes in acceleration are the hardest to predict**
- **Jerk correlates with relative velocity of neighbors** (vⱼ - vᵢ) — social structure confirmed
- **LASSO did not select jerk** as a useful predictor of Δx

### Why LASSO and social analysis disagree — they're asking different questions
- LASSO: does the jerk Taylor term (∝ dt³ ≈ 6×10⁻⁵ at 25fps) improve single-step position prediction? → No, it's invisible at this timescale
- Social analysis: does jᵢ correlate structurally with vⱼ - vᵢ? → Yes
- These are **compatible**: jerk doesn't help predict x(t+dt) directly, but it IS the governing equation of social forcing

### Key reframe
Jerk is not a Taylor correction term here — it is the **governing equation** of the social dynamics:

```
j_i = f_θ({vⱼ - vᵢ}, {xⱼ - xᵢ})
```

The network's job is to predict how acceleration changes as a function of relative neighbor state. Position rollout follows from integrating this.

---

## 3. Threshold / event-triggered hypothesis

Fish likely respond to neighbors crossing a proximity or speed threshold — not continuous smooth forcing. Evidence:
- Changes in acceleration are bursty, not smooth
- Social jerk correlation suggests interaction is context-dependent
- Prediction difficulty is likely event-triggered (high-jerk moments)

### Diagnostics to check
- **Jerk distribution**: heavy-tailed or bimodal → threshold/spike process, not continuous
- **Conditional analysis**: bin timesteps by nearest-neighbor distance, check if r(jᵢ, vⱼ-vᵢ) is strong at close range and near zero at distance
- **Synchronous jerk spikes**: plot mean |j| across fish over time — bursts = collective events

---

## 4. Spring-mass modifications (simplest first)

### Stage 1 — Distance-gated springs (most important, do first)
```python
F_ij = k_ij * (x_j - x_i - l_ij)  if |x_j - x_i| < r*  else 0
```
Analytic jerk becomes:
```
j_i = (1/m_i) * Σⱼ k_ij*(vⱼ - vᵢ) * 1[|xⱼ - xᵢ| < r*]
```
- Produces sparse, event-triggered jerk
- Tests whether GNN can recover threshold r* from data
- Minimal change to existing simulator

### Stage 2 — Velocity-dependent (viscous) coupling
```python
F_ij = k_ij*(x_j - x_i - l_ij) + c_ij*(v_j - v_i)
```
Jerk now depends on relative acceleration:
```
j_i = (1/m_i) * Σⱼ [k_ij*(vⱼ-vᵢ) + c_ij*(aⱼ-aᵢ)]
```
- Tests whether adding (aⱼ - aᵢ) as edge feature improves predictions

### Stage 3 — Two-timescale dynamics
```python
F_i = -κ*x_i  +  Σⱼ k_ij*(x_j - x_i) * 1[|xⱼ - xᵢ| < r*]
#      slow           fast (social)
```
- Separates individual swimming dynamics (slow) from social collision avoidance (fast)
- Can train two sub-networks and test decomposability

### Stage 4 — Soft threshold (smooth gating)
```python
F_ij = k_ij*(x_j - x_i - l_ij) * sigmoid((r* - |x_j - x_i|) / τ)
```
- Smooth analog of hard gate; GNN can learn this as attention weight over edges
- More realistic for fish

### Stage 5 — Stochastic forcing
```python
m_i * x_i'' = F_i_social + η_i(t)   # η = colored noise
```
- Tests separation of social forcing from intrinsic stochasticity

---

## 5. Loss function

### Problem with pure jerk loss
- Dominated by rare large-jerk events (threshold crossings)
- Most timesteps have near-zero jerk → flat loss surface, poor gradients

### Problem with pure position loss
- Jerk contributes O(dt³) to position error — tiny at 25fps
- Gradient signal to jerk network is near zero
- Social jerk term gets undertrained (same reason LASSO missed it)

### Recommended: normalized multi-term loss

$$\mathcal{L} = \frac{\|x_{pred} - x_{true}\|^2}{\sigma_x^2} + \lambda_a \frac{\|a_{pred} - a_{true}\|^2}{\sigma_a^2} + \lambda_j \frac{\|j_{pred} - j_{true}\|^2}{\sigma_j^2}$$

- **Position term**: trains slow dynamics
- **Acceleration term**: directly supervises the hardest-to-predict quantity, gives jerk network strong gradient without needing to propagate through dt³
- **Jerk term**: trains social interaction directly
- Normalize by variance so all three terms are dimensionless O(1) — λ_a and λ_j become interpretable as relative importance weights, not scale corrections

### Practical recommendation
Start with the normalized multi-term loss without event weighting. Add event weighting (upweight jerk loss near threshold crossings) only after basic architecture is confirmed working — requires knowing or estimating r*.

### Where to get a_true and j_true
- Synthetic: analytically from simulator
- Real fish: finite differences on observed positions (noisy but usable as training signal)

---

## 6. Immediate next steps (simplest first)

1. Add **distance-gated springs** to spring-mass simulator
2. Switch to **normalized multi-term loss** (position + acceleration + jerk)
3. Confirm GNN learns sparse event-triggered jerk on synthetic data
4. Check whether learned edge activations recover proximity threshold r*
5. Then move to fish data
