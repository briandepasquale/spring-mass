# Jerk-GNN: Design Summary

## Problem
Predict dynamics of an unknown physical system where only position `x` is observed. Use a spring-mass system (6 bodies) as a synthetic testbed where ground truth is available.

## Core Idea
Explicitly model **jerk** `j = d³x/dt³` as the learned quantity, then derive `a`, `v`, `x_next` via analytic integration (**hard consistency**). This keeps the network's learned output physically interpretable.

Jerk is not merely a Taylor correction term. It is the **governing equation** of the interaction forcing:

```
j_i = (1/m_i) * Σ_j k_ij * (v_j - v_i)
```

The network learns how acceleration changes as a function of relative neighbor state. Position rollout follows from integrating this — not from fitting position directly.

## Integration Scheme
From constant-jerk Taylor expansion:

```
x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt² + (1/6)*j(t)*dt³
v(t+dt) = v(t) + a(t)*dt + 0.5*j(t)*dt²
a(t+dt) = a(t) + j(t)*dt
```

Truncation error is O(dt⁴) — favorable for small dt.

## Architecture: GNN
- **Nodes**: each body, features = (x history window, v₀, a₀)
- **Edges**: pairwise connections (known topology in synthetic case; learned in real)
- **Message passing**: relative state between neighbors → learned messages
- **Output**: scalar jerk `j_i` per node (2D for fish)
- `a`, `v`, `x_next` computed deterministically from `j` — no free parameters

### Why GNN over MLP
- Jerk at node `i` depends on relative velocities of neighbors: `j_i = (1/m_i) Σ k_ij (v_j - v_i)`
- GNN message passing mirrors this interaction structure
- Learned edge weights are proxies for unknown coupling strengths (implicit system ID)
- Permutation invariant over neighbors

## Hard vs Soft Consistency
| | Hard | Soft |
|---|---|---|
| How | Only predict `j`; derive `a`,`v` analytically | Predict `j`,`a`,`v` independently + penalty loss |
| Consistency | Exact by construction | Approximate |
| Interpretability | High — `j` is the sole learned quantity | Medium — `a`,`v` absorb model error |
| Error accumulation | Compounds through integration | Self-correcting |
| **Choice** | **✓ Selected** | |

## Loss Function

### Why position loss alone fails
The jerk contribution to predicted position is O(dt³) ≈ 10⁻⁴ at typical dt. The loss surface is nearly flat with respect to jerk — the model learns the **wrong sign** (r ≈ −0.7) without direct supervision.

### Spring-mass loss (synthetic, unnormalized)
```
L = L_pos + λ * L_jerk
```
Works because analytic jerk supervision is exact and scales are compatible.

### Recommended: normalized multi-term loss (for real data)
$$\mathcal{L} = \frac{\|x_{\text{pred}} - x_{\text{true}}\|^2}{\sigma_x^2} + \lambda_a\,\frac{\|a_{\text{pred}} - a_{\text{true}}\|^2}{\sigma_a^2} + \lambda_j\,\frac{\|j_{\text{pred}} - j_{\text{true}}\|^2}{\sigma_j^2}$$

Dividing by training-data variance makes each term dimensionless O(1). λ_a and λ_j are interpretable relative importance weights, not scale corrections. Each term serves a distinct role:
- **Position term**: trains slow dynamics (velocity contribution dominates)
- **Acceleration term**: direct supervision for the hardest-to-predict quantity; gives jerk network strong gradient without needing to backprop through O(dt³)
- **Jerk term**: trains the social interaction law directly

### Practical guidance
- Pure jerk loss is dominated by rare large-jerk events → flat loss surface at most timesteps
- Start with normalized multi-term loss, λ_a = λ_j = 1.0
- Add event weighting (upweight near threshold crossings) only after basic architecture is confirmed working

## Single-step vs Unrolled Training
| | Single-step | Unrolled |
|---|---|---|
| Loss | `‖x(t+dt) - x_true‖²` at each `t` | `Σ_t ‖x(t) - x_true(t)‖²` |
| Training stability | Fast, stable | Gradient issues over long rollouts |
| Test robustness | Errors compound at rollout | More robust |
| **Choice** | **✓ Selected** (with caveat below) |

**Caveat**: with hard consistency, errors in `j` compound directly into `a` and `v`. Monitor rollout RMSE growth; if significant, consider curriculum training (gradually increase rollout length).

## Inputs to Network
Since only `x` is observed:
- **History window** `[x(t-k), ..., x(t)]` of length `k=5` as node features
- `v₀`, `a₀` estimated via **finite differences** (clean in noiseless synthetic case)
- At rollout: FD computed from model's own predicted `x` — main source of error accumulation

## Diagnostics for Model Underspecification

If derivative order is wrong, the missing terms get absorbed into biased lower-order estimates. Learned "acceleration" is no longer acceleration — it's acceleration + jerk-like compensation. This destroys interpretability.

**Structured residuals**: errors are not random — they correlate with the missing derivative. Plot rollout error vs instantaneous |jerk| — spikes during high-jerk events indicate the model is underspecified.

**dt-invariance test**: train at two different dt values. If learned coefficients change substantially, the model is absorbing missing Taylor terms. A correctly specified model's coefficients should be dt-invariant.

## Synthetic Testbed (Spring-Mass)
- 6 bodies, random masses in [0.5, 2.0]
- Sparse random spring connectivity (60%), plus chain backbone
- True jerk is analytic: `j_i = (1/m_i) Σ k_ij (v_j - v_i)`
- Enables direct supervision and ablation of jerk estimation quality

### Planned synthetic extensions (simplest first)
1. **Distance-gated springs**: `F_ij = k_ij*(x_j−x_i−l_ij) * 1[|x_j−x_i| < r*]` — sparse, event-triggered jerk; tests threshold recovery
2. **Velocity-dependent coupling**: adds `c_ij*(v_j−v_i)`; jerk depends on relative acceleration
3. **Two-timescale dynamics**: slow individual + fast social sub-networks
4. **Soft threshold**: sigmoid gating; GNN learns as attention weight
5. **Stochastic forcing**: separates social forcing from intrinsic noise

## Real Application: Collective Fish Dynamics
See `fish_gnn.ipynb` and `fish_derivative_analysis.ipynb`.

- 4 stickleback (28 dpf), SLEAP tracking, 25 fps
- LASSO: order 2 sufficient for single-step prediction (jerk Taylor term ≈ 6×10⁻⁵, invisible)
- Social analysis: |r| > 0.1 between fish jerk and relative neighbor velocity for all pairs — GNN is right architecture
- Fish likely respond to proximity/speed thresholds (event-triggered, not continuous)
- FishGNN uses normalized 3-term loss; FD jerk as noisy but usable supervision

## Files
- `jerk_gnn.ipynb` — simulation, GNN implementation, training loop, rollout plots
- `fish_gnn.ipynb` — FishGNN on real SLEAP data; normalized multi-term loss; order-3 hard consistency
- `fish_derivative_analysis.ipynb` — derivative order selection, social jerk structure, SINDy

## Key Design Decisions to Revisit
1. **History length** `k` — longer helps jerk estimation but increases input dim
2. **GNN depth** — more layers = larger receptive field (multi-hop interactions)
3. **Edge features** — currently uses true `k_ij`; for real system, replace with learned or uniform weights
4. **Curriculum training** — if single-step rollout error grows badly, unroll gradually
5. **Noise robustness** — FD-estimated `v`,`a` will be noisy in real system; may need smoothing or a learned state estimator front-end
6. **Threshold recovery** — distance-gated spring extension will test whether GNN can identify proximity threshold r* from data

---

## Is FNO a Good Fit Here?

**Short answer: not naturally, but there is a graph variant.**

### Standard FNO
- Designed for **PDE operators on regular grids**: learns a mapping `u(x,t) → u(x,t+dt)` in Fourier space
- Assumes a **continuous spatial domain** with uniform discretization
- Core operation: pointwise multiplication in frequency domain, which requires a regular lattice

### Why it's a mismatch for this problem
- Spring-mass bodies are **discrete, irregularly connected nodes** — no spatial grid
- FNO's Fourier transform requires a fixed regular domain; arbitrary graph topology breaks this
- The interaction structure here is fundamentally **graph-structured**, not field-structured

### Where FNO would make sense
- If your real system is a **spatially extended field** (e.g., neural recordings on a 2D array, fluid dynamics, continuous elastic medium)
- C. elegans SVAE work is a much better FNO use case — neural activity as a spatiotemporal field over the worm body axis

### Graph FNO (GFNO / Geo-FNO)
Yes, it exists. Key papers:
- **Geo-FNO** (Li et al. 2023) — extends FNO to irregular geometries by learning a latent regular grid mapping
- **Graph Neural Operator (GNO)** — original formulation, uses graph structure to approximate the integral kernel
- **GNOT** (Hao et al. 2023) — general neural operator on arbitrary geometries via attention

For this spring-mass problem, a standard GNN is cleaner and more interpretable than a Graph FNO — the interaction structure is already discrete and the physics doesn't benefit from a frequency-domain inductive bias.

### Bottom line
| System type | Recommended |
|---|---|
| Discrete interacting bodies (this problem) | GNN ✓ |
| Spatiotemporal neural field (C. elegans SVAE) | FNO ✓ |
| Irregular geometry PDE | Geo-FNO / GNO |
