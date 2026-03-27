# Context: Fish Collective Dynamics — Derivative Order for Rollout

## Background (bring this to a new session)

We were building a **Jerk-GNN** for predicting dynamics of an unknown physical system. The synthetic testbed is a 6-body spring-mass system. The core idea:

- Only **position x** is observed
- A GNN predicts **jerk** j = d³x/dt³ as the sole learned quantity
- **Hard consistency**: a and v are derived analytically from j (not learned)
- Integration scheme: `x(t+dt) = x(t) + v·dt + ½a·dt² + ⅙j·dt³`
- **Single-step training** on position prediction loss
- Truncation error is O(dt⁴)

---

## Question: What is the highest-order derivative to learn?

**Context**: Want to apply the same model to collective fish dynamics, where only position x is observed.

### Why higher derivatives vanish at coarse dt

Taylor expansion terms scale as dt^k. The ratio of jerk to velocity term is:

```
(1/6 · j · dt³) / (v · dt) = (j / 6v) · dt²
```

At coarse dt this goes to zero as dt² — higher orders are suppressed not because they're small physically, but because dt^k kills them. You literally cannot distinguish them from data at coarse resolution.

### Diagnostics to determine effective order

1. **Autocorrelation of each derivative** — once a derivative looks like white noise, higher orders carry no predictive signal
2. **Power spectra** — flat spectrum = white noise = nothing to learn
3. **Taylor truncation RMSE vs order** — plot RMSE as function of k; plateau = effective order
4. **Variance explained by each term**: `r_k = Var(x^(k) · dt^k / k!) / Var(Δx)`
5. **LASSO order selection** (see below)

### LASSO-based automated order selection

Set up sparse regression where each Taylor term is a feature:

```
Δx = β₁·v·dt + β₂·(a/2)·dt² + β₃·(j/6)·dt³ + β₄·(s/24)·dt⁴ + ε
```

- Under perfect physics all βₖ = 1
- LASSO shrinks irrelevant βₖ → 0
- Run LARS stability path to see order in which terms enter
- Run at multiple dt values — stable coefficients = real signal
- **SINDy** (Sparse Identification of Nonlinear Dynamics) is this idea generalized — also tests whether social interaction terms (position/velocity differences between fish) are needed

### The core tension at small dt

At small dt, the prediction target `x(t+dt) - x(t) ≈ v·dt` is tiny relative to noise. The jerk term ~dt³ is even tinier. Fix: **normalize target by dt**, i.e. predict velocity-like quantities rather than displacements. The jerk-GNN architecture already handles this correctly — the network predicts j (an O(1) quantity) and dt factors are handled analytically.

### For fish specifically

- Cruise behavior: likely **order 2** (acceleration) sufficient
- Rapid turns / predator escape: **order 3** (jerk) may matter
- Key check: does jᵢ correlate with relative velocity of neighbors (vⱼ - vᵢ)? If yes → jerk encodes real social interaction

---

## Data

**File**: `skeleton_28dpf_v1_complete_500_002_testinference_000_rlt_250731_v0_analysis.h5`  
**Format**: SLEAP pose tracking (HDF5)  
**Structure**:
```
tracks: (4, 2, 6, 14510)  →  (fish, xy, node, frame)
```
- 4 fish (track_0 … track_3)
- 2 coordinates (x, y) in pixels
- 6 body nodes: eye_L, eye_R, body_1, body_2, body_3, tail
- 14,510 frames (~9.7 min at assumed 25 fps)
- Tracking completeness: >99% for all fish
- Arena: x=56–917 px, y=88–896 px
- Per-frame displacement (body_1): mean=1.09 px, std=1.34 px, p95=3.56 px

**Load code**:
```python
import h5py, numpy as np

with h5py.File('path/to/file.h5', 'r') as f:
    tracks     = f['tracks'][:]           # (4, 2, 6, 14510)
    node_names = [n.decode() for n in f['node_names'][:]]
    # ['eye_L', 'eye_R', 'body_1', 'body_2', 'body_3', 'tail']

# Reshape to (T, fish, node, xy)
pos_all = tracks.transpose(3, 0, 2, 1)
pos     = pos_all[:, :, 2, :]   # body_1 centroid: (14510, 4, 2)
```

---

## Analysis Notebook

`fish_derivative_analysis.ipynb` runs the following on the SLEAP data:

1. Raw trajectory visualization (spatial + time series)
2. Finite difference derivatives up to order 4 (position → snap)
3. Autocorrelation + whiteness test per derivative order
4. Power spectra per derivative order
5. Taylor truncation RMSE vs order (with marginal gain plot)
6. LASSO order selection + regularization path
7. SINDy sparse dynamics identification
8. Social structure of jerk: does jᵢ correlate with vⱼ - vᵢ?
9. Summary printout with recommendation

**Key packages**: `h5py`, `numpy`, `scipy`, `scikit-learn`, `pysindy`, `matplotlib`

---

## Next Steps

- Run the notebook on the fish data and read the summary section
- Decision rule:
  - Jerk Taylor term > ~1% of step size AND structured autocorrelation → use order 3
  - LASSO zeroes out jerk → order 2 sufficient
  - Social jerk correlation |r| > 0.1 → jerk encodes fish interaction, model explicitly with GNN
- Once order is determined, adapt the **Jerk-GNN** architecture (from spring-mass testbed) to fish:
  - Nodes = fish, edges = inferred social interactions
  - Input = position history window (FD estimates of v, a fed as node features)
  - Output = jerk per fish (or acceleration if order 2 suffices)
  - Hard consistency integration unchanged
