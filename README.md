# Event Detection with Temporal Pattern Recognition

## Problem Statement

Predict binary target events (`y`) in time-series data grouped by (`g1`, `g2`) pairs. Each group represents an independent sequence of observations over time (`relative_date_number`). The challenge involves:

- **Highly imbalanced data**: ~4-5% positive rate
- **Temporal dependencies**: Events cluster in specific cycle phases
- **Group heterogeneity**: Different (`g1`, `g2`) pairs have varying baseline behaviors
- **12-day seasonal cycle**: Demand patterns repeat every 12 days

## Key Insight: x12 as Leading Indicator

Feature `x12` (binary 0/1) serves as the primary signal for target events. However:
- Raw x12 has high false positive rate (single-day blips outside seasonal windows)
- True events require **sustained x12 activity** (2+ consecutive days) during **demand cycle phases** (days 4-8)

## Solution Architecture

### 1. Feature Engineering Pipeline (`new_features.py`)

**Temporal Structure Features**
- `t_norm`: Position in sequence (0-1) for comparable time perception across short/long sequences
- `cycle_pos`, `cycle_sin/cos`: 12-day cyclical encoding capturing "where in the month"
- `cycle_high_risk`: Binary flag for days 4-8 of cycle

**x12 State Machine**
- `x12_run`: Consecutive days of x12=1 (persistence metric)
- `x12_regime`: Score 0-3 combining (run≥2, ratio&gt;1.2, in high-risk)
- `x12_onset`: Transition from 0→1 (activation moment)
- `latent_event_state`: Strict criteria for established events (run≥3, ratio&gt;1.3, high-risk)

**Contextual Features**
- Rolling statistics (3-day window): mean, std, max, ratio_to_mean
- Lag features: t-1 and t-3 values and deltas
- Peer anomaly: Deviation from same-g1 group averages
- Cumulative statistics: Historical mean, percentile rank

**Interaction Features**
- `x12_cycle`: x12 active during demand window
- `x12_onset_near_peak`: Activation within 1 day of peak risk
- `x12_dist_weighted`: Activity weighted by proximity to peak

### 2. Model: XGBoost with Custom Regularization

```python
params = {
   'objective': 'binary:logistic',
   'scale_pos_weight': ~20,  # Auto-calculated per fold
   'max_depth': 6,           # Constrained for stability
   'min_child_weight': 20,   # Prevent overfitting to small groups
   'subsample': 0.6,         # Aggressive row sampling
   'colsample_bytree': 0.7,  # Feature sampling
   'reg_alpha': 0.2,         # L1 regularization
   'reg_lambda': 1.5,        # L2 regularization
}
```

**Why XGBoost:
- Handles mixed feature types (binary, continuous, cyclical)
- Native missing value support
- GPU acceleration for fast iteration

### 3. Post-Processing: Domain-Knowledge Filtering
Raw predictions filtered through expert rules:
```python
def post_process_predictions(df, proba, threshold, low_threshold=0.15, min_run=2):
    # Keep prediction if:
    # 1. Raw probability above threshold AND
    # 2. Any of:
    #    - Previous day also predicted (continuity)
    #    - In high-risk cycle phase (timing forgiveness)
    #    - Weak signal but sustained x12_run≥2 (persistence)
    #    - At sequence end (last-chance capture)
```
Purpose: Reduce false positives from single-day x12 occurances while preserving true sustained events.

### 4. Validation Strategy
**GroupKFold with stratification on y:**

- Prevents leakage: Same (g1, g2) never in train and validation
- 5 folds with g1_g2 concatenated as group key

**Threshold Optimization:**  
- Per-fold: Grid search 0.05-0.5 (200 steps) with post-processing

**Global:**  
- Single threshold on pooled OOF predictions

| Metric              | Value       |
| ------------------- | ----------- |
| Mean CV F1          | 0.43 ± 0.008|
| Best Fold F1        | 0.448       |
| Optimized Threshold | ~0.39       |


### 5. Key Design Decisions
**Exclude raw x12 from model inputs:**
- Use derived features (run, regime, onset) to force learning temporal patterns, not just correlation

**Dual threshold system:**
- Strict threshold for raw predictions, relaxed criteria for rescue in post-process

**Cycle-aware features:**
- Sin/cos encoding prevents "day 11 vs day 0" discontinuity

**Peer comparison:** 
- g1-level aggregates detect group-wide anomalies vs local deviations
