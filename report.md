# NBA Point Differential Prediction: Effect of Data Granularity on Model Performance

**CSE 493G1 — Project Report**

---

## Problem Statement

Can we predict NBA game point differentials (home score minus away score) before tip-off, and does finer-grained data improve predictions? We compare five levels of data granularity across three model families using 10 seasons of NBA data (2015–2025).

## Data

All data was collected via the `nba_api` Python package. We used seasons 2015–2023 for training, 2023–24 for validation, and 2024–25 as a held-out test set (~1,225 games).

### Granularity Levels

| Level | Description | Features | Source |
|-------|-------------|----------|--------|
| **L1** — Season Aggregates | Per-team season averages (W%, net rating, pace, eFG%, etc.) for both teams, computed up to but not including the current game | 55 | `leaguedashteamstats` |
| **L2** — Game-Level Rolling | Per-team box score stats (PTS, REB, AST, FG%, etc.) averaged over a rolling window of recent games | 55 | `teamgamelog` |
| **L3** — Player-Level Rolling | Top-8 players by minutes per team; individual stats averaged over a rolling window, then aggregated to team means and standard deviations | 61 | `playergamelog` |
| **L5** — Schedule Context | Rest days, back-to-back flags, win streaks, fatigue index, consecutive road games, home/away win%, season phase | 24 | Derived from game dates |
| **L5+L1** — Combined | Season aggregates + schedule context + interaction terms (rest x quality, B2B x net rating, etc.) | 78–104 | L1 + L5 combined |

All features are strictly pre-game (no data leakage). Differential features (home minus away) are included for all stats.

## Models

| Model | Description |
|-------|-------------|
| **Ridge Regression** | Linear regression with L2 regularization. Alpha selected via cross-validation over [0.001, 5000]. |
| **XGBoost** | Gradient-boosted trees. Hyperparameters tuned via grid search (765 configurations, GPU-accelerated). Best config: depth=6, lr=0.03, subsample=0.6. |
| **Neural Network** | Fully connected network (PyTorch). Architecture and hyperparameters tuned via sweep (1,296 configurations on RTX 5090). Best config: 2 hidden layers (64, 32), dropout=0.5, lr=0.002. Final predictions averaged over 5 random seeds. |

## Key Experiments

### 1. Baseline Comparison (L1 vs L2 vs L3)

| Level | Ridge MAE | XGBoost MAE | NN MAE |
|-------|-----------|-------------|--------|
| L1 Season Agg | **10.62** | 10.76 | 10.73 |
| L2 Rolling-10 | 11.39 | 11.53 | 11.52 |
| L3 Player Roll-5 | 11.72 | 11.76 | 11.77 |

**Finding**: Coarser data wins. Season-level aggregates outperform game-level and player-level data by a wide margin. This was the opposite of our initial hypothesis.

### 2. Rolling Window Ablation (L2 and L3)

We tested window sizes from 3 to 40 games. Longer windows dramatically improved L2 and L3:

| Window | L2 Ridge MAE | L3 Ridge MAE |
|--------|-------------|-------------|
| 5 | 11.61 | 11.67 |
| 10 | 11.37 | 11.41 |
| **20** | **11.23** | **11.22** |
| 30 | 11.22 | 11.22 |

**Finding**: Window size was the single most impactful hyperparameter across all experiments. A 20-game window (~25% of a season) best balances stability and recency. With proper windowing, L2 and L3 converge to similar performance, but neither approaches L1.

### 3. Schedule Context Features (L5)

We engineered 24 context features from game dates (no additional API calls): rest days, back-to-back flags, games in last 7 days, win streaks, consecutive road games, home/away win percentages, and a composite fatigue index. We also created interaction terms (e.g., net rating x rest days, B2B x net rating) for a total of up to 104 features when combined with L1.

| Configuration | Ridge MAE | XGBoost MAE | NN MAE |
|--------------|-----------|-------------|--------|
| L5 standalone | 11.43 | 11.20 | 11.26 |
| L1 + L5 base | **10.53** | 10.60 | 10.56 |
| L1 + L5 enhanced (104 feat) | 10.53 | 10.53 | **10.51** |

**Finding**: Context features alone predict about as well as L2/L3. Combined with L1, they provide a consistent improvement across all models — the best single new signal we found.

### 4. Feature Importance

The top 20 features by XGBoost importance split evenly between L1 season stats and L5 context:

**L1 (team quality):** DIFF_NET_RATING, DIFF_W_PCT, DIFF_OFF_RATING, DIFF_DEF_RATING, DIFF_PTS, HOME/AWAY_NET_RATING, DIFF_EFG_PCT, HOME_GP, HOME_DEF_RATING

**L5 (schedule context):** HOME_IS_B2B, WINPCT_x_NETRTG, NETRTG_x_REST, AWAY_B2B_x_NETRTG, AWAY_IS_B2B, AWAY_REST_CAT, DIFF_FATIGUE_IDX, AWAY_REST_DAYS, AWAY_HOME_WIN_PCT, DIFF_OVERALL_WIN_PCT

Net rating differential alone is by far the most important feature (importance = 0.087), confirming that team quality difference is the primary driver of point differential.

### 5. Comprehensive Model Comparison (L1+L5 Enhanced, 104 Features)

| Model | Test MAE | Test R² |
|-------|----------|---------|
| Ridge | 10.529 | 0.268 |
| XGBoost (tuned, d=6) | 10.526 | 0.265 |
| NN (64,32), 5-seed avg | 10.506 | 0.271 |
| **Ensemble (55% XGB + 45% NN)** | **10.480** | **0.271** |

## Final Results

| Rank | Configuration | MAE | R² | Improvement vs L1 baseline |
|------|--------------|-----|----|----|
| 1 | L5+L1 enhanced, Ensemble (55%X+45%N) | **10.48** | 0.271 | -0.14 (1.3%) |
| 2 | L5+L1 enhanced, NN (64,32) 5-seed | 10.51 | 0.271 | -0.11 (1.1%) |
| 3 | L5+L1 top-20 features, Ridge | 10.52 | 0.273 | -0.10 (1.0%) |
| 4 | L5+L1 base, Ridge | 10.53 | 0.271 | -0.10 (0.9%) |
| 5 | L1+L2+L3 Ensemble (70/15/15) | 10.56 | 0.265 | -0.06 (0.6%) |
| 6 | L1 Ridge baseline | 10.62 | 0.265 | — |
| 7 | L2 Rolling-20, Ridge | 11.23 | 0.187 | +0.61 |
| 8 | L3 Agg Rolling-20, Ridge | 11.22 | 0.169 | +0.60 |
| 9 | L5 Standalone, XGBoost | 11.20 | 0.189 | +0.59 |

## Conclusions

**1. Granularity ranking: L1+L5 > L1 > L2 ~ L3 ~ L5 standalone.**
Season-level aggregates are the best predictor of team quality because they average over the most data. Finer-grained data (game-level, player-level) introduces noise that outweighs any additional signal. Schedule context is the most valuable complement to season stats.

**2. The prediction ceiling is ~10.5 MAE / ~27% R².**
After testing 2,000+ model configurations across 5 feature sets, our best model predicts point differentials to within ~10.5 points on average and explains ~27% of variance. NBA games have a point differential standard deviation of ~13 points; the remaining ~73% variance comes from in-game randomness (shooting variance, foul trouble, in-game injuries, referee calls) that no pre-game model can capture.

**3. Simple models match or beat complex ones.**
Ridge regression is competitive with XGBoost and neural networks at every granularity level. The best NN architecture was just 2 hidden layers (64, 32) with 50% dropout — barely more expressive than a linear model. This is characteristic of low signal-to-noise prediction tasks where the primary risk is overfitting.

**4. Data and feature engineering matter more than model choice.**
The biggest improvements came from fixing rolling window sizes (-0.45 MAE for L3) and adding schedule context features (-0.09 MAE). All model hyperparameter tuning combined contributed only ~-0.02 MAE. This is consistent with the widely observed pattern that data quality drives ML performance more than model complexity.

## Limitations

- **No injury data**: Player availability is likely the largest unexploited signal, but is not reliably available through `nba_api`.
- **No lineup data**: We use team-level and top-8-player aggregates, not actual starting lineups.
- **No betting market features**: Vegas lines are strong predictors but were excluded to focus on statistical features.
- **COVID seasons** (2019-20, 2020-21): Shortened/bubble seasons may introduce distributional shift.
- **API data gaps**: Some games have missing box scores or play-by-play data, handled via imputation or exclusion.
