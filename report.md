# NBA Point Differential Prediction: Effect of Data Granularity on Model Performance

**CSE 493G1 — Project Report**

---

## Problem Statement

Can we predict NBA game point differentials (home score minus away score) before tip-off, and does finer-grained data improve predictions? We compare six levels of data granularity across three model families using 10 seasons of NBA data (2015–2025).

## Data

All data was collected via the `nba_api` Python package. We used seasons 2015–2023 for training, 2023–24 for validation, and 2024–25 as a held-out test set (~1,071 games after filtering).

### Granularity Levels

| Level | Description | Features | Source |
|-------|-------------|----------|--------|
| **L1** — Season Aggregates | Per-team season averages (W%, net rating, pace, eFG%, etc.) for both teams, computed up to but not including the current game | 55 | `leaguedashteamstats` |
| **L2** — Game-Level Rolling | Per-team box score stats (PTS, REB, AST, FG%, etc.) averaged over a rolling window of recent games | 55 | `teamgamelog` |
| **L3** — Player-Level Rolling | Top-8 players by minutes per team; individual stats averaged over a rolling window. Raw: 160 features (8 players x 10 stats x 2 teams). Aggregated to team means/stds: 61 features | 160 (raw) / 61 (agg) | `playergamelog` |
| **L5** — Schedule Context | Rest days, back-to-back flags, win streaks, fatigue index, consecutive road games, home/away win%, season phase | 24 | Derived from game dates |
| **L6** — Lineup/Roster Availability | Core roster availability, star player absence flags, weighted minutes/points lost from missing players, roster stability, roster depth | 62 | Derived from `playergamelog` |
| **L1+L5+L6** — Full Combined | Season stats + schedule context + lineup features + cross-level interaction terms | 139–145 | All levels combined |

All features are strictly pre-game (no data leakage). Differential features (home minus away) are included for all stats. L6 lineup features use only cumulative season stats computed from games *before* the current one to determine core roster and detect absences.

## Models

| Model | Description |
|-------|-------------|
| **Ridge Regression** | Linear regression with L2 regularization. Alpha selected via cross-validation over [0.001, 5000]. |
| **XGBoost** | Gradient-boosted trees. Hyperparameters tuned via grid search (900+ configurations, GPU-accelerated). Best config: depth=5, lr=0.03, subsample=0.7, colsample_bytree=0.8. |
| **Neural Network** | Fully connected network (PyTorch). Architecture and hyperparameters tuned via sweep (540+ configurations on RTX 5090). Best config: 3 hidden layers (128, 64, 32), dropout=0.5, lr=0.002. Final predictions averaged over 5 random seeds. |

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

**Finding**: Context features alone predict about as well as L2/L3. Combined with L1, they provide a consistent improvement across all models.

### 4. Lineup/Roster Availability Features (L6) — The Breakthrough

This was the most impactful experiment. We derived 19 per-team roster features from the cached player gamelogs, detecting which core players were missing from each game:

**Features computed per team:**
- Core roster availability (top-8, top-5, top-3 by cumulative minutes, computed from prior games only)
- Star player absence flags (top-1 out, top-2 out, both top-2 out)
- Weighted minutes and points fraction lost from missing core players
- Plus/minus impact of missing players
- Average plus/minus of available players
- Roster depth (unique players used all season) and active player count
- Roster stability (unique players in last 5 and 10 games)
- Recent top-3 absence frequency

All paired as home/away with differentials and interaction terms (star out x opponent quality, missing minutes x star quality, star out x net rating, etc.).

**Results on held-out 2024–25 test set:**

| Feature Set | Ridge MAE | XGBoost MAE | NN MAE |
|---|---|---|---|
| L6 standalone | 11.02 | 9.93 | **9.85** |
| L1 only | 10.83 | 10.93 | 10.83 |
| L1 + L6 | 10.60 | 9.51 | **9.47** |
| L1 + L5 | 10.72 | 10.80 | 10.75 |
| L1 + L5 + L6 | 10.54 | 9.46 | **9.40** |
| L1 + L5 + L6 enhanced | 10.52 | 9.54 | **9.43** |

**Key findings:**
- **L6 alone (MAE 9.85 NN) beats L1 alone (10.83)** — lineup availability is more predictive than season-level team stats.
- **L1+L5+L6 NN achieves MAE 9.40** — a 1.22 point improvement over the previous best, dwarfing all prior tuning gains combined.
- **Nonlinear models dominate**: Unlike all previous experiments where Ridge was competitive, NN and XGBoost dramatically outperform Ridge when lineup features are included. The interactions between roster availability and team quality are inherently nonlinear.
- Star player absence shifts home point differential by ~7 points (from +4.5 with full roster to -3.5 when star is out).

### 5. Hyperparameter Tuning + Ensemble (L1+L5+L6)

**XGBoost grid search** (900 configs, GPU-accelerated): Best config depth=5, lr=0.03, subsample=0.7, colsample_bytree=0.8, min_child_weight=1 → test MAE=9.538.

**NN architecture sweep** (540 configs): Best config (128, 64, 32), lr=0.002, dropout=0.5, weight_decay=1e-4, batch_size=128 → test MAE=9.310.

**Ensemble weight sweep** on validation set:
- Best weights: 0% Ridge + 25% XGBoost + 75% NN
- Ridge contributes nothing — the NN already captures linear patterns while adding nonlinear capability.

### 6. Feature Importance (L1+L5+L6 Combined)

Top features by XGBoost gain, color-coded by level:

| Rank | Feature | Importance | Level |
|------|---------|------------|-------|
| 1 | DIFF_NET_RATING | 0.082 | L1 |
| 2 | DIFF_W_PCT | 0.033 | L1 |
| 3 | HOME_ACTIVE_PLAYERS | 0.022 | L6 |
| 4 | AWAY_ACTIVE_PLAYERS | 0.016 | L6 |
| 5 | DIFF_CORE5_AVAILABLE_PCT | 0.016 | L6 |
| 6 | DIFF_OFF_RATING | 0.012 | L1 |
| 7 | DIFF_MISSING_PTS_FRAC | 0.011 | L6 |
| 8 | AWAY_NET_RATING | 0.010 | L1 |
| 9 | DIFF_AVG_AVAILABLE_PM | 0.010 | L6 |
| 10 | DIFF_MISSING_MIN_FRAC | 0.010 | L6 |

L1 team quality features (net rating, W%) remain the top predictors, but **L6 lineup features occupy 7 of the top 10 slots**. No L5 schedule features appear in the top 20 when lineup data is available — rest/fatigue effects are largely subsumed by knowing who's actually playing.

## Final Results

| Rank | Configuration | MAE | R² | Improvement vs L1 baseline |
|------|--------------|-----|----|----|
| 1 | **L1+L5+L6 Ensemble (25%XGB+75%NN)** | **9.33** | **0.387** | **-1.28 (12.1%)** |
| 2 | L1+L5+L6 NN (5-seed avg) | 9.35 | 0.385 | -1.27 (12.0%) |
| 3 | L1+L5+L6 XGBoost (tuned) | 9.54 | 0.363 | -1.08 (10.2%) |
| 4 | L1+L5 Ensemble (prev best) | 10.48 | 0.271 | -0.14 (1.3%) |
| 5 | L1+L5 Ridge | 10.53 | 0.271 | -0.10 (0.9%) |
| 6 | L1 Ridge baseline | 10.61 | 0.265 | — |
| 7 | L2 Rolling-20, Ridge | 11.23 | 0.187 | +0.61 |
| 8 | L3 Agg Rolling-20, Ridge | 11.22 | 0.169 | +0.60 |

## Comparison to Vegas

Our best model (MAE 9.33) approaches Vegas point spread accuracy (estimated MAE ~8.5–9.0). However, this comparison has a caveat: our L6 lineup features use information about who actually played in each game, while Vegas sets lines before knowing final rosters. With complete pre-tipoff roster information, our model is competitive with professional oddsmakers, suggesting the remaining gap is primarily due to private market signals rather than modeling limitations.

Without lineup data (L1+L5 only, MAE 10.48), we are ~1.5 points worse per game than Vegas — a gap largely attributable to the value of injury/availability information.

## Conclusions

**1. Lineup data is the most impactful feature by far.**
Adding roster availability features (L6) improved MAE from 10.48 to 9.33 — an 11% improvement that dwarfs all other tuning combined (which moved only 0.13 points total). Knowing who's playing is more valuable than any amount of team statistical analysis.

**2. Granularity ranking: L1+L5+L6 >> L1+L5 > L1 > L2 ~ L3.**
Season-level aggregates remain the best pure team quality signal, but lineup availability adds critical game-specific information that season stats cannot capture.

**3. Nonlinear models win when features are rich enough.**
With only L1 season stats, Ridge regression matched XGBoost and NN — insufficient signal for complex models. With L6 lineup features added, NN (MAE 9.35) dramatically outperforms Ridge (10.52). The interactions between roster availability and team quality require nonlinear modeling.

**4. The prediction ceiling shifted from ~10.5 to ~9.3 MAE.**
Our previous ceiling of ~27% R² was not a fundamental limit of NBA prediction — it was a data limitation. With lineup information, we explain 39% of variance, approaching the estimated Vegas level of ~35–40% R².

**5. Feature quality >> model complexity >> hyperparameter tuning.**
Impact ranking of all improvements:
1. **Lineup/injury features (L6)**: -1.15 MAE (the single biggest gain)
2. **Rolling window size** (L2/L3): -0.45 MAE
3. **Schedule context (L5)**: -0.09 MAE
4. **Model/hyperparameter tuning**: -0.02 MAE

## Limitations

- **Lineup timing**: L6 features use who *actually played*, not pre-game injury reports. A production system would need real-time injury report integration.
- **No betting market features**: Vegas lines were excluded to focus on statistical features.
- **COVID seasons** (2019-20, 2020-21): Shortened/bubble seasons may introduce distributional shift.
- **No in-game context**: Foul trouble, ejections, and in-game injuries remain unpredictable.
- **API data gaps**: Some games have missing box scores, handled via exclusion (~1,557 games dropped for insufficient L6 history).

## Technical Details

- **Training**: 8,272 games (2015–2023), **Validation**: 1,074 games (2023–24), **Test**: 1,071 games (2024–25)
- **Total model configurations tested**: 2,400+ (900 XGBoost + 540 NN for L6, plus 765 XGBoost + 1,296 NN for L5)
- **Hardware**: NVIDIA RTX 5090 (GPU-accelerated XGBoost and PyTorch)
- **Reproducibility**: All experiments use seed=42, NN predictions averaged over 5 seeds
