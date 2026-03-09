# Claude Code Instructions: NBA Point-Differential Prediction

## Goal
Build a Jupyter notebook (`nba_point_diff.ipynb`) that predicts NBA game point differentials and compares model performance across **four levels of data granularity**. This is a research project for CSE 493G1.

---

## Project Structure

```
nba-point-diff/
├── nba_point_diff.ipynb       # Main notebook
├── data/
│   ├── raw/                   # Raw pulled data
│   └── processed/             # Cleaned datasets per granularity level
├── models/                    # Saved model artifacts
├── results/                   # Evaluation outputs, plots
└── requirements.txt
```

---

## Step-by-Step Notebook Sections

### 1. Setup & Installs
- Use Python 3.10+.
- Install and import: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `torch`, `nba_api`, `matplotlib`, `seaborn`, `tqdm`.
- Set random seeds everywhere for reproducibility (seed=42).

### 2. Data Collection

Pull historical NBA data (2015–2025 seasons) using the `nba_api` package (`from nba_api.stats.endpoints import ...`). For each granularity level, create a separate dataframe:

#### Level 1 — Season-Level Team Aggregates (Coarsest)
- Use `leaguedashteamstats` endpoint with `SeasonType='Regular Season'`.
- Features per team per season: W/L%, PPG, OPPG, ORtg, DRtg, NetRtg, Pace, eFG%, TOV%, OREB%, FT rate, etc.
- For each game, the input row is **both teams' season-level stats up to that point** (i.e., rolling season averages *before* the game).
- Target: actual point differential (home score − away score).

#### Level 2 — Game-Level Team Stats
- Use `leaguegamefinder` or `teamgamelog` endpoints.
- Features: per-game box score stats for each team (PTS, REB, AST, STL, BLK, TOV, FG%, 3P%, FT%, plus/minus).
- Construct rolling averages of the **last 10 games** for each team as input features. Include home/away flag.
- Target: point differential of the upcoming game.

#### Level 3 — Player-Level Box Scores
- Use `playergamelog` or `leaguedashplayerstats`.
- For each game, get the box scores of the **top 8 players by minutes** on each team (16 players total).
- Features per player: PTS, REB, AST, STL, BLK, TOV, MIN, FG%, 3P%, USG%, PER (rolling avg of last 5 games).
- Flatten into a single feature vector per game (16 players × N features).
- Target: point differential.

#### Level 4 — Possession-Level Play-by-Play (Finest)
- Use `playbyplayv2` endpoint.
- Aggregate per-game possession-level features: points per possession, transition frequency, turnover rate per possession, second-chance points rate, fast break frequency, paint touches, 3PA rate.
- Compute rolling averages over the **last 5 games** per team.
- Target: point differential.

**Important data hygiene:**
- Never include data from the game being predicted (no leakage).
- Use only data available *before* tip-off of each game.
- Train/test split: seasons 2015–2023 for training, 2023–2024 for validation, 2024–2025 for test.
- Handle missing data (injured players, shortened seasons) gracefully — impute or drop with documentation.
- Add a 1-second delay between API calls to respect rate limits (`time.sleep(1)`).
- Cache all raw API responses to disk (`data/raw/`) so you don't re-pull.

### 3. Feature Engineering & Preprocessing

For each granularity level:
- Standardize all features (zero mean, unit variance) using `StandardScaler`. Fit on train only.
- Create the final `X` (feature matrix) and `y` (point differential) arrays.
- Print shape and a small sample for sanity checking.
- Document the feature list clearly in a markdown cell.

### 4. Model Definitions

Train **three model families** on **each** of the four datasets (12 model runs total):

#### A. Ridge Regression (Baseline)
```python
from sklearn.linear_model import RidgeCV
model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
```

#### B. Gradient-Boosted Trees
```python
from xgboost import XGBRegressor
model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=20,
    eval_metric='rmse'
)
```
Use validation set for early stopping.

#### C. Fully Connected Neural Network (PyTorch)
```python
import torch
import torch.nn as nn

class PointDiffNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)
```
- Train with Adam optimizer, lr=1e-3, weight decay=1e-4.
- Use MSE loss.
- Train for up to 200 epochs with early stopping (patience=15) on validation loss.
- Use batch size 64, DataLoader with shuffle=True for training.

### 5. Training Loop

For each (granularity, model) pair:
1. Load the processed dataset.
2. Split into train/val/test.
3. Train the model.
4. Record training time.
5. Save model to `models/`.
6. Log val MSE/MAE during training.

Wrap all of this in a clean loop or function so results are collected into a single summary DataFrame:

```python
results = []  # list of dicts: {granularity, model_name, mse, mae, train_time}
```

### 6. Evaluation

On the **held-out test set** (2024–25 season), compute for each of the 12 runs:
- **MSE** (mean squared error)
- **MAE** (mean absolute error)
- **Median Absolute Error**
- **R² score**

Store all results in a DataFrame.

### 7. Visualization & Analysis

Create the following plots:

1. **Grouped bar chart**: MSE by granularity level, grouped by model type (3 bars per granularity group). Same for MAE. Use `matplotlib` or `seaborn`.
2. **Heatmap**: rows = granularity levels, columns = model types, cell values = MAE. Use `seaborn.heatmap`.
3. **Residual plots**: for the best-performing model at each granularity, plot predicted vs. actual point differentials (scatter with y=x reference line).
4. **Feature importance**: for the XGBoost models, plot top 15 features per granularity level.
5. **Learning curves** (optional): training vs. validation loss over epochs for the neural networks.

### 8. Summary & Conclusions

Add a final markdown cell that:
- Ranks granularity levels by performance.
- Discusses whether finer granularity helped or hurt.
- Notes any model × granularity interaction effects.
- Acknowledges limitations (e.g., injuries not captured, no lineup data, API data gaps).

---

## Key Technical Constraints

- **No data leakage**: only use pre-game information. Double-check rolling averages don't include the current game.
- **API rate limits**: cache everything, add `time.sleep(1)` between calls.
- **Reproducibility**: set `random_state=42` for sklearn, `torch.manual_seed(42)`, `np.random.seed(42)`.
- **Memory**: player-level and play-by-play datasets can be large. Use efficient dtypes (`float32`), process season-by-season if needed.
- **Notebook flow**: every cell should run top-to-bottom cleanly. Use markdown headers to separate sections.

## Output
When complete, the notebook should be fully runnable and self-contained, with all plots inline and a clear written conclusion at the end.
