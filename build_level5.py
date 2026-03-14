#!/usr/bin/env python3
"""
Build Level 5 — Schedule Context Features
Computes rest days, back-to-backs, win streaks, travel fatigue, etc.
from cached game data (no new API calls needed).

Then runs Ridge / XGBoost / NN on:
  - L5 standalone (context features only)
  - L5 + L1 combined (context + season stats)
"""

import os, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
from xgboost import XGBRegressor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

ALL_SEASONS = [f"{y}-{str(y+1)[-2:]}" for y in range(2015, 2025)]
TRAIN_SEASONS = [s for s in ALL_SEASONS if int(s[:4]) <= 2022]
VAL_SEASONS   = [s for s in ALL_SEASONS if int(s[:4]) == 2023]
TEST_SEASONS  = [s for s in ALL_SEASONS if int(s[:4]) == 2024]

# ── Helpers from tune.py ─────────────────────────────────────────────────────

def eval_metrics(y_true, y_pred):
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'median_ae': median_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
    }

class PointDiffNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 128, 64), dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for i, h in enumerate(hidden_dims[:-1]):
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, hidden_dims[-1]), nn.ReLU(), nn.Linear(hidden_dims[-1], 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_nn(X_tr, y_tr, X_v, y_v, input_dim, epochs=200, batch_size=64,
             lr=1e-3, weight_decay=1e-4, patience=15, dropout=0.3,
             hidden_dims=(256, 128, 64)):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PointDiffNet(input_dim, hidden_dims=hidden_dims, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    t_X = torch.tensor(X_tr, dtype=torch.float32)
    t_y = torch.tensor(y_tr, dtype=torch.float32)
    v_X = torch.tensor(X_v, dtype=torch.float32).to(device)
    v_y = torch.tensor(y_v, dtype=torch.float32).to(device)
    loader = DataLoader(TensorDataset(t_X, t_y), batch_size=batch_size, shuffle=True)

    best_val, best_state, wait = float('inf'), None, 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(v_X), v_y).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_val

def predict_nn(model, X):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()

def run_experiment(name, X_tr, y_tr, X_v, y_v, X_te, y_te,
                   ridge_alphas=None, xgb_params=None, nn_params=None):
    """Run all 3 models on given data, return results list."""
    results = []

    # Ridge
    alphas = ridge_alphas or [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    t0 = time.time()
    ridge = RidgeCV(alphas=alphas)
    ridge.fit(X_tr, y_tr)
    y_pred = ridge.predict(X_te)
    m = eval_metrics(y_te, y_pred)
    m.update({'experiment': name, 'model': 'Ridge', 'train_time': time.time()-t0, 'alpha': ridge.alpha_})
    results.append(m)
    print(f"  Ridge:   MAE={m['mae']:.3f}  MSE={m['mse']:.2f}  R²={m['r2']:.3f}  α={ridge.alpha_}")

    # XGBoost
    xp = xgb_params or {'max_depth': 4, 'learning_rate': 0.01, 'subsample': 0.7,
                         'colsample_bytree': 0.7, 'min_child_weight': 3}
    t0 = time.time()
    xgb = XGBRegressor(
        n_estimators=1000,
        early_stopping_rounds=30,
        eval_metric='mae',
        random_state=SEED,
        verbosity=0,
        **xp,
    )
    xgb.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
    y_pred = xgb.predict(X_te)
    m = eval_metrics(y_te, y_pred)
    m.update({'experiment': name, 'model': 'XGBoost', 'train_time': time.time()-t0,
              'best_iteration': xgb.best_iteration})
    results.append(m)
    print(f"  XGBoost: MAE={m['mae']:.3f}  MSE={m['mse']:.2f}  R²={m['r2']:.3f}  trees={xgb.best_iteration}")

    # NN — use smaller network for context features (fewer features)
    np_ = nn_params or {'lr': 1e-3, 'dropout': 0.3, 'batch_size': 64,
                        'weight_decay': 1e-4, 'hidden_dims': (128, 64, 32)}
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t0 = time.time()
    nn_model, best_val = train_nn(
        X_tr, y_tr, X_v, y_v, input_dim=X_tr.shape[1], patience=20, **np_,
    )
    y_pred = predict_nn(nn_model, X_te)
    m = eval_metrics(y_te, y_pred)
    m.update({'experiment': name, 'model': 'NeuralNet', 'train_time': time.time()-t0,
              'best_val_mse': best_val})
    results.append(m)
    print(f"  NN:      MAE={m['mae']:.3f}  MSE={m['mse']:.2f}  R²={m['r2']:.3f}")

    return results, ridge, xgb, nn_model

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: BUILD CONTEXT FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def build_context_features():
    """Load raw game CSVs and compute schedule/context features per team per game."""
    print("=" * 70)
    print("BUILDING LEVEL 5 — SCHEDULE CONTEXT FEATURES")
    print("=" * 70)

    # Load all raw game data
    dfs = []
    for season in ALL_SEASONS:
        path = f'data/raw/games_{season}.csv'
        if os.path.exists(path):
            df = pd.read_csv(path)
            dfs.append(df)
            print(f"  Loaded {path}: {len(df)} rows")
    games = pd.concat(dfs, ignore_index=True)
    print(f"  Total: {len(games)} team-game rows")

    # Parse dates and determine home/away
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    games['IS_HOME'] = games['MATCHUP'].str.contains('vs\\.').astype(int)

    # Sort by team and date for rolling computations
    games = games.sort_values(['TEAM_ID', 'GAME_DATE', 'GAME_ID']).reset_index(drop=True)

    # ── Per-team context features ─────────────────────────────────────────
    print("\n  Computing per-team context features...")

    context_rows = []
    for (team_id, season), team_games in games.groupby(['TEAM_ID', 'SEASON']):
        team_games = team_games.sort_values('GAME_DATE').reset_index(drop=True)

        for i in range(len(team_games)):
            row = team_games.iloc[i]
            game_id = row['GAME_ID']
            game_date = row['GAME_DATE']

            # --- REST_DAYS: days since last game ---
            if i == 0:
                rest_days = 7  # season opener, assume well-rested
            else:
                prev_date = team_games.iloc[i-1]['GAME_DATE']
                rest_days = (game_date - prev_date).days

            # --- IS_B2B ---
            is_b2b = 1 if rest_days <= 1 else 0

            # --- GAMES_IN_LAST_7: count of games in last 7 days (excluding current) ---
            week_ago = game_date - pd.Timedelta(days=7)
            games_last_7 = ((team_games.iloc[:i]['GAME_DATE'] > week_ago)).sum()

            # --- WIN_STREAK: consecutive W/L (positive=wins, negative=losses) ---
            streak = 0
            if i > 0:
                for j in range(i-1, -1, -1):
                    wl = team_games.iloc[j]['WL']
                    if j == i-1:
                        direction = 1 if wl == 'W' else -1
                        streak = direction
                    else:
                        if (wl == 'W' and direction == 1) or (wl == 'L' and direction == -1):
                            streak += direction
                        else:
                            break

            # --- GAME_NUMBER: nth game of season ---
            game_number = i + 1

            # --- CONSECUTIVE_AWAY: count of consecutive road games ending at this game ---
            consec_away = 0
            if row['IS_HOME'] == 0:  # current game is away
                consec_away = 1
                for j in range(i-1, -1, -1):
                    if team_games.iloc[j]['IS_HOME'] == 0:
                        consec_away += 1
                    else:
                        break
            # If home, consec_away = 0

            # --- HOME_WIN_PCT / AWAY_WIN_PCT (rolling, pre-game) ---
            prev_games = team_games.iloc[:i]
            home_games = prev_games[prev_games['IS_HOME'] == 1]
            away_games = prev_games[prev_games['IS_HOME'] == 0]

            home_win_pct = (home_games['WL'] == 'W').mean() if len(home_games) > 0 else 0.5
            away_win_pct = (away_games['WL'] == 'W').mean() if len(away_games) > 0 else 0.5

            # --- OVERALL_WIN_PCT (rolling, pre-game) ---
            overall_win_pct = (prev_games['WL'] == 'W').mean() if len(prev_games) > 0 else 0.5

            context_rows.append({
                'GAME_ID': game_id,
                'TEAM_ID': team_id,
                'SEASON': season,
                'GAME_DATE': game_date,
                'IS_HOME': row['IS_HOME'],
                'REST_DAYS': rest_days,
                'IS_B2B': is_b2b,
                'GAMES_IN_LAST_7': games_last_7,
                'WIN_STREAK': streak,
                'GAME_NUMBER': game_number,
                'CONSECUTIVE_AWAY': consec_away,
                'HOME_WIN_PCT': home_win_pct,
                'AWAY_WIN_PCT': away_win_pct,
                'OVERALL_WIN_PCT': overall_win_pct,
            })

    ctx = pd.DataFrame(context_rows)
    print(f"  Context features computed: {ctx.shape}")

    # ── Sanity checks ─────────────────────────────────────────────────────
    print(f"\n  Sanity checks:")
    print(f"    REST_DAYS: mean={ctx['REST_DAYS'].mean():.2f}, median={ctx['REST_DAYS'].median():.0f}")
    print(f"    IS_B2B rate: {ctx['IS_B2B'].mean():.3f}")
    print(f"    WIN_STREAK: range [{ctx['WIN_STREAK'].min()}, {ctx['WIN_STREAK'].max()}]")
    print(f"    GAMES_IN_LAST_7: mean={ctx['GAMES_IN_LAST_7'].mean():.2f}")
    print(f"    CONSECUTIVE_AWAY: mean={ctx['CONSECUTIVE_AWAY'].mean():.2f}, max={ctx['CONSECUTIVE_AWAY'].max()}")

    # ── Pair home/away into single rows ───────────────────────────────────
    print("\n  Pairing home/away teams...")

    home_ctx = ctx[ctx['IS_HOME'] == 1].copy()
    away_ctx = ctx[ctx['IS_HOME'] == 0].copy()

    home_ctx = home_ctx.rename(columns={
        'TEAM_ID': 'HOME_TEAM_ID',
        'REST_DAYS': 'HOME_REST_DAYS',
        'IS_B2B': 'HOME_IS_B2B',
        'GAMES_IN_LAST_7': 'HOME_GAMES_IN_LAST_7',
        'WIN_STREAK': 'HOME_WIN_STREAK',
        'GAME_NUMBER': 'HOME_GAME_NUMBER',
        'CONSECUTIVE_AWAY': 'HOME_CONSECUTIVE_AWAY',
        'HOME_WIN_PCT': 'HOME_HOME_WIN_PCT',
        'AWAY_WIN_PCT': 'HOME_AWAY_WIN_PCT',
        'OVERALL_WIN_PCT': 'HOME_OVERALL_WIN_PCT',
    })
    away_ctx = away_ctx.rename(columns={
        'TEAM_ID': 'AWAY_TEAM_ID',
        'REST_DAYS': 'AWAY_REST_DAYS',
        'IS_B2B': 'AWAY_IS_B2B',
        'GAMES_IN_LAST_7': 'AWAY_GAMES_IN_LAST_7',
        'WIN_STREAK': 'AWAY_WIN_STREAK',
        'GAME_NUMBER': 'AWAY_GAME_NUMBER',
        'CONSECUTIVE_AWAY': 'AWAY_CONSECUTIVE_AWAY',
        'HOME_WIN_PCT': 'AWAY_HOME_WIN_PCT',
        'AWAY_WIN_PCT': 'AWAY_AWAY_WIN_PCT',
        'OVERALL_WIN_PCT': 'AWAY_OVERALL_WIN_PCT',
    })

    home_cols = ['GAME_ID', 'HOME_TEAM_ID', 'HOME_REST_DAYS', 'HOME_IS_B2B',
                 'HOME_GAMES_IN_LAST_7', 'HOME_WIN_STREAK', 'HOME_GAME_NUMBER',
                 'HOME_CONSECUTIVE_AWAY', 'HOME_HOME_WIN_PCT', 'HOME_AWAY_WIN_PCT',
                 'HOME_OVERALL_WIN_PCT']
    away_cols = ['GAME_ID', 'AWAY_TEAM_ID', 'AWAY_REST_DAYS', 'AWAY_IS_B2B',
                 'AWAY_GAMES_IN_LAST_7', 'AWAY_WIN_STREAK', 'AWAY_GAME_NUMBER',
                 'AWAY_CONSECUTIVE_AWAY', 'AWAY_HOME_WIN_PCT', 'AWAY_AWAY_WIN_PCT',
                 'AWAY_OVERALL_WIN_PCT']

    paired = home_ctx[home_cols].merge(away_ctx[away_cols], on='GAME_ID', how='inner')
    print(f"  Paired games: {len(paired)}")

    # ── Differential features ─────────────────────────────────────────────
    paired['REST_ADVANTAGE'] = paired['HOME_REST_DAYS'] - paired['AWAY_REST_DAYS']
    paired['DIFF_WIN_STREAK'] = paired['HOME_WIN_STREAK'] - paired['AWAY_WIN_STREAK']
    paired['DIFF_GAMES_IN_LAST_7'] = paired['HOME_GAMES_IN_LAST_7'] - paired['AWAY_GAMES_IN_LAST_7']
    paired['DIFF_GAME_NUMBER'] = paired['HOME_GAME_NUMBER'] - paired['AWAY_GAME_NUMBER']
    paired['DIFF_OVERALL_WIN_PCT'] = paired['HOME_OVERALL_WIN_PCT'] - paired['AWAY_OVERALL_WIN_PCT']
    paired['HOME_ADV'] = 1.0

    # ── Merge with L1 for POINT_DIFF and SEASON ──────────────────────────
    df_l1 = pd.read_csv('data/processed/level1_season_agg.csv')
    l1_meta = df_l1[['GAME_ID', 'SEASON', 'POINT_DIFF']].drop_duplicates(subset='GAME_ID')

    result = paired.merge(l1_meta, on='GAME_ID', how='inner')
    print(f"  After merging with L1 for POINT_DIFF: {len(result)} games")

    # Save
    os.makedirs('data/processed', exist_ok=True)
    result.to_csv('data/processed/level5_context.csv', index=False)
    print(f"\n  Saved: data/processed/level5_context.csv ({result.shape})")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: RUN MODELS
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Build or load context features
    l5_path = 'data/processed/level5_context.csv'
    if not os.path.exists(l5_path):
        df_l5 = build_context_features()
    else:
        print("Loading cached level5_context.csv...")
        df_l5 = pd.read_csv(l5_path)
        print(f"  Shape: {df_l5.shape}")

    # Define L5 feature columns
    L5_FEAT = [
        'HOME_REST_DAYS', 'HOME_IS_B2B', 'HOME_GAMES_IN_LAST_7',
        'HOME_WIN_STREAK', 'HOME_GAME_NUMBER', 'HOME_CONSECUTIVE_AWAY',
        'HOME_HOME_WIN_PCT', 'HOME_AWAY_WIN_PCT', 'HOME_OVERALL_WIN_PCT',
        'AWAY_REST_DAYS', 'AWAY_IS_B2B', 'AWAY_GAMES_IN_LAST_7',
        'AWAY_WIN_STREAK', 'AWAY_GAME_NUMBER', 'AWAY_CONSECUTIVE_AWAY',
        'AWAY_HOME_WIN_PCT', 'AWAY_AWAY_WIN_PCT', 'AWAY_OVERALL_WIN_PCT',
        'REST_ADVANTAGE', 'DIFF_WIN_STREAK', 'DIFF_GAMES_IN_LAST_7',
        'DIFF_GAME_NUMBER', 'DIFF_OVERALL_WIN_PCT', 'HOME_ADV',
    ]
    L5_FEAT = [c for c in L5_FEAT if c in df_l5.columns]
    print(f"\nL5 features ({len(L5_FEAT)}): {L5_FEAT}")

    # ── Prepare L5 standalone data ────────────────────────────────────────
    all_results = []

    df_clean = df_l5.dropna(subset=L5_FEAT + ['POINT_DIFF'])
    df_tr = df_clean[df_clean['SEASON'].isin(TRAIN_SEASONS)]
    df_v  = df_clean[df_clean['SEASON'].isin(VAL_SEASONS)]
    df_te = df_clean[df_clean['SEASON'].isin(TEST_SEASONS)]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(df_tr[L5_FEAT].values.astype(np.float32))
    X_v  = scaler.transform(df_v[L5_FEAT].values.astype(np.float32))
    X_te = scaler.transform(df_te[L5_FEAT].values.astype(np.float32))
    y_tr = df_tr['POINT_DIFF'].values.astype(np.float32)
    y_v  = df_v['POINT_DIFF'].values.astype(np.float32)
    y_te = df_te['POINT_DIFF'].values.astype(np.float32)

    print(f"\nL5 standalone — Train: {X_tr.shape}, Val: {X_v.shape}, Test: {X_te.shape}")

    print("\n" + "=" * 70)
    print("EXPERIMENT 1: L5 STANDALONE (context features only)")
    print("=" * 70)
    res, _, xgb_l5, _ = run_experiment('L5_standalone', X_tr, y_tr, X_v, y_v, X_te, y_te)
    all_results.extend(res)

    # Print XGBoost feature importance for L5
    print("\n  XGBoost feature importance (L5):")
    imp = pd.Series(xgb_l5.feature_importances_, index=L5_FEAT).sort_values(ascending=False)
    for feat, score in imp.head(15).items():
        print(f"    {feat}: {score:.4f}")

    # ── Prepare L5 + L1 combined ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: L5 + L1 COMBINED (context + season stats)")
    print("=" * 70)

    # Load L1 and build its features (same as tune.py)
    df_l1 = pd.read_csv('data/processed/level1_season_agg.csv')
    L1_STAT_NAMES = ['GP', 'W_PCT', 'PTS_y', 'REB', 'AST', 'STL', 'BLK', 'TOV',
                     'FG_PCT', 'FG3_PCT', 'FT_PCT', 'OFF_RATING', 'DEF_RATING',
                     'NET_RATING', 'PACE', 'EFG_PCT', 'TM_TOV_PCT', 'OREB_PCT']
    L1_FEAT = ([f'HOME_{s}' for s in L1_STAT_NAMES if f'HOME_{s}' in df_l1.columns] +
               [f'AWAY_{s}' for s in L1_STAT_NAMES if f'AWAY_{s}' in df_l1.columns])
    for s in L1_STAT_NAMES:
        hc, ac = f'HOME_{s}', f'AWAY_{s}'
        if hc in df_l1.columns and ac in df_l1.columns:
            df_l1[f'DIFF_{s}'] = df_l1[hc] - df_l1[ac]
            L1_FEAT.append(f'DIFF_{s}')

    # Merge L5 context features into L1
    l5_merge_cols = ['GAME_ID'] + [c for c in L5_FEAT if c != 'HOME_ADV']  # HOME_ADV already in L1
    df_combined = df_l1.merge(df_l5[l5_merge_cols], on='GAME_ID', how='inner')
    df_combined['HOME_ADV'] = 1.0

    COMBINED_FEAT = L1_FEAT + ['HOME_ADV'] + [c for c in L5_FEAT if c != 'HOME_ADV']
    # Remove duplicates
    COMBINED_FEAT = list(dict.fromkeys(COMBINED_FEAT))
    COMBINED_FEAT = [c for c in COMBINED_FEAT if c in df_combined.columns]

    print(f"  Combined features: {len(COMBINED_FEAT)} ({len(L1_FEAT)} L1 + {len(L5_FEAT)} L5, minus overlap)")

    df_clean = df_combined.dropna(subset=COMBINED_FEAT + ['POINT_DIFF'])
    df_tr = df_clean[df_clean['SEASON'].isin(TRAIN_SEASONS)]
    df_v  = df_clean[df_clean['SEASON'].isin(VAL_SEASONS)]
    df_te = df_clean[df_clean['SEASON'].isin(TEST_SEASONS)]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(df_tr[COMBINED_FEAT].values.astype(np.float32))
    X_v  = scaler.transform(df_v[COMBINED_FEAT].values.astype(np.float32))
    X_te = scaler.transform(df_te[COMBINED_FEAT].values.astype(np.float32))
    y_tr = df_tr['POINT_DIFF'].values.astype(np.float32)
    y_v  = df_v['POINT_DIFF'].values.astype(np.float32)
    y_te = df_te['POINT_DIFF'].values.astype(np.float32)

    print(f"  Train: {X_tr.shape}, Val: {X_v.shape}, Test: {X_te.shape}")

    res, _, xgb_combined, _ = run_experiment('L5+L1_combined', X_tr, y_tr, X_v, y_v, X_te, y_te)
    all_results.extend(res)

    # Print XGBoost feature importance for combined
    print("\n  XGBoost feature importance (L5+L1 combined, top 20):")
    imp = pd.Series(xgb_combined.feature_importances_, index=COMBINED_FEAT).sort_values(ascending=False)
    for feat, score in imp.head(20).items():
        print(f"    {feat}: {score:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    df_all = pd.DataFrame(all_results)
    print(df_all[['experiment', 'model', 'mae', 'mse', 'r2']].to_string(index=False))

    print("\n── Comparison to L1 baseline (MAE=10.61) ──")
    for _, row in df_all.iterrows():
        delta = row['mae'] - 10.61
        direction = "worse" if delta > 0 else "better"
        print(f"  {row['experiment']:20s} {row['model']:10s}  MAE={row['mae']:.3f}  ({delta:+.3f}, {direction})")

    # Save results
    os.makedirs('results', exist_ok=True)
    df_all.to_csv('results/level5_results.csv', index=False)
    print(f"\nSaved: results/level5_results.csv")
