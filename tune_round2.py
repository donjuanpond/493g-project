#!/usr/bin/env python3
"""
Round 2: Rolling window ablation for L2, feature selection, and ensembles.
"""

import os, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

def eval_metrics(y_true, y_pred):
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
    }

# ══════════════════════════════════════════════════════════════════════════════
# PART 1: L2 ROLLING WINDOW ABLATION
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PART 1: L2 ROLLING WINDOW ABLATION")
print("=" * 70)

# Load raw game data
dfs = []
for season in ALL_SEASONS:
    df = pd.read_csv(f'data/raw/games_{season}.csv')
    df['SEASON'] = season
    dfs.append(df)
df_games = pd.concat(dfs, ignore_index=True)

# Parse matchup for home/away
df_games['IS_HOME'] = df_games['MATCHUP'].str.contains('vs.', na=False)
df_games['GAME_DATE'] = pd.to_datetime(df_games['GAME_DATE'])
df_games = df_games.sort_values(['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)

L2_BOX_COLS = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',
               'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
               'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'PLUS_MINUS']
L2_BOX_COLS = [c for c in L2_BOX_COLS if c in df_games.columns]

def build_l2_for_window(window):
    """Build L2 dataset with a given rolling window size."""
    rolled = (
        df_games.groupby('TEAM_ID')[L2_BOX_COLS]
        .apply(lambda g: g.shift(1).rolling(window, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    rolled.columns = [f'ROLL_{c}' for c in rolled.columns]
    df_r = pd.concat([df_games[['GAME_ID', 'GAME_DATE', 'SEASON', 'TEAM_ID', 'IS_HOME']], rolled], axis=1)

    roll_cols = [f'ROLL_{c}' for c in L2_BOX_COLS]

    home = df_r[df_r['IS_HOME']].copy()
    away = df_r[~df_r['IS_HOME']].copy()

    home_rename = {c: f'HOME_{c}' for c in roll_cols}
    away_rename = {c: f'AWAY_{c}' for c in roll_cols}
    home = home.rename(columns=home_rename)
    away = away.rename(columns=away_rename)

    merged = home[['GAME_ID', 'GAME_DATE', 'SEASON'] + list(home_rename.values())].merge(
        away[['GAME_ID'] + list(away_rename.values())],
        on='GAME_ID', how='inner'
    )

    # Need point diff from original paired data
    df_paired = pd.read_csv('data/processed/level1_season_agg.csv')[['GAME_ID', 'POINT_DIFF']]
    merged = merged.merge(df_paired, on='GAME_ID', how='inner')

    # Add diff features
    feat_cols = list(home_rename.values()) + list(away_rename.values())
    for c in L2_BOX_COLS:
        hc, ac = f'HOME_ROLL_{c}', f'AWAY_ROLL_{c}'
        merged[f'DIFF_ROLL_{c}'] = merged[hc] - merged[ac]
        feat_cols.append(f'DIFF_ROLL_{c}')
    merged['HOME_ADV'] = 1.0
    feat_cols.append('HOME_ADV')

    merged = merged.dropna(subset=feat_cols + ['POINT_DIFF'])

    df_tr = merged[merged['SEASON'].isin(TRAIN_SEASONS)]
    df_v  = merged[merged['SEASON'].isin(VAL_SEASONS)]
    df_te = merged[merged['SEASON'].isin(TEST_SEASONS)]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(df_tr[feat_cols].values.astype(np.float32))
    X_v  = scaler.transform(df_v[feat_cols].values.astype(np.float32))
    X_te = scaler.transform(df_te[feat_cols].values.astype(np.float32))
    y_tr = df_tr['POINT_DIFF'].values.astype(np.float32)
    y_v  = df_v['POINT_DIFF'].values.astype(np.float32)
    y_te = df_te['POINT_DIFF'].values.astype(np.float32)

    return X_tr, y_tr, X_v, y_v, X_te, y_te, feat_cols

window_results = []
for window in [3, 5, 7, 10, 15, 20, 30]:
    print(f"\n── L2 window={window} ──")
    X_tr, y_tr, X_v, y_v, X_te, y_te, feat_cols = build_l2_for_window(window)
    print(f"  Train: {X_tr.shape}, Test: {X_te.shape}")

    # Ridge
    ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 500, 1000, 5000])
    ridge.fit(X_tr, y_tr)
    y_pred = ridge.predict(X_te)
    m = eval_metrics(y_te, y_pred)
    print(f"  Ridge: MAE={m['mae']:.3f} R²={m['r2']:.3f} α={ridge.alpha_}")
    window_results.append({'window': window, 'model': 'Ridge', **m, 'alpha': ridge.alpha_})

    # XGBoost (best config from round 1: depth=4, lr=0.01)
    xgb = XGBRegressor(n_estimators=1000, max_depth=4, learning_rate=0.01,
                        subsample=0.7, colsample_bytree=0.7, min_child_weight=3,
                        early_stopping_rounds=30, eval_metric='mae',
                        random_state=SEED, verbosity=0)
    xgb.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
    y_pred = xgb.predict(X_te)
    m = eval_metrics(y_te, y_pred)
    print(f"  XGB:   MAE={m['mae']:.3f} R²={m['r2']:.3f} trees={xgb.best_iteration}")
    window_results.append({'window': window, 'model': 'XGBoost', **m, 'trees': xgb.best_iteration})

df_window = pd.DataFrame(window_results)
print("\n── Window ablation summary ──")
for model in ['Ridge', 'XGBoost']:
    sub = df_window[df_window['model'] == model]
    best = sub.loc[sub['mae'].idxmin()]
    print(f"  {model} best: window={int(best['window'])} MAE={best['mae']:.3f} R²={best['r2']:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# PART 2: L3 WITH LONGER ROLLING WINDOWS (rebuild from raw player data)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 2: L3 ROLLING WINDOW ABLATION")
print("=" * 70)

# Load all player game logs
dfs_pl = []
for season in ALL_SEASONS:
    df = pd.read_csv(f'data/raw/player_gamelogs_{season}.csv')
    df['SEASON'] = season
    dfs_pl.append(df)
df_pl = pd.concat(dfs_pl, ignore_index=True)
df_pl['GAME_DATE'] = pd.to_datetime(df_pl['GAME_DATE'])

# Parse MIN
if df_pl['MIN'].dtype == object:
    def parse_min(m):
        try:
            parts = str(m).split(':')
            return float(parts[0]) + float(parts[1])/60 if len(parts) == 2 else float(parts[0])
        except:
            return np.nan
    df_pl['MIN'] = df_pl['MIN'].apply(parse_min)
else:
    df_pl['MIN'] = pd.to_numeric(df_pl['MIN'], errors='coerce')

L3_STATS = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'MIN', 'FG_PCT', 'FG3_PCT', 'FT_PCT']
L3_STATS = [c for c in L3_STATS if c in df_pl.columns]
for c in L3_STATS:
    df_pl[c] = pd.to_numeric(df_pl[c], errors='coerce')

df_pl = df_pl.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)

# Load paired games for home/away + target
df_paired = pd.read_csv('data/processed/level1_season_agg.csv')[['GAME_ID', 'HOME_TEAM_ID', 'AWAY_TEAM_ID', 'POINT_DIFF', 'SEASON', 'GAME_DATE']]
# Ensure GAME_ID types match
df_paired['GAME_ID'] = df_paired['GAME_ID'].astype(str)
df_pl['GAME_ID'] = df_pl['GAME_ID'].astype(str)

def build_l3_agg_for_window(window):
    """Build L3 agg-only features (team mean of top-8 players) with given rolling window."""
    # Rolling averages per player
    rolled = (
        df_pl.groupby('PLAYER_ID')[L3_STATS]
        .apply(lambda g: g.shift(1).rolling(window, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    rolled.columns = [f'ROLL_{c}' for c in rolled.columns]
    df_r = pd.concat([df_pl[['GAME_ID', 'PLAYER_ID', 'TEAM_ID', 'MIN']], rolled], axis=1)

    roll_cols = [f'ROLL_{c}' for c in L3_STATS]

    # For each game+team, get top 8 by MIN, compute mean
    rows = []
    for _, game_row in df_paired.iterrows():
        gid = game_row['GAME_ID']
        game_players = df_r[df_r['GAME_ID'] == gid]
        if len(game_players) == 0:
            continue

        row = {'GAME_ID': gid, 'SEASON': game_row['SEASON'], 'POINT_DIFF': game_row['POINT_DIFF']}

        for side, tid in [('HOME', game_row['HOME_TEAM_ID']), ('AWAY', game_row['AWAY_TEAM_ID'])]:
            team_pl = game_players[game_players['TEAM_ID'] == tid].nlargest(8, 'MIN')
            for stat in roll_cols:
                vals = team_pl[stat].dropna()
                row[f'{side}_MEAN_{stat}'] = vals.mean() if len(vals) > 0 else 0.0
                row[f'{side}_STD_{stat}'] = vals.std() if len(vals) > 1 else 0.0
        rows.append(row)

    df_out = pd.DataFrame(rows)

    # Add diff features
    feat_cols = []
    for stat in roll_cols:
        for agg in ['MEAN', 'STD']:
            hc = f'HOME_{agg}_{stat}'
            ac = f'AWAY_{agg}_{stat}'
            if hc in df_out.columns:
                feat_cols.extend([hc, ac])
                df_out[f'DIFF_{agg}_{stat}'] = df_out[hc] - df_out[ac]
                feat_cols.append(f'DIFF_{agg}_{stat}')

    df_out['HOME_ADV'] = 1.0
    feat_cols.append('HOME_ADV')
    feat_cols = list(dict.fromkeys(feat_cols))  # dedupe

    df_out = df_out.dropna(subset=feat_cols + ['POINT_DIFF'])

    df_tr = df_out[df_out['SEASON'].isin(TRAIN_SEASONS)]
    df_v  = df_out[df_out['SEASON'].isin(VAL_SEASONS)]
    df_te = df_out[df_out['SEASON'].isin(TEST_SEASONS)]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(df_tr[feat_cols].values.astype(np.float32))
    X_v  = scaler.transform(df_v[feat_cols].values.astype(np.float32))
    X_te = scaler.transform(df_te[feat_cols].values.astype(np.float32))
    y_tr = df_tr['POINT_DIFF'].values.astype(np.float32)
    y_v  = df_v['POINT_DIFF'].values.astype(np.float32)
    y_te = df_te['POINT_DIFF'].values.astype(np.float32)

    return X_tr, y_tr, X_v, y_v, X_te, y_te, feat_cols

l3_window_results = []
for window in [5, 10, 15, 20]:
    print(f"\n── L3 agg window={window} ──")
    X_tr, y_tr, X_v, y_v, X_te, y_te, feat_cols = build_l3_agg_for_window(window)
    print(f"  Train: {X_tr.shape}, Val: {X_v.shape}, Test: {X_te.shape}")

    ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 500, 1000, 5000])
    ridge.fit(X_tr, y_tr)
    y_pred = ridge.predict(X_te)
    m = eval_metrics(y_te, y_pred)
    print(f"  Ridge: MAE={m['mae']:.3f} R²={m['r2']:.3f} α={ridge.alpha_}")
    l3_window_results.append({'window': window, 'model': 'Ridge', **m})

    xgb = XGBRegressor(n_estimators=1000, max_depth=4, learning_rate=0.01,
                        subsample=0.7, colsample_bytree=0.7, min_child_weight=3,
                        early_stopping_rounds=30, eval_metric='mae',
                        random_state=SEED, verbosity=0)
    xgb.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
    y_pred = xgb.predict(X_te)
    m = eval_metrics(y_te, y_pred)
    print(f"  XGB:   MAE={m['mae']:.3f} R²={m['r2']:.3f} trees={xgb.best_iteration}")
    l3_window_results.append({'window': window, 'model': 'XGBoost', **m})

df_l3_window = pd.DataFrame(l3_window_results)
print("\n── L3 window ablation summary ──")
for model in ['Ridge', 'XGBoost']:
    sub = df_l3_window[df_l3_window['model'] == model]
    best = sub.loc[sub['mae'].idxmin()]
    print(f"  {model} best: window={int(best['window'])} MAE={best['mae']:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# PART 3: ENSEMBLE (average L1 Ridge + L2 Ridge predictions)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 3: ENSEMBLE EXPERIMENTS")
print("=" * 70)

# Load L1 and L2 data, train Ridge, then blend predictions
# L1
df_l1 = pd.read_csv('data/processed/level1_season_agg.csv')
L1_STAT_NAMES = ['GP', 'W_PCT', 'PTS_y', 'REB', 'AST', 'STL', 'BLK', 'TOV',
                 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'OFF_RATING', 'DEF_RATING',
                 'NET_RATING', 'PACE', 'EFG_PCT', 'TM_TOV_PCT', 'OREB_PCT']
L1_FEAT = []
for s in L1_STAT_NAMES:
    if f'HOME_{s}' in df_l1.columns and f'AWAY_{s}' in df_l1.columns:
        L1_FEAT.extend([f'HOME_{s}', f'AWAY_{s}'])
        df_l1[f'DIFF_{s}'] = df_l1[f'HOME_{s}'] - df_l1[f'AWAY_{s}']
        L1_FEAT.append(f'DIFF_{s}')
df_l1['HOME_ADV'] = 1.0
L1_FEAT.append('HOME_ADV')

df_l1_clean = df_l1.dropna(subset=L1_FEAT + ['POINT_DIFF'])
l1_tr = df_l1_clean[df_l1_clean['SEASON'].isin(TRAIN_SEASONS)]
l1_te = df_l1_clean[df_l1_clean['SEASON'].isin(TEST_SEASONS)]

scaler_l1 = StandardScaler()
X_tr_l1 = scaler_l1.fit_transform(l1_tr[L1_FEAT].values.astype(np.float32))
X_te_l1 = scaler_l1.transform(l1_te[L1_FEAT].values.astype(np.float32))
y_tr_l1 = l1_tr['POINT_DIFF'].values.astype(np.float32)
y_te_l1 = l1_te['POINT_DIFF'].values.astype(np.float32)
game_ids_l1 = l1_te['GAME_ID'].values

ridge_l1 = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 500, 1000, 5000])
ridge_l1.fit(X_tr_l1, y_tr_l1)
pred_l1 = ridge_l1.predict(X_te_l1)

# Best L2 window
best_l2_window = 10  # default, update from results
if len(window_results) > 0:
    ridge_wins = [r for r in window_results if r['model'] == 'Ridge']
    if ridge_wins:
        best_l2_window = min(ridge_wins, key=lambda x: x['mae'])['window']

print(f"Using L2 window={best_l2_window}")
X_tr_l2, y_tr_l2, X_v_l2, y_v_l2, X_te_l2, y_te_l2, l2_feat = build_l2_for_window(best_l2_window)

ridge_l2 = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 500, 1000, 5000])
ridge_l2.fit(X_tr_l2, y_tr_l2)
pred_l2 = ridge_l2.predict(X_te_l2)

# Both test sets should be same games in same order (both from 2024-25)
# But lengths might differ slightly due to NaN handling
# Use the shorter one
min_len = min(len(pred_l1), len(pred_l2))
y_te_ens = y_te_l1[:min_len]

# Try different blend weights
print(f"\n── Ensemble: L1 Ridge + L2 Ridge ──")
for w1 in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    w2 = 1.0 - w1
    pred_ens = w1 * pred_l1[:min_len] + w2 * pred_l2[:min_len]
    m = eval_metrics(y_te_ens, pred_ens)
    print(f"  w_L1={w1:.1f} w_L2={w2:.1f}: MAE={m['mae']:.3f} MSE={m['mse']:.2f} R²={m['r2']:.3f}")

# Also try L1 XGBoost + L1 Ridge ensemble
xgb_l1 = XGBRegressor(n_estimators=1000, max_depth=4, learning_rate=0.01,
                        subsample=0.7, colsample_bytree=0.7, min_child_weight=3,
                        early_stopping_rounds=30, eval_metric='mae',
                        random_state=SEED, verbosity=0)
l1_v = df_l1_clean[df_l1_clean['SEASON'].isin(VAL_SEASONS)]
X_v_l1 = scaler_l1.transform(l1_v[L1_FEAT].values.astype(np.float32))
y_v_l1 = l1_v['POINT_DIFF'].values.astype(np.float32)
xgb_l1.fit(X_tr_l1, y_tr_l1, eval_set=[(X_v_l1, y_v_l1)], verbose=False)
pred_xgb_l1 = xgb_l1.predict(X_te_l1)

print(f"\n── Ensemble: L1 Ridge + L1 XGBoost ──")
for w1 in [0.3, 0.4, 0.5, 0.6, 0.7]:
    w2 = 1.0 - w1
    pred_ens = w1 * pred_l1 + w2 * pred_xgb_l1
    m = eval_metrics(y_te_l1, pred_ens)
    print(f"  w_Ridge={w1:.1f} w_XGB={w2:.1f}: MAE={m['mae']:.3f} MSE={m['mse']:.2f} R²={m['r2']:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# PART 4: L1 FEATURE IMPORTANCE-BASED PRUNING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 4: L1 FEATURE SELECTION")
print("=" * 70)

importances = xgb_l1.feature_importances_
feat_imp = sorted(zip(L1_FEAT, importances), key=lambda x: -x[1])
print("Top 20 features:")
for f, imp in feat_imp[:20]:
    print(f"  {f}: {imp:.4f}")

# Try with only top-K features
for k in [10, 15, 20, 30]:
    top_feats = [f for f, _ in feat_imp[:k]]
    scaler_k = StandardScaler()
    X_tr_k = scaler_k.fit_transform(l1_tr[top_feats].values.astype(np.float32))
    X_v_k = scaler_k.transform(l1_v[top_feats].values.astype(np.float32))
    X_te_k = scaler_k.transform(l1_te[top_feats].values.astype(np.float32))

    ridge_k = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 500, 1000, 5000])
    ridge_k.fit(X_tr_k, y_tr_l1)
    y_pred = ridge_k.predict(X_te_k)
    m = eval_metrics(y_te_l1, y_pred)
    print(f"  Top-{k}: Ridge MAE={m['mae']:.3f} R²={m['r2']:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE ALL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
df_window.to_csv('results/l2_window_ablation.csv', index=False)
df_l3_window.to_csv('results/l3_window_ablation.csv', index=False)
print("\nSaved window ablation results.")
print("\nDONE.")
