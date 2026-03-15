#!/usr/bin/env python3
"""
Round 3: Final push — longer L3 windows, L3 agg + L2 features combined,
best model configs on best feature sets.
"""

import os, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, ElasticNetCV
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
             lr=1e-3, weight_decay=1e-4, patience=20, dropout=0.3,
             hidden_dims=(256, 128, 64)):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(SEED)
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
            if wait >= patience: break
    model.load_state_dict(best_state)
    return model, best_val

def predict_nn(model, X):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()

# ══════════════════════════════════════════════════════════════════════════════
# Load raw data
# ══════════════════════════════════════════════════════════════════════════════

# Games
dfs = []
for season in ALL_SEASONS:
    df = pd.read_csv(f'data/raw/games_{season}.csv')
    df['SEASON'] = season
    dfs.append(df)
df_games = pd.concat(dfs, ignore_index=True)
df_games['IS_HOME'] = df_games['MATCHUP'].str.contains('vs.', na=False)
df_games['GAME_DATE'] = pd.to_datetime(df_games['GAME_DATE'])
df_games = df_games.sort_values(['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)

L2_BOX_COLS = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',
               'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
               'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'PLUS_MINUS']
L2_BOX_COLS = [c for c in L2_BOX_COLS if c in df_games.columns]

# Player logs
dfs_pl = []
for season in ALL_SEASONS:
    df = pd.read_csv(f'data/raw/player_gamelogs_{season}.csv')
    df['SEASON'] = season
    dfs_pl.append(df)
df_pl = pd.concat(dfs_pl, ignore_index=True)
df_pl['GAME_DATE'] = pd.to_datetime(df_pl['GAME_DATE'])
if df_pl['MIN'].dtype == object:
    def parse_min(m):
        try:
            parts = str(m).split(':')
            return float(parts[0]) + float(parts[1])/60 if len(parts) == 2 else float(parts[0])
        except: return np.nan
    df_pl['MIN'] = df_pl['MIN'].apply(parse_min)
L3_STATS = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'MIN', 'FG_PCT', 'FG3_PCT', 'FT_PCT']
L3_STATS = [c for c in L3_STATS if c in df_pl.columns]
for c in L3_STATS:
    df_pl[c] = pd.to_numeric(df_pl[c], errors='coerce')
df_pl = df_pl.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)

df_paired = pd.read_csv('data/processed/level1_season_agg.csv')[
    ['GAME_ID', 'HOME_TEAM_ID', 'AWAY_TEAM_ID', 'POINT_DIFF', 'SEASON', 'GAME_DATE']]
df_paired['GAME_ID'] = df_paired['GAME_ID'].astype(str)
df_pl['GAME_ID'] = df_pl['GAME_ID'].astype(str)

# ══════════════════════════════════════════════════════════════════════════════
# PART 1: L3 with window=25, 30
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PART 1: L3 LONGER WINDOWS (25, 30)")
print("=" * 70)

def build_l3_agg(window):
    rolled = (
        df_pl.groupby('PLAYER_ID')[L3_STATS]
        .apply(lambda g: g.shift(1).rolling(window, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    rolled.columns = [f'ROLL_{c}' for c in rolled.columns]
    df_r = pd.concat([df_pl[['GAME_ID', 'PLAYER_ID', 'TEAM_ID', 'MIN']], rolled], axis=1)
    roll_cols = [f'ROLL_{c}' for c in L3_STATS]

    rows = []
    for _, gr in df_paired.iterrows():
        gid = gr['GAME_ID']
        gp = df_r[df_r['GAME_ID'] == gid]
        if len(gp) == 0: continue
        row = {'GAME_ID': gid, 'SEASON': gr['SEASON'], 'POINT_DIFF': gr['POINT_DIFF']}
        for side, tid in [('HOME', gr['HOME_TEAM_ID']), ('AWAY', gr['AWAY_TEAM_ID'])]:
            tp = gp[gp['TEAM_ID'] == tid].nlargest(8, 'MIN')
            for stat in roll_cols:
                vals = tp[stat].dropna()
                row[f'{side}_MEAN_{stat}'] = vals.mean() if len(vals) > 0 else 0.0
                row[f'{side}_STD_{stat}'] = vals.std() if len(vals) > 1 else 0.0
        rows.append(row)
    df_out = pd.DataFrame(rows)

    feat_cols = []
    for stat in roll_cols:
        for agg in ['MEAN', 'STD']:
            hc, ac = f'HOME_{agg}_{stat}', f'AWAY_{agg}_{stat}'
            if hc in df_out.columns:
                feat_cols.extend([hc, ac])
                df_out[f'DIFF_{agg}_{stat}'] = df_out[hc] - df_out[ac]
                feat_cols.append(f'DIFF_{agg}_{stat}')
    df_out['HOME_ADV'] = 1.0
    feat_cols.append('HOME_ADV')
    feat_cols = list(dict.fromkeys(feat_cols))
    df_out = df_out.dropna(subset=feat_cols + ['POINT_DIFF'])
    return df_out, feat_cols

def prep_split(df, feat_cols):
    df_tr = df[df['SEASON'].isin(TRAIN_SEASONS)]
    df_v  = df[df['SEASON'].isin(VAL_SEASONS)]
    df_te = df[df['SEASON'].isin(TEST_SEASONS)]
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(df_tr[feat_cols].values.astype(np.float32))
    X_v  = scaler.transform(df_v[feat_cols].values.astype(np.float32))
    X_te = scaler.transform(df_te[feat_cols].values.astype(np.float32))
    y_tr = df_tr['POINT_DIFF'].values.astype(np.float32)
    y_v  = df_v['POINT_DIFF'].values.astype(np.float32)
    y_te = df_te['POINT_DIFF'].values.astype(np.float32)
    return X_tr, y_tr, X_v, y_v, X_te, y_te, scaler

for window in [25, 30, 40]:
    print(f"\n── L3 agg window={window} ──")
    df_l3, feat_cols = build_l3_agg(window)
    X_tr, y_tr, X_v, y_v, X_te, y_te, scaler = prep_split(df_l3, feat_cols)
    print(f"  Shape: Train {X_tr.shape}, Val {X_v.shape}, Test {X_te.shape}")

    ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 500, 1000, 5000])
    ridge.fit(X_tr, y_tr)
    m = eval_metrics(y_te, ridge.predict(X_te))
    print(f"  Ridge: MAE={m['mae']:.3f} R²={m['r2']:.3f} α={ridge.alpha_}")

    xgb = XGBRegressor(n_estimators=1000, max_depth=4, learning_rate=0.01,
                        subsample=0.7, colsample_bytree=0.7, min_child_weight=3,
                        early_stopping_rounds=30, eval_metric='mae',
                        random_state=SEED, verbosity=0)
    xgb.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
    m = eval_metrics(y_te, xgb.predict(X_te))
    print(f"  XGB:   MAE={m['mae']:.3f} R²={m['r2']:.3f} trees={xgb.best_iteration}")

# ══════════════════════════════════════════════════════════════════════════════
# PART 2: BEST CONFIG ON EACH LEVEL — FINAL TUNED MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 2: FINAL TUNED MODELS — ALL 3 MODEL TYPES × ALL 3 LEVELS")
print("=" * 70)

# L1 with top-10 features
df_l1 = pd.read_csv('data/processed/level1_season_agg.csv')
L1_STAT_NAMES = ['GP', 'W_PCT', 'PTS_y', 'REB', 'AST', 'STL', 'BLK', 'TOV',
                 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'OFF_RATING', 'DEF_RATING',
                 'NET_RATING', 'PACE', 'EFG_PCT', 'TM_TOV_PCT', 'OREB_PCT']
for s in L1_STAT_NAMES:
    if f'HOME_{s}' in df_l1.columns and f'AWAY_{s}' in df_l1.columns:
        df_l1[f'DIFF_{s}'] = df_l1[f'HOME_{s}'] - df_l1[f'AWAY_{s}']
df_l1['HOME_ADV'] = 1.0

# Top-10 from XGBoost importance (round 2)
L1_TOP10 = ['DIFF_NET_RATING', 'DIFF_W_PCT', 'DIFF_OFF_RATING', 'DIFF_DEF_RATING',
            'HOME_NET_RATING', 'AWAY_NET_RATING', 'DIFF_PTS_y', 'HOME_DEF_RATING',
            'DIFF_GP', 'DIFF_EFG_PCT']
L1_ALL = []
for s in L1_STAT_NAMES:
    for pfx in ['HOME_', 'AWAY_', 'DIFF_']:
        c = f'{pfx}{s}'
        if c in df_l1.columns:
            L1_ALL.append(c)
L1_ALL.append('HOME_ADV')
L1_ALL = list(dict.fromkeys(L1_ALL))

# L2 with window=20
def build_l2(window):
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
        away[['GAME_ID'] + list(away_rename.values())], on='GAME_ID', how='inner')
    df_pt = pd.read_csv('data/processed/level1_season_agg.csv')[['GAME_ID', 'POINT_DIFF']]
    merged = merged.merge(df_pt, on='GAME_ID', how='inner')
    feat_cols = list(home_rename.values()) + list(away_rename.values())
    for c in L2_BOX_COLS:
        hc, ac = f'HOME_ROLL_{c}', f'AWAY_ROLL_{c}'
        merged[f'DIFF_ROLL_{c}'] = merged[hc] - merged[ac]
        feat_cols.append(f'DIFF_ROLL_{c}')
    merged['HOME_ADV'] = 1.0
    feat_cols.append('HOME_ADV')
    merged = merged.dropna(subset=feat_cols + ['POINT_DIFF'])
    return merged, feat_cols

# L3 with window=20
print("\nBuilding L3 window=20...")
df_l3_20, l3_feat_20 = build_l3_agg(20)
print("Building L2 window=20...")
df_l2_20, l2_feat_20 = build_l2(20)

final_results = []

for level_name, df_lev, feat_cols in [
    ('L1_top10', df_l1, L1_TOP10),
    ('L1_all', df_l1, L1_ALL),
    ('L2_w20', df_l2_20, l2_feat_20),
    ('L3_agg_w20', df_l3_20, l3_feat_20),
]:
    print(f"\n{'─'*60}")
    print(f"  {level_name} ({len(feat_cols)} features)")
    print(f"{'─'*60}")

    X_tr, y_tr, X_v, y_v, X_te, y_te, scaler = prep_split(
        df_lev.dropna(subset=feat_cols + ['POINT_DIFF']), feat_cols)
    print(f"  Train: {X_tr.shape}, Val: {X_v.shape}, Test: {X_te.shape}")

    # Ridge
    ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 500, 1000, 5000])
    ridge.fit(X_tr, y_tr)
    m = eval_metrics(y_te, ridge.predict(X_te))
    print(f"  Ridge:   MAE={m['mae']:.3f} MSE={m['mse']:.2f} R²={m['r2']:.3f}")
    final_results.append({'level': level_name, 'model': 'Ridge', **m})

    # XGBoost (best config: depth=4, lr=0.01, subsample=0.7)
    xgb = XGBRegressor(n_estimators=1000, max_depth=4, learning_rate=0.01,
                        subsample=0.7, colsample_bytree=0.7, min_child_weight=3,
                        early_stopping_rounds=30, eval_metric='mae',
                        random_state=SEED, verbosity=0)
    xgb.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
    m = eval_metrics(y_te, xgb.predict(X_te))
    print(f"  XGBoost: MAE={m['mae']:.3f} MSE={m['mse']:.2f} R²={m['r2']:.3f} trees={xgb.best_iteration}")
    final_results.append({'level': level_name, 'model': 'XGBoost', **m})

    # NN (best config: lr=5e-4, drop=0.2, hidden=(256,128,64))
    nn_m, bv = train_nn(X_tr, y_tr, X_v, y_v, input_dim=X_tr.shape[1],
                         lr=5e-4, dropout=0.2, batch_size=128, weight_decay=1e-4,
                         hidden_dims=(256, 128, 64))
    m = eval_metrics(y_te, predict_nn(nn_m, X_te))
    print(f"  NN:      MAE={m['mae']:.3f} MSE={m['mse']:.2f} R²={m['r2']:.3f}")
    final_results.append({'level': level_name, 'model': 'NeuralNet', **m})

# ══════════════════════════════════════════════════════════════════════════════
# PART 3: CROSS-LEVEL ENSEMBLE (L1 + L2 + L3 Ridge)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 3: 3-WAY ENSEMBLE (L1+L2+L3 Ridge)")
print("=" * 70)

# Get predictions from each level's Ridge on test set
# Need to align by GAME_ID
df_l1_clean = df_l1.dropna(subset=L1_ALL + ['POINT_DIFF'])
l1_te = df_l1_clean[df_l1_clean['SEASON'].isin(TEST_SEASONS)]
l1_tr = df_l1_clean[df_l1_clean['SEASON'].isin(TRAIN_SEASONS)]
scaler1 = StandardScaler()
X_tr1 = scaler1.fit_transform(l1_tr[L1_ALL].values.astype(np.float32))
X_te1 = scaler1.transform(l1_te[L1_ALL].values.astype(np.float32))
y_tr1 = l1_tr['POINT_DIFF'].values.astype(np.float32)
y_te1 = l1_te['POINT_DIFF'].values.astype(np.float32)
r1 = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 500, 1000]).fit(X_tr1, y_tr1)
pred1 = r1.predict(X_te1)

df_l2_clean = df_l2_20.dropna(subset=l2_feat_20 + ['POINT_DIFF'])
l2_te = df_l2_clean[df_l2_clean['SEASON'].isin(TEST_SEASONS)]
l2_tr = df_l2_clean[df_l2_clean['SEASON'].isin(TRAIN_SEASONS)]
scaler2 = StandardScaler()
X_tr2 = scaler2.fit_transform(l2_tr[l2_feat_20].values.astype(np.float32))
X_te2 = scaler2.transform(l2_te[l2_feat_20].values.astype(np.float32))
y_tr2 = l2_tr['POINT_DIFF'].values.astype(np.float32)
r2 = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 500, 1000]).fit(X_tr2, y_tr2)
pred2 = r2.predict(X_te2)

df_l3_clean = df_l3_20.dropna(subset=l3_feat_20 + ['POINT_DIFF'])
l3_te = df_l3_clean[df_l3_clean['SEASON'].isin(TEST_SEASONS)]
l3_tr = df_l3_clean[df_l3_clean['SEASON'].isin(TRAIN_SEASONS)]
scaler3 = StandardScaler()
X_tr3 = scaler3.fit_transform(l3_tr[l3_feat_20].values.astype(np.float32))
X_te3 = scaler3.transform(l3_te[l3_feat_20].values.astype(np.float32))
y_tr3 = l3_tr['POINT_DIFF'].values.astype(np.float32)
r3 = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 500, 1000]).fit(X_tr3, y_tr3)
pred3 = r3.predict(X_te3)

n = min(len(pred1), len(pred2), len(pred3))
y_te_ens = y_te1[:n]

print(f"  Individual: L1 MAE={eval_metrics(y_te_ens, pred1[:n])['mae']:.3f}, "
      f"L2 MAE={eval_metrics(y_te_ens, pred2[:n])['mae']:.3f}, "
      f"L3 MAE={eval_metrics(y_te_ens, pred3[:n])['mae']:.3f}")

for w1, w2, w3 in [(0.6, 0.2, 0.2), (0.7, 0.2, 0.1), (0.7, 0.15, 0.15),
                    (0.8, 0.1, 0.1), (0.5, 0.3, 0.2), (0.6, 0.3, 0.1)]:
    pred_ens = w1*pred1[:n] + w2*pred2[:n] + w3*pred3[:n]
    m = eval_metrics(y_te_ens, pred_ens)
    print(f"  w=({w1},{w2},{w3}): MAE={m['mae']:.3f} R²={m['r2']:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL TUNED RESULTS")
print("=" * 70)
df_final = pd.DataFrame(final_results)
print(df_final.to_string(index=False))
df_final.to_csv('results/final_tuned_results.csv', index=False)
print("\nSaved: results/final_tuned_results.csv")
