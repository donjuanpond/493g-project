#!/usr/bin/env python3
"""
Hyperparameter tuning script for NBA Point Differential Prediction.
Loads processed data, runs experiments for L1-L3, logs results.
"""

import os, time, json, warnings, itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
from sklearn.decomposition import PCA
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

# ── Helpers ───────────────────────────────────────────────────────────────────

def eval_metrics(y_true, y_pred):
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'median_ae': median_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
    }

def load_level(path, feature_cols, target='POINT_DIFF'):
    df = pd.read_csv(path)

    # Only keep feature cols that exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    df_train = df[df['SEASON'].isin(TRAIN_SEASONS)]
    df_val   = df[df['SEASON'].isin(VAL_SEASONS)]
    df_test  = df[df['SEASON'].isin(TEST_SEASONS)]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(df_train[feature_cols].values.astype(np.float32))
    X_v  = scaler.transform(df_val[feature_cols].values.astype(np.float32))
    X_te = scaler.transform(df_test[feature_cols].values.astype(np.float32))

    y_tr = df_train[target].values.astype(np.float32)
    y_v  = df_val[target].values.astype(np.float32)
    y_te = df_test[target].values.astype(np.float32)

    return X_tr, y_tr, X_v, y_v, X_te, y_te, feature_cols, scaler

# ── NN definition ─────────────────────────────────────────────────────────────

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

# ── Run an experiment ─────────────────────────────────────────────────────────

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
    xp = xgb_params or {}
    t0 = time.time()
    xgb = XGBRegressor(
        n_estimators=xp.get('n_estimators', 500),
        max_depth=xp.get('max_depth', 6),
        learning_rate=xp.get('learning_rate', 0.05),
        subsample=xp.get('subsample', 0.8),
        colsample_bytree=xp.get('colsample_bytree', 0.8),
        min_child_weight=xp.get('min_child_weight', 1),
        reg_alpha=xp.get('reg_alpha', 0),
        reg_lambda=xp.get('reg_lambda', 1),
        early_stopping_rounds=20,
        eval_metric='mae',
        random_state=SEED,
        verbosity=0,
    )
    xgb.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
    y_pred = xgb.predict(X_te)
    m = eval_metrics(y_te, y_pred)
    m.update({'experiment': name, 'model': 'XGBoost', 'train_time': time.time()-t0,
              'best_iteration': xgb.best_iteration, **{f'xgb_{k}': v for k, v in xp.items()}})
    results.append(m)
    print(f"  XGBoost: MAE={m['mae']:.3f}  MSE={m['mse']:.2f}  R²={m['r2']:.3f}  trees={xgb.best_iteration}")

    # NN
    np_ = nn_params or {}
    t0 = time.time()
    nn_model, best_val = train_nn(
        X_tr, y_tr, X_v, y_v, input_dim=X_tr.shape[1],
        lr=np_.get('lr', 1e-3),
        dropout=np_.get('dropout', 0.3),
        batch_size=np_.get('batch_size', 64),
        weight_decay=np_.get('weight_decay', 1e-4),
        hidden_dims=np_.get('hidden_dims', (256, 128, 64)),
        patience=np_.get('patience', 15),
    )
    y_pred = predict_nn(nn_model, X_te)
    m = eval_metrics(y_te, y_pred)
    m.update({'experiment': name, 'model': 'NeuralNet', 'train_time': time.time()-t0,
              'best_val_mse': best_val, **{f'nn_{k}': v for k, v in np_.items()}})
    results.append(m)
    print(f"  NN:      MAE={m['mae']:.3f}  MSE={m['mse']:.2f}  R²={m['r2']:.3f}")

    return results, ridge, xgb, nn_model

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    all_results = []

    # ── Load processed datasets ───────────────────────────────────────────────
    print("=" * 70)
    print("LOADING PROCESSED DATA")
    print("=" * 70)

    # Level 1 — numeric stat columns only
    df_l1 = pd.read_csv('data/processed/level1_season_agg.csv')
    L1_STAT_NAMES = ['GP', 'W_PCT', 'PTS_y', 'REB', 'AST', 'STL', 'BLK', 'TOV',
                     'FG_PCT', 'FG3_PCT', 'FT_PCT', 'OFF_RATING', 'DEF_RATING',
                     'NET_RATING', 'PACE', 'EFG_PCT', 'TM_TOV_PCT', 'OREB_PCT']
    L1_FEAT = ([f'HOME_{s}' for s in L1_STAT_NAMES if f'HOME_{s}' in df_l1.columns] +
               [f'AWAY_{s}' for s in L1_STAT_NAMES if f'AWAY_{s}' in df_l1.columns])
    # Add differential features on the fly
    for s in L1_STAT_NAMES:
        hc, ac = f'HOME_{s}', f'AWAY_{s}'
        if hc in df_l1.columns and ac in df_l1.columns:
            df_l1[f'DIFF_{s}'] = df_l1[hc] - df_l1[ac]
            L1_FEAT.append(f'DIFF_{s}')
    df_l1['HOME_ADV'] = 1.0
    L1_FEAT.append('HOME_ADV')
    print(f"L1: {len(L1_FEAT)} features, {len(df_l1)} games")

    # Level 2 — rolling stat columns
    df_l2 = pd.read_csv('data/processed/level2_rolling10.csv')
    l2_roll_cols = [c.replace('HOME_', '') for c in df_l2.columns if c.startswith('HOME_ROLL_')]
    L2_FEAT = ([f'HOME_{s}' for s in l2_roll_cols] + [f'AWAY_{s}' for s in l2_roll_cols])
    for s in l2_roll_cols:
        hc, ac = f'HOME_{s}', f'AWAY_{s}'
        if hc in df_l2.columns and ac in df_l2.columns:
            df_l2[f'DIFF_{s}'] = df_l2[hc] - df_l2[ac]
            L2_FEAT.append(f'DIFF_{s}')
    if 'HOME_ADV' not in df_l2.columns:
        df_l2['HOME_ADV'] = 1.0
    L2_FEAT.append('HOME_ADV')
    L2_FEAT = [c for c in L2_FEAT if c in df_l2.columns]
    print(f"L2: {len(L2_FEAT)} features, {len(df_l2)} games")

    # Level 3 — player-level features
    df_l3 = pd.read_csv('data/processed/level3_player_rolling5.csv')
    L3_PLAYER = [c for c in df_l3.columns if c.startswith(('HOME_P', 'AWAY_P'))]
    # Detect stat names from HOME_P1_ prefix
    l3_stat_names = [c.replace('HOME_P1_', '') for c in df_l3.columns if c.startswith('HOME_P1_')]
    # Build team aggregates on the fly (mean of 8 players per side)
    for side in ['HOME', 'AWAY']:
        for stat in l3_stat_names:
            slot_cols = [f'{side}_P{p}_{stat}' for p in range(1, 9) if f'{side}_P{p}_{stat}' in df_l3.columns]
            if slot_cols:
                df_l3[f'{side}_TEAM_{stat}'] = df_l3[slot_cols].mean(axis=1)
    # Also add std (spread of talent)
    for side in ['HOME', 'AWAY']:
        for stat in l3_stat_names:
            slot_cols = [f'{side}_P{p}_{stat}' for p in range(1, 9) if f'{side}_P{p}_{stat}' in df_l3.columns]
            if slot_cols:
                df_l3[f'{side}_TEAM_STD_{stat}'] = df_l3[slot_cols].std(axis=1)
    L3_TEAM_AGG = [c for c in df_l3.columns if c.startswith(('HOME_TEAM_', 'AWAY_TEAM_'))]
    # Differential features
    L3_DIFF = []
    for stat in l3_stat_names:
        hc, ac = f'HOME_TEAM_{stat}', f'AWAY_TEAM_{stat}'
        if hc in df_l3.columns and ac in df_l3.columns:
            df_l3[f'DIFF_TEAM_{stat}'] = df_l3[hc] - df_l3[ac]
            L3_DIFF.append(f'DIFF_TEAM_{stat}')
        hc_s, ac_s = f'HOME_TEAM_STD_{stat}', f'AWAY_TEAM_STD_{stat}'
        if hc_s in df_l3.columns and ac_s in df_l3.columns:
            df_l3[f'DIFF_TEAM_STD_{stat}'] = df_l3[hc_s] - df_l3[ac_s]
            L3_DIFF.append(f'DIFF_TEAM_STD_{stat}')
    L3_FEAT_FULL = L3_PLAYER + L3_TEAM_AGG + L3_DIFF
    L3_FEAT_AGG_ONLY = L3_TEAM_AGG + L3_DIFF
    print(f"L3 full: {len(L3_FEAT_FULL)} features, L3 agg-only: {len(L3_FEAT_AGG_ONLY)} features, {len(df_l3)} games")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1: BASELINE REPLICATION (verify we match notebook results)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 1: BASELINE REPLICATION")
    print("=" * 70)

    for level_name, df, feat_cols in [
        ('L1_baseline', df_l1, L1_FEAT),
        ('L2_baseline', df_l2, L2_FEAT),
        ('L3_baseline_full', df_l3, L3_FEAT_FULL),
    ]:
        print(f"\n── {level_name} ({len(feat_cols)} features) ──")
        X_tr, y_tr, X_v, y_v, X_te, y_te, used_cols, scaler = load_level(
            None, feat_cols, target='POINT_DIFF'
        ) if False else (None,)*8  # placeholder

        # Inline loading since we already have df
        feat_cols_valid = [c for c in feat_cols if c in df.columns]
        df_clean = df.dropna(subset=feat_cols_valid + ['POINT_DIFF'])
        df_tr = df_clean[df_clean['SEASON'].isin(TRAIN_SEASONS)]
        df_v  = df_clean[df_clean['SEASON'].isin(VAL_SEASONS)]
        df_te = df_clean[df_clean['SEASON'].isin(TEST_SEASONS)]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(df_tr[feat_cols_valid].values.astype(np.float32))
        X_v  = scaler.transform(df_v[feat_cols_valid].values.astype(np.float32))
        X_te = scaler.transform(df_te[feat_cols_valid].values.astype(np.float32))
        y_tr = df_tr['POINT_DIFF'].values.astype(np.float32)
        y_v  = df_v['POINT_DIFF'].values.astype(np.float32)
        y_te = df_te['POINT_DIFF'].values.astype(np.float32)

        print(f"  Train: {X_tr.shape}, Val: {X_v.shape}, Test: {X_te.shape}")

        res, _, _, _ = run_experiment(level_name, X_tr, y_tr, X_v, y_v, X_te, y_te)
        all_results.extend(res)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2: L3 — AGG-ONLY FEATURES (team means instead of 160 player slots)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 2: L3 AGG-ONLY (team means + diffs instead of 160 player slots)")
    print("=" * 70)

    feat_cols_valid = [c for c in L3_FEAT_AGG_ONLY if c in df_l3.columns]
    df_clean = df_l3.dropna(subset=feat_cols_valid + ['POINT_DIFF'])
    df_tr = df_clean[df_clean['SEASON'].isin(TRAIN_SEASONS)]
    df_v  = df_clean[df_clean['SEASON'].isin(VAL_SEASONS)]
    df_te = df_clean[df_clean['SEASON'].isin(TEST_SEASONS)]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(df_tr[feat_cols_valid].values.astype(np.float32))
    X_v  = scaler.transform(df_v[feat_cols_valid].values.astype(np.float32))
    X_te = scaler.transform(df_te[feat_cols_valid].values.astype(np.float32))
    y_tr = df_tr['POINT_DIFF'].values.astype(np.float32)
    y_v  = df_v['POINT_DIFF'].values.astype(np.float32)
    y_te = df_te['POINT_DIFF'].values.astype(np.float32)

    print(f"  Train: {X_tr.shape}, Val: {X_v.shape}, Test: {X_te.shape}")
    res, _, _, _ = run_experiment('L3_agg_only', X_tr, y_tr, X_v, y_v, X_te, y_te)
    all_results.extend(res)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 3: L3 — PCA on full player features
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 3: L3 with PCA (reduce 160 player dims)")
    print("=" * 70)

    feat_cols_valid = [c for c in L3_FEAT_FULL if c in df_l3.columns]
    df_clean = df_l3.dropna(subset=feat_cols_valid + ['POINT_DIFF'])
    df_tr = df_clean[df_clean['SEASON'].isin(TRAIN_SEASONS)]
    df_v  = df_clean[df_clean['SEASON'].isin(VAL_SEASONS)]
    df_te = df_clean[df_clean['SEASON'].isin(TEST_SEASONS)]

    scaler = StandardScaler()
    X_tr_raw = scaler.fit_transform(df_tr[feat_cols_valid].values.astype(np.float32))
    X_v_raw  = scaler.transform(df_v[feat_cols_valid].values.astype(np.float32))
    X_te_raw = scaler.transform(df_te[feat_cols_valid].values.astype(np.float32))
    y_tr = df_tr['POINT_DIFF'].values.astype(np.float32)
    y_v  = df_v['POINT_DIFF'].values.astype(np.float32)
    y_te = df_te['POINT_DIFF'].values.astype(np.float32)

    for n_comp in [30, 50]:
        print(f"\n── L3 PCA n_components={n_comp} ──")
        pca = PCA(n_components=n_comp, random_state=SEED)
        X_tr_pca = pca.fit_transform(X_tr_raw)
        X_v_pca  = pca.transform(X_v_raw)
        X_te_pca = pca.transform(X_te_raw)
        var_explained = pca.explained_variance_ratio_.sum()
        print(f"  Variance explained: {var_explained:.3f}")
        print(f"  Train: {X_tr_pca.shape}, Val: {X_v_pca.shape}, Test: {X_te_pca.shape}")

        res, _, _, _ = run_experiment(f'L3_PCA{n_comp}', X_tr_pca.astype(np.float32), y_tr,
                                       X_v_pca.astype(np.float32), y_v,
                                       X_te_pca.astype(np.float32), y_te)
        all_results.extend(res)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 4: XGBoost HYPERPARAMETER TUNING (all levels)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 4: XGBOOST GRID SEARCH")
    print("=" * 70)

    xgb_grid = [
        {'max_depth': 4, 'learning_rate': 0.01, 'subsample': 0.7, 'colsample_bytree': 0.7, 'min_child_weight': 3},
        {'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 1},
        {'max_depth': 6, 'learning_rate': 0.01, 'subsample': 0.7, 'colsample_bytree': 0.7, 'min_child_weight': 3},
        {'max_depth': 6, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 1},
        {'max_depth': 8, 'learning_rate': 0.01, 'subsample': 0.7, 'colsample_bytree': 0.6, 'min_child_weight': 5},
        {'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.9, 'min_child_weight': 1},
        {'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 3, 'reg_alpha': 0.1, 'reg_lambda': 5},
        {'max_depth': 6, 'learning_rate': 0.03, 'subsample': 0.7, 'colsample_bytree': 0.7, 'min_child_weight': 5, 'reg_alpha': 0.5, 'reg_lambda': 10},
    ]

    # Prepare data for each level
    level_data = {}
    for level_name, df, feat_cols in [
        ('L1', df_l1, L1_FEAT),
        ('L2', df_l2, L2_FEAT),
        ('L3_agg', df_l3, L3_FEAT_AGG_ONLY),
    ]:
        feat_valid = [c for c in feat_cols if c in df.columns]
        df_clean = df.dropna(subset=feat_valid + ['POINT_DIFF'])
        df_tr = df_clean[df_clean['SEASON'].isin(TRAIN_SEASONS)]
        df_v  = df_clean[df_clean['SEASON'].isin(VAL_SEASONS)]
        df_te = df_clean[df_clean['SEASON'].isin(TEST_SEASONS)]
        scaler = StandardScaler()
        level_data[level_name] = {
            'X_tr': scaler.fit_transform(df_tr[feat_valid].values.astype(np.float32)),
            'y_tr': df_tr['POINT_DIFF'].values.astype(np.float32),
            'X_v': scaler.transform(df_v[feat_valid].values.astype(np.float32)),
            'y_v': df_v['POINT_DIFF'].values.astype(np.float32),
            'X_te': scaler.transform(df_te[feat_valid].values.astype(np.float32)),
            'y_te': df_te['POINT_DIFF'].values.astype(np.float32),
            'scaler': scaler,
            'feat_cols': feat_valid,
        }

    xgb_tuning_results = []
    for level_name, ld in level_data.items():
        print(f"\n── XGBoost tuning: {level_name} ──")
        for i, params in enumerate(xgb_grid):
            t0 = time.time()
            xgb_m = XGBRegressor(
                n_estimators=1000,
                early_stopping_rounds=30,
                eval_metric='mae',
                random_state=SEED,
                verbosity=0,
                **params,
            )
            xgb_m.fit(ld['X_tr'], ld['y_tr'], eval_set=[(ld['X_v'], ld['y_v'])], verbose=False)
            y_pred_v = xgb_m.predict(ld['X_v'])
            y_pred_t = xgb_m.predict(ld['X_te'])
            m_v = eval_metrics(ld['y_v'], y_pred_v)
            m_t = eval_metrics(ld['y_te'], y_pred_t)
            elapsed = time.time() - t0
            row = {
                'level': level_name, 'config_idx': i, **params,
                'val_mae': m_v['mae'], 'val_mse': m_v['mse'],
                'test_mae': m_t['mae'], 'test_mse': m_t['mse'], 'test_r2': m_t['r2'],
                'trees': xgb_m.best_iteration, 'time': elapsed,
            }
            xgb_tuning_results.append(row)
            print(f"  Config {i}: val_MAE={m_v['mae']:.3f} test_MAE={m_t['mae']:.3f} "
                  f"depth={params['max_depth']} lr={params['learning_rate']} trees={xgb_m.best_iteration}")

    df_xgb_tune = pd.DataFrame(xgb_tuning_results)

    # Best XGB per level
    print("\n── Best XGBoost configs (by val MAE) ──")
    for level_name in level_data:
        sub = df_xgb_tune[df_xgb_tune['level'] == level_name]
        best = sub.loc[sub['val_mae'].idxmin()]
        print(f"  {level_name}: val_MAE={best['val_mae']:.3f} test_MAE={best['test_mae']:.3f} "
              f"depth={int(best['max_depth'])} lr={best['learning_rate']}")
        all_results.append({
            'experiment': f'{level_name}_xgb_tuned', 'model': 'XGBoost_tuned',
            'mae': best['test_mae'], 'mse': best['test_mse'], 'r2': best['test_r2'],
            'val_mae': best['val_mae'], 'train_time': best['time'],
        })

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 5: NN HYPERPARAMETER SWEEP (all levels)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 5: NEURAL NETWORK SWEEP")
    print("=" * 70)

    nn_configs = [
        {'lr': 1e-3, 'dropout': 0.3, 'batch_size': 64, 'weight_decay': 1e-4, 'hidden_dims': (256, 128, 64)},
        {'lr': 5e-4, 'dropout': 0.2, 'batch_size': 128, 'weight_decay': 1e-4, 'hidden_dims': (256, 128, 64)},
        {'lr': 1e-3, 'dropout': 0.4, 'batch_size': 64, 'weight_decay': 1e-3, 'hidden_dims': (128, 64, 32)},
        {'lr': 5e-4, 'dropout': 0.3, 'batch_size': 64, 'weight_decay': 1e-4, 'hidden_dims': (512, 256, 128)},
        {'lr': 1e-3, 'dropout': 0.2, 'batch_size': 32, 'weight_decay': 1e-5, 'hidden_dims': (256, 128, 64)},
        {'lr': 3e-4, 'dropout': 0.3, 'batch_size': 64, 'weight_decay': 1e-3, 'hidden_dims': (128, 64)},
    ]

    nn_tuning_results = []
    for level_name, ld in level_data.items():
        print(f"\n── NN tuning: {level_name} ({ld['X_tr'].shape[1]} features) ──")
        for i, cfg in enumerate(nn_configs):
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            t0 = time.time()
            nn_m, best_val = train_nn(
                ld['X_tr'], ld['y_tr'], ld['X_v'], ld['y_v'],
                input_dim=ld['X_tr'].shape[1], patience=20, **cfg,
            )
            y_pred_t = predict_nn(nn_m, ld['X_te'])
            m_t = eval_metrics(ld['y_te'], y_pred_t)
            elapsed = time.time() - t0
            row = {
                'level': level_name, 'config_idx': i,
                **{f'nn_{k}': str(v) for k, v in cfg.items()},
                'val_mse': best_val,
                'test_mae': m_t['mae'], 'test_mse': m_t['mse'], 'test_r2': m_t['r2'],
                'time': elapsed,
            }
            nn_tuning_results.append(row)
            hd = cfg['hidden_dims']
            print(f"  Config {i}: test_MAE={m_t['mae']:.3f} val_MSE={best_val:.2f} "
                  f"lr={cfg['lr']} drop={cfg['dropout']} hidden={hd} ({elapsed:.1f}s)")

    df_nn_tune = pd.DataFrame(nn_tuning_results)

    # Best NN per level
    print("\n── Best NN configs (by val MSE) ──")
    for level_name in level_data:
        sub = df_nn_tune[df_nn_tune['level'] == level_name]
        best = sub.loc[sub['val_mse'].idxmin()]
        print(f"  {level_name}: val_MSE={best['val_mse']:.2f} test_MAE={best['test_mae']:.3f}")
        all_results.append({
            'experiment': f'{level_name}_nn_tuned', 'model': 'NeuralNet_tuned',
            'mae': best['test_mae'], 'mse': best['test_mse'], 'r2': best['test_r2'],
            'val_mse': best['val_mse'], 'train_time': best['time'],
        })

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 6: RIDGE WITH WIDER ALPHA RANGE
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 6: RIDGE WIDER ALPHA SEARCH")
    print("=" * 70)

    wide_alphas = [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000]
    for level_name, ld in level_data.items():
        ridge = RidgeCV(alphas=wide_alphas)
        ridge.fit(ld['X_tr'], ld['y_tr'])
        y_pred = ridge.predict(ld['X_te'])
        m = eval_metrics(ld['y_te'], y_pred)
        print(f"  {level_name}: MAE={m['mae']:.3f} MSE={m['mse']:.2f} R²={m['r2']:.3f} α={ridge.alpha_}")
        all_results.append({
            'experiment': f'{level_name}_ridge_wide', 'model': 'Ridge_wide',
            'mae': m['mae'], 'mse': m['mse'], 'r2': m['r2'], 'alpha': ridge.alpha_,
        })

    # ══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FULL RESULTS SUMMARY")
    print("=" * 70)

    df_all = pd.DataFrame(all_results)
    print(df_all[['experiment', 'model', 'mae', 'mse', 'r2']].to_string(index=False))

    # Save
    df_all.to_csv('results/tuning_results.csv', index=False)
    df_xgb_tune.to_csv('results/xgb_grid_search.csv', index=False)
    df_nn_tune.to_csv('results/nn_sweep.csv', index=False)
    print("\nSaved: results/tuning_results.csv, results/xgb_grid_search.csv, results/nn_sweep.csv")
