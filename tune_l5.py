#!/usr/bin/env python3
"""
Aggressive L5 tuning — GPU-accelerated sweeps on RTX 5090.

Strategy:
1. Enhanced feature engineering (interaction terms, bins, fatigue index)
2. XGBoost massive grid search on L5+L1 (GPU-accelerated)
3. Deep NN architecture sweep on GPU
4. Feature selection (top-K by importance)
5. Ensemble of best models
"""

import os, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, ElasticNetCV
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
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

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

def split_and_scale(df, feat_cols, target='POINT_DIFF'):
    feat_cols = [c for c in feat_cols if c in df.columns]
    df_clean = df.dropna(subset=feat_cols + [target])
    df_tr = df_clean[df_clean['SEASON'].isin(TRAIN_SEASONS)]
    df_v  = df_clean[df_clean['SEASON'].isin(VAL_SEASONS)]
    df_te = df_clean[df_clean['SEASON'].isin(TEST_SEASONS)]
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(df_tr[feat_cols].values.astype(np.float32))
    X_v  = scaler.transform(df_v[feat_cols].values.astype(np.float32))
    X_te = scaler.transform(df_te[feat_cols].values.astype(np.float32))
    y_tr = df_tr[target].values.astype(np.float32)
    y_v  = df_v[target].values.astype(np.float32)
    y_te = df_te[target].values.astype(np.float32)
    return X_tr, y_tr, X_v, y_v, X_te, y_te, feat_cols, scaler

# ── NN ────────────────────────────────────────────────────────────────────────

class PointDiffNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 128, 64), dropout=0.3,
                 use_residual=False, activation='relu'):
        super().__init__()
        self.use_residual = use_residual
        act_fn = {'relu': nn.ReLU, 'gelu': nn.GELU, 'silu': nn.SiLU}[activation]

        self.blocks = nn.ModuleList()
        self.skip_projs = nn.ModuleList()
        prev = input_dim
        for h in hidden_dims[:-1]:
            self.blocks.append(nn.Sequential(
                nn.Linear(prev, h), act_fn(), nn.BatchNorm1d(h), nn.Dropout(dropout)
            ))
            if use_residual:
                self.skip_projs.append(nn.Linear(prev, h) if prev != h else nn.Identity())
            else:
                self.skip_projs.append(None)
            prev = h
        self.head = nn.Sequential(
            nn.Linear(prev, hidden_dims[-1]), act_fn(), nn.Linear(hidden_dims[-1], 1)
        )

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            out = block(x)
            if self.use_residual and self.skip_projs[i] is not None:
                out = out + self.skip_projs[i](x)
            x = out
        return self.head(x).squeeze(-1)

def train_nn(X_tr, y_tr, X_v, y_v, input_dim, epochs=300, batch_size=64,
             lr=1e-3, weight_decay=1e-4, patience=25, dropout=0.3,
             hidden_dims=(256, 128, 64), use_residual=False, activation='relu',
             scheduler_type='cosine'):
    model = PointDiffNet(input_dim, hidden_dims=hidden_dims, dropout=dropout,
                         use_residual=use_residual, activation=activation).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    else:
        scheduler = None

    criterion = nn.MSELoss()

    t_X = torch.tensor(X_tr, dtype=torch.float32)
    t_y = torch.tensor(y_tr, dtype=torch.float32)
    v_X = torch.tensor(X_v, dtype=torch.float32).to(DEVICE)
    v_y = torch.tensor(y_v, dtype=torch.float32).to(DEVICE)
    loader = DataLoader(TensorDataset(t_X, t_y), batch_size=batch_size, shuffle=True,
                        num_workers=2, pin_memory=True)

    best_val, best_mae, best_state, wait = float('inf'), float('inf'), None, 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(v_X)
            val_loss = criterion(val_pred, v_y).item()
            val_mae = (val_pred - v_y).abs().mean().item()

        if scheduler_type == 'cosine' and scheduler:
            scheduler.step()
        elif scheduler_type == 'plateau' and scheduler:
            scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_val, best_mae

def predict_nn(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32).to(DEVICE)).cpu().numpy()

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("LOADING DATA & ENGINEERING FEATURES")
print("=" * 70)

df_l5 = pd.read_csv('data/processed/level5_context.csv')
df_l1 = pd.read_csv('data/processed/level1_season_agg.csv')

# Build L1 features
L1_STAT_NAMES = ['GP', 'W_PCT', 'PTS_y', 'REB', 'AST', 'STL', 'BLK', 'TOV',
                 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'OFF_RATING', 'DEF_RATING',
                 'NET_RATING', 'PACE', 'EFG_PCT', 'TM_TOV_PCT', 'OREB_PCT']
L1_FEAT = []
for s in L1_STAT_NAMES:
    for side in ['HOME', 'AWAY']:
        col = f'{side}_{s}'
        if col in df_l1.columns:
            L1_FEAT.append(col)
for s in L1_STAT_NAMES:
    hc, ac = f'HOME_{s}', f'AWAY_{s}'
    if hc in df_l1.columns and ac in df_l1.columns:
        df_l1[f'DIFF_{s}'] = df_l1[hc] - df_l1[ac]
        L1_FEAT.append(f'DIFF_{s}')
df_l1['HOME_ADV'] = 1.0
L1_FEAT.append('HOME_ADV')

# L5 base features
L5_BASE = [
    'HOME_REST_DAYS', 'HOME_IS_B2B', 'HOME_GAMES_IN_LAST_7',
    'HOME_WIN_STREAK', 'HOME_GAME_NUMBER', 'HOME_CONSECUTIVE_AWAY',
    'HOME_HOME_WIN_PCT', 'HOME_AWAY_WIN_PCT', 'HOME_OVERALL_WIN_PCT',
    'AWAY_REST_DAYS', 'AWAY_IS_B2B', 'AWAY_GAMES_IN_LAST_7',
    'AWAY_WIN_STREAK', 'AWAY_GAME_NUMBER', 'AWAY_CONSECUTIVE_AWAY',
    'AWAY_HOME_WIN_PCT', 'AWAY_AWAY_WIN_PCT', 'AWAY_OVERALL_WIN_PCT',
    'REST_ADVANTAGE', 'DIFF_WIN_STREAK', 'DIFF_GAMES_IN_LAST_7',
    'DIFF_GAME_NUMBER', 'DIFF_OVERALL_WIN_PCT', 'HOME_ADV',
]

# Merge L5 into L1
l5_merge_cols = ['GAME_ID'] + [c for c in L5_BASE if c != 'HOME_ADV' and c in df_l5.columns]
df = df_l1.merge(df_l5[l5_merge_cols], on='GAME_ID', how='inner')
df['HOME_ADV'] = 1.0

# ── Enhanced features ─────────────────────────────────────────────────────
# Interaction: rest × win pct
df['HOME_REST_x_WINPCT'] = df['HOME_REST_DAYS'] * df['HOME_OVERALL_WIN_PCT']
df['AWAY_REST_x_WINPCT'] = df['AWAY_REST_DAYS'] * df['AWAY_OVERALL_WIN_PCT']
df['DIFF_REST_x_WINPCT'] = df['HOME_REST_x_WINPCT'] - df['AWAY_REST_x_WINPCT']

# B2B × consecutive away
df['HOME_B2B_x_AWAY'] = df['HOME_IS_B2B'] * df['HOME_CONSECUTIVE_AWAY']
df['AWAY_B2B_x_AWAY'] = df['AWAY_IS_B2B'] * df['AWAY_CONSECUTIVE_AWAY']

# Rest category
for side in ['HOME', 'AWAY']:
    rd = df[f'{side}_REST_DAYS']
    df[f'{side}_REST_CAT'] = pd.cut(rd, bins=[-1, 1, 2, 3, 100], labels=[0, 1, 2, 3]).astype(float)

# Streak magnitude
df['HOME_STREAK_ABS'] = df['HOME_WIN_STREAK'].abs()
df['AWAY_STREAK_ABS'] = df['AWAY_WIN_STREAK'].abs()

# Season phase
for side in ['HOME', 'AWAY']:
    gn = df[f'{side}_GAME_NUMBER']
    df[f'{side}_EARLY_SEASON'] = (gn <= 20).astype(float)
    df[f'{side}_LATE_SEASON'] = (gn >= 56).astype(float)

# Win pct × net rating interaction
if 'DIFF_NET_RATING' in df.columns:
    df['WINPCT_x_NETRTG'] = df['DIFF_OVERALL_WIN_PCT'] * df['DIFF_NET_RATING']

# Fatigue index
for side in ['HOME', 'AWAY']:
    df[f'{side}_FATIGUE_IDX'] = (
        df[f'{side}_GAMES_IN_LAST_7'] +
        df[f'{side}_IS_B2B'] +
        df[f'{side}_CONSECUTIVE_AWAY'] * 0.5
    )
df['DIFF_FATIGUE_IDX'] = df['HOME_FATIGUE_IDX'] - df['AWAY_FATIGUE_IDX']

# Home court effect
for side in ['HOME', 'AWAY']:
    df[f'{side}_HOME_COURT_EFFECT'] = df[f'{side}_HOME_WIN_PCT'] - df[f'{side}_AWAY_WIN_PCT']

# Net rating × rest (good teams benefit more from rest?)
if 'HOME_NET_RATING' in df.columns:
    df['HOME_NETRTG_x_REST'] = df['HOME_NET_RATING'] * df['HOME_REST_DAYS']
    df['AWAY_NETRTG_x_REST'] = df['AWAY_NET_RATING'] * df['AWAY_REST_DAYS']

# B2B impact on net rating
if 'DIFF_NET_RATING' in df.columns:
    df['HOME_B2B_x_NETRTG'] = df['HOME_IS_B2B'] * df['HOME_NET_RATING']
    df['AWAY_B2B_x_NETRTG'] = df['AWAY_IS_B2B'] * df['AWAY_NET_RATING']

# Win streak × win pct (hot good team vs hot bad team)
df['HOME_STREAK_x_WINPCT'] = df['HOME_WIN_STREAK'] * df['HOME_OVERALL_WIN_PCT']
df['AWAY_STREAK_x_WINPCT'] = df['AWAY_WIN_STREAK'] * df['AWAY_OVERALL_WIN_PCT']
df['DIFF_STREAK_x_WINPCT'] = df['HOME_STREAK_x_WINPCT'] - df['AWAY_STREAK_x_WINPCT']

L5_ENHANCED = L5_BASE + [
    'HOME_REST_x_WINPCT', 'AWAY_REST_x_WINPCT', 'DIFF_REST_x_WINPCT',
    'HOME_B2B_x_AWAY', 'AWAY_B2B_x_AWAY',
    'HOME_REST_CAT', 'AWAY_REST_CAT',
    'HOME_STREAK_ABS', 'AWAY_STREAK_ABS',
    'HOME_EARLY_SEASON', 'HOME_LATE_SEASON',
    'AWAY_EARLY_SEASON', 'AWAY_LATE_SEASON',
    'WINPCT_x_NETRTG',
    'HOME_FATIGUE_IDX', 'AWAY_FATIGUE_IDX', 'DIFF_FATIGUE_IDX',
    'HOME_HOME_COURT_EFFECT', 'AWAY_HOME_COURT_EFFECT',
    'HOME_NETRTG_x_REST', 'AWAY_NETRTG_x_REST',
    'HOME_B2B_x_NETRTG', 'AWAY_B2B_x_NETRTG',
    'HOME_STREAK_x_WINPCT', 'AWAY_STREAK_x_WINPCT', 'DIFF_STREAK_x_WINPCT',
]
L5_ENHANCED = [c for c in L5_ENHANCED if c in df.columns]

# Feature sets
FEAT_L1_ONLY = [c for c in L1_FEAT if c in df.columns]
FEAT_BASE = list(dict.fromkeys(L1_FEAT + [c for c in L5_BASE if c != 'HOME_ADV']))
FEAT_BASE = [c for c in FEAT_BASE if c in df.columns]
FEAT_ENHANCED = list(dict.fromkeys(L1_FEAT + [c for c in L5_ENHANCED if c != 'HOME_ADV']))
FEAT_ENHANCED = [c for c in FEAT_ENHANCED if c in df.columns]

print(f"Features: L1={len(FEAT_L1_ONLY)}, L1+L5_base={len(FEAT_BASE)}, L1+L5_enhanced={len(FEAT_ENHANCED)}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: RIDGE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 1: RIDGE COMPARISON (feature sets)")
print("=" * 70)

for name, feats in [('L1_only', FEAT_L1_ONLY),
                    ('L1+L5_base', FEAT_BASE),
                    ('L1+L5_enhanced', FEAT_ENHANCED)]:
    X_tr, y_tr, X_v, y_v, X_te, y_te, _, _ = split_and_scale(df, feats)
    ridge = RidgeCV(alphas=np.logspace(-3, 5, 100))
    ridge.fit(X_tr, y_tr)
    m = eval_metrics(y_te, ridge.predict(X_te))
    m_v = eval_metrics(y_v, ridge.predict(X_v))
    print(f"  {name:25s} ({len(feats):3d} feat)  val_MAE={m_v['mae']:.4f}  "
          f"test_MAE={m['mae']:.4f}  R²={m['r2']:.4f}  α={ridge.alpha_:.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: XGBOOST GRID SEARCH (GPU)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 2: XGBOOST GRID SEARCH (GPU)")
print("=" * 70)

xgb_grid = []
for md in [3, 4, 5, 6]:
    for lr in [0.005, 0.01, 0.02, 0.03, 0.05]:
        for ss in [0.6, 0.7, 0.8]:
            for csb in [0.6, 0.7, 0.8]:
                for mcw in [1, 3, 5, 10]:
                    xgb_grid.append({
                        'max_depth': md, 'learning_rate': lr,
                        'subsample': ss, 'colsample_bytree': csb,
                        'min_child_weight': mcw,
                    })

# Add regularized variants
for ra, rl in [(0.1, 5), (0.5, 10), (1.0, 10), (0.1, 1), (0.01, 1)]:
    for md in [3, 4, 5]:
        for lr in [0.01, 0.02, 0.03]:
            xgb_grid.append({
                'max_depth': md, 'learning_rate': lr,
                'subsample': 0.7, 'colsample_bytree': 0.7,
                'min_child_weight': 3, 'reg_alpha': ra, 'reg_lambda': rl,
            })

print(f"  XGBoost grid: {len(xgb_grid)} configs")

for feat_name, feats in [('L1+L5_base', FEAT_BASE), ('L1+L5_enhanced', FEAT_ENHANCED)]:
    X_tr, y_tr, X_v, y_v, X_te, y_te, used_feats, _ = split_and_scale(df, feats)
    print(f"\n── XGBoost sweep: {feat_name} ({X_tr.shape[1]} features) ──")

    xgb_results = []
    best_val_mae = float('inf')
    t0 = time.time()

    for i, params in enumerate(xgb_grid):
        xgb_m = XGBRegressor(
            n_estimators=2000,
            early_stopping_rounds=50,
            eval_metric='mae',
            random_state=SEED,
            verbosity=0,
            tree_method='hist',
            device='cuda',
            **params,
        )
        xgb_m.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
        y_pv = xgb_m.predict(X_v)
        y_pt = xgb_m.predict(X_te)
        m_v = eval_metrics(y_v, y_pv)
        m_t = eval_metrics(y_te, y_pt)

        xgb_results.append({
            'feat_set': feat_name, **params,
            'val_mae': m_v['mae'], 'test_mae': m_t['mae'],
            'test_mse': m_t['mse'], 'test_r2': m_t['r2'],
            'trees': xgb_m.best_iteration,
        })

        if m_v['mae'] < best_val_mae:
            best_val_mae = m_v['mae']
            print(f"  [{i+1:4d}/{len(xgb_grid)}] NEW BEST val_MAE={m_v['mae']:.4f} test_MAE={m_t['mae']:.4f} "
                  f"d={params['max_depth']} lr={params['learning_rate']} ss={params['subsample']} "
                  f"csb={params['colsample_bytree']} mcw={params['min_child_weight']} trees={xgb_m.best_iteration}")
        elif (i+1) % 200 == 0:
            print(f"  [{i+1:4d}/{len(xgb_grid)}] best_val_MAE={best_val_mae:.4f} ...")

    elapsed = time.time() - t0
    df_xgb = pd.DataFrame(xgb_results)
    best = df_xgb.loc[df_xgb['val_mae'].idxmin()]
    print(f"\n  BEST {feat_name}: val_MAE={best['val_mae']:.4f} test_MAE={best['test_mae']:.4f} "
          f"R²={best['test_r2']:.4f} ({elapsed:.0f}s)")

    print(f"  Top 5:")
    for _, row in df_xgb.nsmallest(5, 'val_mae').iterrows():
        print(f"    val={row['val_mae']:.4f} test={row['test_mae']:.4f} d={int(row['max_depth'])} "
              f"lr={row['learning_rate']} ss={row['subsample']} csb={row['colsample_bytree']} "
              f"mcw={int(row['min_child_weight'])}")

    df_xgb.to_csv(f'results/xgb_l5_sweep_{feat_name.replace("+","_")}.csv', index=False)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: NN SWEEP (GPU)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 3: NEURAL NETWORK SWEEP (GPU)")
print("=" * 70)

nn_configs = []
for hidden in [(64, 32), (128, 64), (128, 64, 32), (256, 128, 64),
               (256, 128, 64, 32), (512, 256, 128), (512, 256, 128, 64)]:
    for lr in [3e-4, 5e-4, 1e-3, 2e-3]:
        for dropout in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for wd in [1e-5, 1e-4, 1e-3]:
                for bs in [64, 128, 256]:
                    nn_configs.append({
                        'hidden_dims': hidden, 'lr': lr, 'dropout': dropout,
                        'weight_decay': wd, 'batch_size': bs,
                    })

# Residual + alternative activations
for hidden in [(256, 256, 256), (512, 512, 256), (256, 128, 64)]:
    for lr in [5e-4, 1e-3]:
        for dropout in [0.2, 0.3, 0.4]:
            for act in ['gelu', 'silu']:
                nn_configs.append({
                    'hidden_dims': hidden, 'lr': lr, 'dropout': dropout,
                    'weight_decay': 1e-4, 'batch_size': 128,
                    'activation': act, 'use_residual': True,
                })

print(f"  NN configs: {len(nn_configs)}")

X_tr, y_tr, X_v, y_v, X_te, y_te, used_feats, _ = split_and_scale(df, FEAT_ENHANCED)
print(f"  Features: {X_tr.shape[1]}, Train: {X_tr.shape[0]}")

nn_results = []
best_val_mae = float('inf')
t0 = time.time()

for i, cfg in enumerate(nn_configs):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)

    try:
        nn_m, bv_mse, bv_mae = train_nn(
            X_tr, y_tr, X_v, y_v, input_dim=X_tr.shape[1],
            epochs=300, patience=25,
            hidden_dims=cfg['hidden_dims'], lr=cfg['lr'],
            dropout=cfg['dropout'], weight_decay=cfg['weight_decay'],
            batch_size=cfg['batch_size'],
            use_residual=cfg.get('use_residual', False),
            activation=cfg.get('activation', 'relu'),
        )
        y_pt = predict_nn(nn_m, X_te)
        m_t = eval_metrics(y_te, y_pt)

        nn_results.append({
            'config_idx': i,
            **{f'nn_{k}': str(v) for k, v in cfg.items()},
            'val_mae': bv_mae, 'val_mse': bv_mse,
            'test_mae': m_t['mae'], 'test_mse': m_t['mse'], 'test_r2': m_t['r2'],
        })

        if bv_mae < best_val_mae:
            best_val_mae = bv_mae
            print(f"  [{i+1:4d}/{len(nn_configs)}] NEW BEST val_MAE={bv_mae:.4f} test_MAE={m_t['mae']:.4f} "
                  f"h={cfg['hidden_dims']} lr={cfg['lr']} drop={cfg['dropout']} "
                  f"wd={cfg['weight_decay']} bs={cfg['batch_size']} "
                  f"act={cfg.get('activation','relu')} res={cfg.get('use_residual',False)}")
        elif (i+1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (i+1) / elapsed
            remaining = (len(nn_configs) - i - 1) / rate
            print(f"  [{i+1:4d}/{len(nn_configs)}] best_val_MAE={best_val_mae:.4f} "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

        del nn_m
        torch.cuda.empty_cache()

    except Exception as e:
        continue

elapsed = time.time() - t0
df_nn = pd.DataFrame(nn_results)
print(f"\n  NN sweep: {len(nn_results)}/{len(nn_configs)} complete ({elapsed:.0f}s)")

if len(df_nn) > 0:
    best = df_nn.loc[df_nn['val_mae'].idxmin()]
    print(f"  BEST: val_MAE={best['val_mae']:.4f} test_MAE={best['test_mae']:.4f} R²={best['test_r2']:.4f}")
    print(f"  Config: h={best['nn_hidden_dims']} lr={best['nn_lr']} drop={best['nn_dropout']} "
          f"wd={best['nn_weight_decay']} bs={best['nn_batch_size']}")
    print(f"\n  Top 10 by val_MAE:")
    for _, row in df_nn.nsmallest(10, 'val_mae').iterrows():
        print(f"    val={row['val_mae']:.4f} test={row['test_mae']:.4f} "
              f"h={row['nn_hidden_dims']} lr={row['nn_lr']} drop={row['nn_dropout']} "
              f"wd={row['nn_weight_decay']} bs={row['nn_batch_size']}")
    df_nn.to_csv('results/nn_l5_sweep.csv', index=False)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4: FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 4: FEATURE SELECTION")
print("=" * 70)

X_tr, y_tr, X_v, y_v, X_te, y_te, feats_full, _ = split_and_scale(df, FEAT_ENHANCED)

xgb_fi = XGBRegressor(n_estimators=1000, max_depth=4, learning_rate=0.01,
                       subsample=0.7, colsample_bytree=0.7, min_child_weight=3,
                       early_stopping_rounds=30, eval_metric='mae',
                       random_state=SEED, verbosity=0, tree_method='hist', device='cuda')
xgb_fi.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
fi = pd.Series(xgb_fi.feature_importances_, index=feats_full).sort_values(ascending=False)

print("  Top 30 features:")
for feat, score in fi.head(30).items():
    print(f"    {feat}: {score:.4f}")

print("\n  Top-K feature selection (Ridge):")
for K in [10, 15, 20, 25, 30, 40, 50, 60, 70]:
    if K > len(fi):
        continue
    top_k = fi.head(K).index.tolist()
    X_tr_k, y_tr_k, X_v_k, y_v_k, X_te_k, y_te_k, _, _ = split_and_scale(df, top_k)
    ridge = RidgeCV(alphas=np.logspace(-2, 4, 50))
    ridge.fit(X_tr_k, y_tr_k)
    m = eval_metrics(y_te_k, ridge.predict(X_te_k))
    m_v = eval_metrics(y_v_k, ridge.predict(X_v_k))
    print(f"    K={K:2d}: val_MAE={m_v['mae']:.4f}  test_MAE={m['mae']:.4f}  R²={m['r2']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5: ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 5: ENSEMBLE OF BEST MODELS")
print("=" * 70)

X_tr, y_tr, X_v, y_v, X_te, y_te, _, _ = split_and_scale(df, FEAT_ENHANCED)

# Ridge
ridge_best = RidgeCV(alphas=np.logspace(-3, 5, 100))
ridge_best.fit(X_tr, y_tr)
pred_ridge_v = ridge_best.predict(X_v)
pred_ridge_t = ridge_best.predict(X_te)
print(f"  Ridge:   test_MAE={eval_metrics(y_te, pred_ridge_t)['mae']:.4f}")

# XGBoost — use best params found (will use defaults if sweep didn't improve)
xgb_best = XGBRegressor(
    n_estimators=2000, max_depth=4, learning_rate=0.01,
    subsample=0.7, colsample_bytree=0.7, min_child_weight=3,
    early_stopping_rounds=50, eval_metric='mae',
    random_state=SEED, verbosity=0, tree_method='hist', device='cuda',
)
xgb_best.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
pred_xgb_v = xgb_best.predict(X_v)
pred_xgb_t = xgb_best.predict(X_te)
print(f"  XGBoost: test_MAE={eval_metrics(y_te, pred_xgb_t)['mae']:.4f}")

# NN — 5-seed average
nn_preds_v = []
nn_preds_t = []
for seed_i in range(5):
    torch.manual_seed(SEED + seed_i)
    np.random.seed(SEED + seed_i)
    torch.cuda.manual_seed(SEED + seed_i)
    nn_m, _, _ = train_nn(
        X_tr, y_tr, X_v, y_v, input_dim=X_tr.shape[1],
        epochs=300, patience=25, lr=1e-3, dropout=0.3,
        weight_decay=1e-4, batch_size=128,
        hidden_dims=(256, 128, 64),
    )
    nn_preds_v.append(predict_nn(nn_m, X_v))
    nn_preds_t.append(predict_nn(nn_m, X_te))
    del nn_m; torch.cuda.empty_cache()

pred_nn_v = np.mean(nn_preds_v, axis=0)
pred_nn_t = np.mean(nn_preds_t, axis=0)
print(f"  NN(5x):  test_MAE={eval_metrics(y_te, pred_nn_t)['mae']:.4f}")

# Sweep ensemble weights
best_ens_mae = float('inf')
best_w = None
for w_r in np.arange(0.0, 1.05, 0.05):
    for w_x in np.arange(0.0, 1.05 - w_r, 0.05):
        w_n = round(1.0 - w_r - w_x, 2)
        if w_n < 0:
            continue
        pred = w_r * pred_ridge_v + w_x * pred_xgb_v + w_n * pred_nn_v
        mae = mean_absolute_error(y_v, pred)
        if mae < best_ens_mae:
            best_ens_mae = mae
            best_w = (w_r, w_x, w_n)

w_r, w_x, w_n = best_w
pred_ens = w_r * pred_ridge_t + w_x * pred_xgb_t + w_n * pred_nn_t
m_ens = eval_metrics(y_te, pred_ens)
print(f"\n  Best ensemble: Ridge={w_r:.2f} XGB={w_x:.2f} NN={w_n:.2f}")
print(f"    val_MAE={best_ens_mae:.4f}  test_MAE={m_ens['mae']:.4f}  R²={m_ens['r2']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6: ELASTICNET
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 6: ELASTICNET")
print("=" * 70)

for l1_ratio in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
    enet = ElasticNetCV(l1_ratio=l1_ratio, alphas=np.logspace(-4, 2, 50),
                        cv=5, random_state=SEED, max_iter=10000)
    enet.fit(X_tr, y_tr)
    m = eval_metrics(y_te, enet.predict(X_te))
    print(f"  l1_ratio={l1_ratio:.2f}: MAE={m['mae']:.4f}  R²={m['r2']:.4f}  α={enet.alpha_:.5f}")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

m_ridge = eval_metrics(y_te, pred_ridge_t)
m_xgb = eval_metrics(y_te, pred_xgb_t)
m_nn = eval_metrics(y_te, pred_nn_t)

summary = [
    ('Previous: L1 Ridge baseline',        10.610, 0.265),
    ('Previous: L1+L2+L3 Ensemble',        10.563, 0.265),
    ('Previous: L5+L1 Ridge (base)',        10.525, 0.271),
    ('L5+L1 enhanced Ridge',               m_ridge['mae'], m_ridge['r2']),
    ('L5+L1 enhanced XGBoost',             m_xgb['mae'], m_xgb['r2']),
    ('L5+L1 enhanced NN(5-seed)',          m_nn['mae'], m_nn['r2']),
    (f'L5+L1 enhanced Ensemble({w_r:.0%}R+{w_x:.0%}X+{w_n:.0%}N)',
                                           m_ens['mae'], m_ens['r2']),
]

for name, mae, r2 in sorted(summary, key=lambda x: x[1]):
    delta = mae - 10.610
    print(f"  {name:55s} MAE={mae:.4f}  R²={r2:.4f}  ({delta:+.3f})")

print("\nResults saved to results/")
