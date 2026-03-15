#!/usr/bin/env python3
"""
Level 6: Injury/Lineup Features — Roster Availability Signal

For each game, detects which of a team's core players are missing (injured/resting)
and quantifies the impact. Uses only pre-game information (season stats computed
from games *before* the current one).

Features built:
 - Core roster availability (top-8 by minutes, computed from prior games only)
 - Weighted minutes/points lost from missing core players
 - Star player absence flags (top-1, top-2, top-3 missing)
 - Rolling roster stability (how many different players used in last N games)
 - Cumulative missed games by core players
 - Roster strength differential between teams

Then trains Ridge/XGBoost/NN on:
 1. L6 standalone
 2. L6 + L1 (season stats + lineup)
 3. L6 + L1 + L5 (season stats + schedule + lineup) — the full combined model
 4. Comprehensive hyperparameter tuning + ensemble
 5. Comparison graphs
"""

import os, time, warnings, itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
from xgboost import XGBRegressor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

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
             scheduler_type='cosine', verbose=False):
    model = PointDiffNet(input_dim, hidden_dims=hidden_dims, dropout=dropout,
                         use_residual=use_residual, activation=activation).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
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
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        model.eval()
        with torch.no_grad():
            val_pred = model(v_X)
            val_loss = criterion(val_pred, v_y).item()
            val_mae = (val_pred - v_y).abs().mean().item()
        train_losses.append(epoch_loss / n_batches)
        val_losses.append(val_loss)
        if scheduler:
            scheduler.step()
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
    return model, best_val, best_mae, train_losses, val_losses

def predict_nn(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32).to(DEVICE)).cpu().numpy()

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 0: BUILD LINEUP/INJURY FEATURES
# ══════════════════════════════════════════════════════════════════════════════

LINEUP_CSV = 'data/processed/level6_lineup.csv'

def build_lineup_features():
    """Build roster availability features from player gamelogs.

    Key insight: we use only data BEFORE each game (no leakage).
    For each team-season, we track a running 'core roster' based on minutes
    played in all prior games that season. Then for each game, we check which
    core players are absent and quantify the impact.
    """
    print("\n" + "=" * 70)
    print("BUILDING LINEUP/INJURY FEATURES FROM PLAYER GAMELOGS")
    print("=" * 70)

    # Load all player gamelogs
    all_players = []
    for season in ALL_SEASONS:
        path = f'data/raw/player_gamelogs_{season}.csv'
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['SEASON'] = season
            all_players.append(df)
            print(f"  Loaded {path}: {len(df)} rows")
    players = pd.concat(all_players, ignore_index=True)
    players['MIN'] = pd.to_numeric(players['MIN'], errors='coerce').fillna(0)
    players['PTS'] = pd.to_numeric(players['PTS'], errors='coerce').fillna(0)
    players['PLUS_MINUS'] = pd.to_numeric(players['PLUS_MINUS'], errors='coerce').fillna(0)
    players['GAME_DATE'] = pd.to_datetime(players['GAME_DATE'])

    # Load game schedule for GAME_ID → home/away mapping
    all_games = []
    for season in ALL_SEASONS:
        path = f'data/raw/games_{season}.csv'
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['SEASON'] = season
            all_games.append(df)
    games = pd.concat(all_games, ignore_index=True)
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])

    # Identify home/away for each game
    games['IS_HOME'] = games['MATCHUP'].str.contains('vs.').astype(int)
    home_games = games[games['IS_HOME'] == 1][['GAME_ID', 'TEAM_ID', 'SEASON', 'PTS', 'PLUS_MINUS']].copy()
    away_games = games[games['IS_HOME'] == 0][['GAME_ID', 'TEAM_ID', 'SEASON', 'PTS', 'PLUS_MINUS']].copy()
    home_games.columns = ['GAME_ID', 'HOME_TEAM_ID', 'SEASON', 'HOME_PTS', 'HOME_PLUS_MINUS']
    away_games.columns = ['GAME_ID', 'AWAY_TEAM_ID', 'SEASON', 'AWAY_PTS', 'AWAY_PLUS_MINUS']
    game_pairs = home_games.merge(away_games, on=['GAME_ID', 'SEASON'], how='inner')
    game_pairs['POINT_DIFF'] = game_pairs['HOME_PTS'] - game_pairs['AWAY_PTS']

    # Add game dates
    game_dates = games.groupby('GAME_ID')['GAME_DATE'].first().reset_index()
    game_pairs = game_pairs.merge(game_dates, on='GAME_ID')

    # Sort games chronologically
    game_pairs = game_pairs.sort_values('GAME_DATE').reset_index(drop=True)
    players = players.sort_values(['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)

    print(f"\n  Total game pairs: {len(game_pairs)}")
    print(f"  Total player-game rows: {len(players)}")

    # ── Compute per-game lineup features ──────────────────────────────────────
    # For each team-game, we need:
    # 1. Who played in this game (from player gamelogs)
    # 2. Who are the 'core' players (top-8 by cumulative minutes BEFORE this game)
    # 3. How many core players are missing, and their importance

    print("\n  Computing per-team-game features (this may take a minute)...")

    # Pre-compute: for each team-season, get sorted list of games and
    # cumulative player stats up to each game
    team_game_features = {}  # (GAME_ID, TEAM_ID) -> feature dict

    # Group players by team and season
    for (team_id, season), team_season_players in players.groupby(['TEAM_ID', 'SEASON']):
        # Get all games this team played this season, in order
        team_games_mask = (games['TEAM_ID'] == team_id) & (games['SEASON'] == season)
        team_game_list = games[team_games_mask].sort_values('GAME_DATE')
        game_ids_ordered = team_game_list['GAME_ID'].tolist()
        game_dates_ordered = team_game_list['GAME_DATE'].tolist()

        if len(game_ids_ordered) == 0:
            continue

        # Build cumulative stats per player, game by game
        # Track: total_min, total_pts, total_pm, games_played per player
        cum_stats = {}  # player_id -> {min, pts, pm, gp}
        prev_game_players = set()  # players who played in most recent game
        roster_last_5 = []  # list of sets of player_ids for last 5 games
        roster_last_10 = []

        for game_idx, game_id in enumerate(game_ids_ordered):
            # Players who played in THIS game
            game_players = team_season_players[team_season_players['GAME_ID'] == game_id]
            current_player_ids = set(game_players['PLAYER_ID'].values)

            # ── Features based on PRE-GAME state (cum_stats from before this game) ──
            features = {}

            if game_idx >= 5:  # Need some history
                # Core roster: top-8 by cumulative minutes BEFORE this game
                if len(cum_stats) >= 3:
                    sorted_players = sorted(cum_stats.items(),
                                            key=lambda x: x[1]['min'], reverse=True)
                    top8 = sorted_players[:min(8, len(sorted_players))]
                    top3 = sorted_players[:min(3, len(sorted_players))]
                    top5 = sorted_players[:min(5, len(sorted_players))]

                    top8_ids = {p[0] for p in top8}
                    top3_ids = {p[0] for p in top3}
                    top5_ids = {p[0] for p in top5}

                    # Total expected minutes/pts from core
                    total_core_min = sum(p[1]['min'] for p in top8)
                    total_core_pts = sum(p[1]['pts'] for p in top8)
                    total_core_gp = sum(p[1]['gp'] for p in top8)

                    # Who's missing from the core?
                    missing_from_top8 = top8_ids - current_player_ids
                    missing_from_top3 = top3_ids - current_player_ids
                    missing_from_top5 = top5_ids - current_player_ids

                    # How many core players missing
                    features['CORE8_MISSING'] = len(missing_from_top8)
                    features['CORE3_MISSING'] = len(missing_from_top3)
                    features['CORE5_MISSING'] = len(missing_from_top5)

                    # Fraction of core available
                    features['CORE8_AVAILABLE_PCT'] = 1.0 - len(missing_from_top8) / len(top8)
                    features['CORE5_AVAILABLE_PCT'] = 1.0 - len(missing_from_top5) / len(top5)

                    # Star flags
                    features['STAR1_OUT'] = 1.0 if top8[0][0] not in current_player_ids else 0.0
                    features['STAR2_OUT'] = 1.0 if len(top8) > 1 and top8[1][0] not in current_player_ids else 0.0
                    features['TOP2_BOTH_OUT'] = 1.0 if features['STAR1_OUT'] and features['STAR2_OUT'] else 0.0

                    # Weighted minutes/points lost from missing players
                    missing_min_frac = 0.0
                    missing_pts_frac = 0.0
                    missing_pm_impact = 0.0
                    for p_id, stats in top8:
                        if p_id not in current_player_ids:
                            if total_core_min > 0:
                                missing_min_frac += stats['min'] / total_core_min
                            if total_core_pts > 0:
                                missing_pts_frac += stats['pts'] / total_core_pts
                            if stats['gp'] > 0:
                                missing_pm_impact += stats['pm'] / stats['gp']

                    features['MISSING_MIN_FRAC'] = missing_min_frac
                    features['MISSING_PTS_FRAC'] = missing_pts_frac
                    features['MISSING_PM_IMPACT'] = missing_pm_impact

                    # Average quality of available players (avg plus/minus)
                    available_pm = []
                    for p_id, stats in sorted_players:
                        if p_id in current_player_ids and stats['gp'] > 0:
                            available_pm.append(stats['pm'] / stats['gp'])
                    features['AVG_AVAILABLE_PM'] = np.mean(available_pm) if available_pm else 0.0

                    # Roster depth: total players who've played this season
                    features['ROSTER_DEPTH'] = len(cum_stats)

                    # Current game roster size
                    features['ACTIVE_PLAYERS'] = len(current_player_ids)

                    # Star player's avg PTS and MIN (for interaction later)
                    star_avg_pts = top8[0][1]['pts'] / max(1, top8[0][1]['gp'])
                    star_avg_min = top8[0][1]['min'] / max(1, top8[0][1]['gp'])
                    features['STAR_AVG_PTS'] = star_avg_pts
                    features['STAR_AVG_MIN'] = star_avg_min

                    # ── Roster stability: unique players in last 5 and 10 games ──
                    if len(roster_last_5) >= 5:
                        unique_5 = len(set().union(*roster_last_5[-5:]))
                        features['UNIQUE_PLAYERS_LAST5'] = unique_5
                    else:
                        features['UNIQUE_PLAYERS_LAST5'] = float('nan')

                    if len(roster_last_10) >= 10:
                        unique_10 = len(set().union(*roster_last_10[-10:]))
                        features['UNIQUE_PLAYERS_LAST10'] = unique_10
                    else:
                        features['UNIQUE_PLAYERS_LAST10'] = float('nan')

                    # ── Consecutive games with a core player out ──
                    # How many of the last 5 games had ANY top-3 player missing?
                    # (computed from roster_last_5)
                    if len(roster_last_5) >= 3:
                        recent_top3_missing = 0
                        for recent_roster in roster_last_5[-5:]:
                            if top3_ids - recent_roster:
                                recent_top3_missing += 1
                        features['RECENT_TOP3_MISSING_COUNT'] = recent_top3_missing
                    else:
                        features['RECENT_TOP3_MISSING_COUNT'] = float('nan')

                else:
                    # Not enough player history yet
                    features = {k: float('nan') for k in [
                        'CORE8_MISSING', 'CORE3_MISSING', 'CORE5_MISSING',
                        'CORE8_AVAILABLE_PCT', 'CORE5_AVAILABLE_PCT',
                        'STAR1_OUT', 'STAR2_OUT', 'TOP2_BOTH_OUT',
                        'MISSING_MIN_FRAC', 'MISSING_PTS_FRAC', 'MISSING_PM_IMPACT',
                        'AVG_AVAILABLE_PM', 'ROSTER_DEPTH', 'ACTIVE_PLAYERS',
                        'STAR_AVG_PTS', 'STAR_AVG_MIN',
                        'UNIQUE_PLAYERS_LAST5', 'UNIQUE_PLAYERS_LAST10',
                        'RECENT_TOP3_MISSING_COUNT'
                    ]}
            else:
                # First 5 games of season — not enough history
                features = {k: float('nan') for k in [
                    'CORE8_MISSING', 'CORE3_MISSING', 'CORE5_MISSING',
                    'CORE8_AVAILABLE_PCT', 'CORE5_AVAILABLE_PCT',
                    'STAR1_OUT', 'STAR2_OUT', 'TOP2_BOTH_OUT',
                    'MISSING_MIN_FRAC', 'MISSING_PTS_FRAC', 'MISSING_PM_IMPACT',
                    'AVG_AVAILABLE_PM', 'ROSTER_DEPTH', 'ACTIVE_PLAYERS',
                    'STAR_AVG_PTS', 'STAR_AVG_MIN',
                    'UNIQUE_PLAYERS_LAST5', 'UNIQUE_PLAYERS_LAST10',
                    'RECENT_TOP3_MISSING_COUNT'
                ]}

            team_game_features[(game_id, team_id)] = features

            # ── Update cumulative stats AFTER computing features (no leakage) ──
            for _, row in game_players.iterrows():
                pid = row['PLAYER_ID']
                if pid not in cum_stats:
                    cum_stats[pid] = {'min': 0, 'pts': 0, 'pm': 0, 'gp': 0}
                cum_stats[pid]['min'] += row['MIN']
                cum_stats[pid]['pts'] += row['PTS']
                cum_stats[pid]['pm'] += row['PLUS_MINUS']
                cum_stats[pid]['gp'] += 1

            roster_last_5.append(current_player_ids)
            roster_last_10.append(current_player_ids)
            if len(roster_last_5) > 5:
                roster_last_5 = roster_last_5[-5:]
            if len(roster_last_10) > 10:
                roster_last_10 = roster_last_10[-10:]

    print(f"  Computed features for {len(team_game_features)} team-game pairs")

    # ── Pair home/away and build final dataframe ─────────────────────────────

    L6_FEAT_NAMES = [
        'CORE8_MISSING', 'CORE3_MISSING', 'CORE5_MISSING',
        'CORE8_AVAILABLE_PCT', 'CORE5_AVAILABLE_PCT',
        'STAR1_OUT', 'STAR2_OUT', 'TOP2_BOTH_OUT',
        'MISSING_MIN_FRAC', 'MISSING_PTS_FRAC', 'MISSING_PM_IMPACT',
        'AVG_AVAILABLE_PM', 'ROSTER_DEPTH', 'ACTIVE_PLAYERS',
        'STAR_AVG_PTS', 'STAR_AVG_MIN',
        'UNIQUE_PLAYERS_LAST5', 'UNIQUE_PLAYERS_LAST10',
        'RECENT_TOP3_MISSING_COUNT'
    ]

    rows = []
    for _, gp in game_pairs.iterrows():
        game_id = gp['GAME_ID']
        home_id = gp['HOME_TEAM_ID']
        away_id = gp['AWAY_TEAM_ID']

        home_feat = team_game_features.get((game_id, home_id), None)
        away_feat = team_game_features.get((game_id, away_id), None)

        if home_feat is None or away_feat is None:
            continue

        row = {
            'GAME_ID': game_id,
            'HOME_TEAM_ID': home_id,
            'AWAY_TEAM_ID': away_id,
            'SEASON': gp['SEASON'],
            'GAME_DATE': gp['GAME_DATE'],
            'POINT_DIFF': gp['POINT_DIFF'],
        }
        for feat in L6_FEAT_NAMES:
            row[f'HOME_{feat}'] = home_feat.get(feat, float('nan'))
            row[f'AWAY_{feat}'] = away_feat.get(feat, float('nan'))

        rows.append(row)

    df_l6 = pd.DataFrame(rows)

    # ── Add differential features ────────────────────────────────────────────
    for feat in L6_FEAT_NAMES:
        hcol = f'HOME_{feat}'
        acol = f'AWAY_{feat}'
        if hcol in df_l6.columns and acol in df_l6.columns:
            df_l6[f'DIFF_{feat}'] = df_l6[hcol] - df_l6[acol]

    # ── Add interaction features ──────────────────────────────────────────────
    # Star out × opponent quality (if opponent is strong, star absence hurts more)
    df_l6['HOME_STAR_OUT_x_OPP_PM'] = df_l6['HOME_STAR1_OUT'] * df_l6['AWAY_AVG_AVAILABLE_PM']
    df_l6['AWAY_STAR_OUT_x_OPP_PM'] = df_l6['AWAY_STAR1_OUT'] * df_l6['HOME_AVG_AVAILABLE_PM']

    # Missing minutes × star quality
    df_l6['HOME_MISSING_MIN_x_STAR_PTS'] = df_l6['HOME_MISSING_MIN_FRAC'] * df_l6['HOME_STAR_AVG_PTS']
    df_l6['AWAY_MISSING_MIN_x_STAR_PTS'] = df_l6['AWAY_MISSING_MIN_FRAC'] * df_l6['AWAY_STAR_AVG_PTS']

    df_l6['HOME_ADV'] = 1.0

    print(f"\n  Final L6 dataframe: {df_l6.shape}")
    print(f"  Non-null rows: {df_l6.dropna().shape[0]}")

    # Save
    df_l6.to_csv(LINEUP_CSV, index=False)
    print(f"  Saved to {LINEUP_CSV}")

    # ── Sanity checks ────────────────────────────────────────────────────────
    print("\n  ── Sanity Checks ──")
    non_null = df_l6.dropna()
    print(f"  CORE8_MISSING: mean={non_null['HOME_CORE8_MISSING'].mean():.2f}, "
          f"max={non_null['HOME_CORE8_MISSING'].max():.0f}")
    print(f"  STAR1_OUT rate: {non_null['HOME_STAR1_OUT'].mean():.3f} "
          f"({non_null['HOME_STAR1_OUT'].mean()*100:.1f}%)")
    print(f"  MISSING_MIN_FRAC: mean={non_null['HOME_MISSING_MIN_FRAC'].mean():.3f}")
    print(f"  ROSTER_DEPTH: mean={non_null['HOME_ROSTER_DEPTH'].mean():.1f}")
    print(f"  AVG_AVAILABLE_PM: mean={non_null['HOME_AVG_AVAILABLE_PM'].mean():.2f}")

    return df_l6


# ══════════════════════════════════════════════════════════════════════════════
# BUILD OR LOAD L6
# ══════════════════════════════════════════════════════════════════════════════

if os.path.exists(LINEUP_CSV):
    print(f"\nLoading cached L6 from {LINEUP_CSV}")
    df_l6 = pd.read_csv(LINEUP_CSV)
    print(f"  Shape: {df_l6.shape}")
else:
    df_l6 = build_lineup_features()

# ══════════════════════════════════════════════════════════════════════════════
# LOAD L1 AND L5, MERGE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("LOADING L1 & L5, MERGING WITH L6")
print("=" * 70)

df_l1 = pd.read_csv('data/processed/level1_season_agg.csv')
df_l5 = pd.read_csv('data/processed/level5_context.csv')

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

# L5 features
L5_BASE = [
    'HOME_REST_DAYS', 'HOME_IS_B2B', 'HOME_GAMES_IN_LAST_7',
    'HOME_WIN_STREAK', 'HOME_GAME_NUMBER', 'HOME_CONSECUTIVE_AWAY',
    'HOME_HOME_WIN_PCT', 'HOME_AWAY_WIN_PCT', 'HOME_OVERALL_WIN_PCT',
    'AWAY_REST_DAYS', 'AWAY_IS_B2B', 'AWAY_GAMES_IN_LAST_7',
    'AWAY_WIN_STREAK', 'AWAY_GAME_NUMBER', 'AWAY_CONSECUTIVE_AWAY',
    'AWAY_HOME_WIN_PCT', 'AWAY_AWAY_WIN_PCT', 'AWAY_OVERALL_WIN_PCT',
    'REST_ADVANTAGE', 'DIFF_WIN_STREAK', 'DIFF_GAMES_IN_LAST_7',
    'DIFF_GAME_NUMBER', 'DIFF_OVERALL_WIN_PCT',
]

# L6 feature list
L6_FEAT_NAMES = [
    'CORE8_MISSING', 'CORE3_MISSING', 'CORE5_MISSING',
    'CORE8_AVAILABLE_PCT', 'CORE5_AVAILABLE_PCT',
    'STAR1_OUT', 'STAR2_OUT', 'TOP2_BOTH_OUT',
    'MISSING_MIN_FRAC', 'MISSING_PTS_FRAC', 'MISSING_PM_IMPACT',
    'AVG_AVAILABLE_PM', 'ROSTER_DEPTH', 'ACTIVE_PLAYERS',
    'STAR_AVG_PTS', 'STAR_AVG_MIN',
    'UNIQUE_PLAYERS_LAST5', 'UNIQUE_PLAYERS_LAST10',
    'RECENT_TOP3_MISSING_COUNT'
]
L6_BASE = []
for feat in L6_FEAT_NAMES:
    for side in ['HOME', 'AWAY']:
        L6_BASE.append(f'{side}_{feat}')
    L6_BASE.append(f'DIFF_{feat}')
# Add interaction features
L6_INTERACTIONS = [
    'HOME_STAR_OUT_x_OPP_PM', 'AWAY_STAR_OUT_x_OPP_PM',
    'HOME_MISSING_MIN_x_STAR_PTS', 'AWAY_MISSING_MIN_x_STAR_PTS',
]
L6_ALL = L6_BASE + L6_INTERACTIONS + ['HOME_ADV']
L6_ALL = [c for c in L6_ALL if c in df_l6.columns]

# ── Merge everything ─────────────────────────────────────────────────────────
# L1 + L6
l5_merge_cols = ['GAME_ID'] + [c for c in L5_BASE if c in df_l5.columns]
df_all = df_l1.merge(df_l6.drop(columns=['SEASON', 'POINT_DIFF', 'HOME_ADV', 'GAME_DATE'],
                                 errors='ignore'),
                      on='GAME_ID', how='inner')
df_all = df_all.merge(df_l5[l5_merge_cols], on='GAME_ID', how='inner')
df_all['HOME_ADV'] = 1.0

# Drop rows with NaN in L6 features (first ~5 games per team per season)
l6_cols_in_df = [c for c in L6_ALL if c in df_all.columns]
pre_drop = len(df_all)
df_all = df_all.dropna(subset=l6_cols_in_df)
print(f"  Dropped {pre_drop - len(df_all)} rows with NaN L6 features ({len(df_all)} remaining)")

# Feature sets
FEAT_L1 = [c for c in L1_FEAT if c in df_all.columns]
FEAT_L6 = [c for c in L6_ALL if c in df_all.columns]
FEAT_L1_L6 = list(dict.fromkeys(FEAT_L1 + FEAT_L6))
FEAT_L1_L5 = list(dict.fromkeys(FEAT_L1 + [c for c in L5_BASE if c in df_all.columns]))
FEAT_L1_L5_L6 = list(dict.fromkeys(FEAT_L1 + [c for c in L5_BASE if c in df_all.columns] + FEAT_L6))

# Enhanced interactions between L6 and L1
if 'DIFF_NET_RATING' in df_all.columns:
    # Star out × net rating (better teams lose more when star is out)
    df_all['HOME_STAR_OUT_x_NETRTG'] = df_all['HOME_STAR1_OUT'] * df_all['HOME_NET_RATING']
    df_all['AWAY_STAR_OUT_x_NETRTG'] = df_all['AWAY_STAR1_OUT'] * df_all['AWAY_NET_RATING']
    # Missing fraction × net rating
    df_all['HOME_MISSING_x_NETRTG'] = df_all['HOME_MISSING_MIN_FRAC'] * df_all['HOME_NET_RATING']
    df_all['AWAY_MISSING_x_NETRTG'] = df_all['AWAY_MISSING_MIN_FRAC'] * df_all['AWAY_NET_RATING']
    # Roster stability × quality
    df_all['HOME_STABILITY_x_NETRTG'] = df_all['HOME_UNIQUE_PLAYERS_LAST10'] * df_all['HOME_NET_RATING']
    df_all['AWAY_STABILITY_x_NETRTG'] = df_all['AWAY_UNIQUE_PLAYERS_LAST10'] * df_all['AWAY_NET_RATING']

    L6_X_L1_INTERACTIONS = [
        'HOME_STAR_OUT_x_NETRTG', 'AWAY_STAR_OUT_x_NETRTG',
        'HOME_MISSING_x_NETRTG', 'AWAY_MISSING_x_NETRTG',
        'HOME_STABILITY_x_NETRTG', 'AWAY_STABILITY_x_NETRTG',
    ]
    FEAT_L1_L5_L6_ENHANCED = list(dict.fromkeys(FEAT_L1_L5_L6 + L6_X_L1_INTERACTIONS))
else:
    FEAT_L1_L5_L6_ENHANCED = FEAT_L1_L5_L6

print(f"\n  Feature counts:")
print(f"    L6 standalone:          {len(FEAT_L6)}")
print(f"    L1 only:                {len(FEAT_L1)}")
print(f"    L1 + L6:                {len(FEAT_L1_L6)}")
print(f"    L1 + L5:                {len(FEAT_L1_L5)}")
print(f"    L1 + L5 + L6:           {len(FEAT_L1_L5_L6)}")
print(f"    L1 + L5 + L6 enhanced:  {len(FEAT_L1_L5_L6_ENHANCED)}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: BASELINE COMPARISON — ALL FEATURE SETS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 1: BASELINE MODELS (Ridge/XGBoost/NN) × ALL FEATURE SETS")
print("=" * 70)

results = []

feature_sets = {
    'L6_only': FEAT_L6,
    'L1_only': FEAT_L1,
    'L1+L6': FEAT_L1_L6,
    'L1+L5': FEAT_L1_L5,
    'L1+L5+L6': FEAT_L1_L5_L6,
    'L1+L5+L6_enh': FEAT_L1_L5_L6_ENHANCED,
}

for fs_name, feats in feature_sets.items():
    print(f"\n  ── {fs_name} ({len(feats)} features) ──")
    X_tr, y_tr, X_v, y_v, X_te, y_te, used_feats, scaler = split_and_scale(df_all, feats)
    print(f"    Train: {X_tr.shape}, Val: {X_v.shape}, Test: {X_te.shape}")

    # Ridge
    t0 = time.time()
    ridge = RidgeCV(alphas=np.logspace(-2, 4, 50))
    ridge.fit(X_tr, y_tr)
    ridge_pred = ridge.predict(X_te)
    ridge_m = eval_metrics(y_te, ridge_pred)
    ridge_m['model'] = 'Ridge'
    ridge_m['feature_set'] = fs_name
    ridge_m['n_features'] = len(used_feats)
    ridge_m['train_time'] = time.time() - t0
    results.append(ridge_m)
    print(f"    Ridge:   MAE={ridge_m['mae']:.3f}, R²={ridge_m['r2']:.3f} (α={ridge.alpha_:.1f})")

    # XGBoost
    t0 = time.time()
    xgb = XGBRegressor(
        n_estimators=1000, max_depth=5, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7, min_child_weight=3,
        tree_method='hist', device='cuda',
        early_stopping_rounds=30, eval_metric='mae',
        random_state=SEED, verbosity=0
    )
    xgb.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
    xgb_pred = xgb.predict(X_te)
    xgb_m = eval_metrics(y_te, xgb_pred)
    xgb_m['model'] = 'XGBoost'
    xgb_m['feature_set'] = fs_name
    xgb_m['n_features'] = len(used_feats)
    xgb_m['train_time'] = time.time() - t0
    results.append(xgb_m)
    print(f"    XGBoost: MAE={xgb_m['mae']:.3f}, R²={xgb_m['r2']:.3f} (trees={xgb.best_iteration})")

    # NN (3-seed average)
    t0 = time.time()
    nn_preds = []
    for seed_i in range(3):
        torch.manual_seed(SEED + seed_i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED + seed_i)
        nn_m, _, _, _, _ = train_nn(X_tr, y_tr, X_v, y_v, len(used_feats),
                                     hidden_dims=(64, 32), lr=2e-3, dropout=0.4,
                                     weight_decay=1e-5, batch_size=128, patience=25)
        nn_preds.append(predict_nn(nn_m, X_te))
    nn_pred = np.mean(nn_preds, axis=0)
    nn_m = eval_metrics(y_te, nn_pred)
    nn_m['model'] = 'NN'
    nn_m['feature_set'] = fs_name
    nn_m['n_features'] = len(used_feats)
    nn_m['train_time'] = time.time() - t0
    results.append(nn_m)
    print(f"    NN:      MAE={nn_m['mae']:.3f}, R²={nn_m['r2']:.3f}")

df_results = pd.DataFrame(results)
df_results.to_csv('results/lineup_baseline_results.csv', index=False)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: XGBOOST GRID SEARCH ON BEST COMBINED SET
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 2: XGBOOST GRID SEARCH (L1+L5+L6 enhanced)")
print("=" * 70)

X_tr, y_tr, X_v, y_v, X_te, y_te, used_feats, scaler = split_and_scale(
    df_all, FEAT_L1_L5_L6_ENHANCED)

xgb_configs = []
for md in [3, 4, 5, 6, 7]:
    for lr in [0.005, 0.01, 0.02, 0.03, 0.05]:
        for ss in [0.6, 0.7, 0.8]:
            for csb in [0.6, 0.7, 0.8]:
                for mcw in [1, 3, 5, 10]:
                    xgb_configs.append({
                        'max_depth': md, 'learning_rate': lr,
                        'subsample': ss, 'colsample_bytree': csb,
                        'min_child_weight': mcw
                    })

print(f"  Testing {len(xgb_configs)} configurations...")
xgb_results = []
best_xgb_mae = float('inf')
best_xgb_cfg = None

for i, cfg in enumerate(xgb_configs):
    xgb_m = XGBRegressor(
        n_estimators=1000, early_stopping_rounds=30, eval_metric='mae',
        tree_method='hist', device='cuda', random_state=SEED, verbosity=0,
        **cfg
    )
    xgb_m.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
    val_pred = xgb_m.predict(X_v)
    val_mae = mean_absolute_error(y_v, val_pred)
    test_pred = xgb_m.predict(X_te)
    test_m = eval_metrics(y_te, test_pred)

    xgb_results.append({**cfg, 'val_mae': val_mae, 'test_mae': test_m['mae'],
                         'test_r2': test_m['r2'], 'trees': xgb_m.best_iteration})

    if val_mae < best_xgb_mae:
        best_xgb_mae = val_mae
        best_xgb_cfg = cfg
        best_xgb_test = test_m

    if (i + 1) % 100 == 0:
        print(f"    {i+1}/{len(xgb_configs)} done, best val MAE={best_xgb_mae:.3f}")

print(f"\n  Best XGBoost config: {best_xgb_cfg}")
print(f"  Best XGBoost test MAE={best_xgb_test['mae']:.3f}, R²={best_xgb_test['r2']:.3f}")

pd.DataFrame(xgb_results).to_csv('results/xgb_lineup_sweep.csv', index=False)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: NN ARCHITECTURE SWEEP
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 3: NN SWEEP (L1+L5+L6 enhanced)")
print("=" * 70)

nn_configs = []
for hd in [(64, 32), (128, 64), (128, 64, 32), (256, 128, 64), (256, 128)]:
    for lr in [5e-4, 1e-3, 2e-3]:
        for drop in [0.2, 0.3, 0.4, 0.5]:
            for wd in [1e-5, 1e-4, 1e-3]:
                for bs in [64, 128, 256]:
                    nn_configs.append({
                        'hidden_dims': hd, 'lr': lr, 'dropout': drop,
                        'weight_decay': wd, 'batch_size': bs
                    })

print(f"  Testing {len(nn_configs)} configurations...")
nn_results_list = []
best_nn_mae = float('inf')
best_nn_cfg = None

for i, cfg in enumerate(nn_configs):
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    nn_m, val_loss, val_mae, _, _ = train_nn(
        X_tr, y_tr, X_v, y_v, len(used_feats),
        hidden_dims=cfg['hidden_dims'], lr=cfg['lr'],
        dropout=cfg['dropout'], weight_decay=cfg['weight_decay'],
        batch_size=cfg['batch_size'], patience=20
    )
    test_pred = predict_nn(nn_m, X_te)
    test_m = eval_metrics(y_te, test_pred)

    nn_results_list.append({
        'hidden_dims': str(cfg['hidden_dims']),
        'lr': cfg['lr'], 'dropout': cfg['dropout'],
        'weight_decay': cfg['weight_decay'], 'batch_size': cfg['batch_size'],
        'val_mae': val_mae, 'test_mae': test_m['mae'], 'test_r2': test_m['r2']
    })

    if val_mae < best_nn_mae:
        best_nn_mae = val_mae
        best_nn_cfg = cfg
        best_nn_test = test_m

    if (i + 1) % 50 == 0:
        print(f"    {i+1}/{len(nn_configs)} done, best val MAE={best_nn_mae:.3f}")

print(f"\n  Best NN config: {best_nn_cfg}")
print(f"  Best NN test MAE={best_nn_test['mae']:.3f}, R²={best_nn_test['r2']:.3f}")

pd.DataFrame(nn_results_list).to_csv('results/nn_lineup_sweep.csv', index=False)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4: ENSEMBLE — BEST MODELS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 4: ENSEMBLE (Ridge + XGBoost + NN)")
print("=" * 70)

# Train best models on the best combined feature set
X_tr, y_tr, X_v, y_v, X_te, y_te, used_feats, scaler = split_and_scale(
    df_all, FEAT_L1_L5_L6_ENHANCED)

# Ridge
ridge_best = RidgeCV(alphas=np.logspace(-2, 4, 50))
ridge_best.fit(X_tr, y_tr)
pred_ridge_v = ridge_best.predict(X_v)
pred_ridge_t = ridge_best.predict(X_te)

# XGBoost with best config
xgb_best = XGBRegressor(
    n_estimators=1000, early_stopping_rounds=30, eval_metric='mae',
    tree_method='hist', device='cuda', random_state=SEED, verbosity=0,
    **best_xgb_cfg
)
xgb_best.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
pred_xgb_v = xgb_best.predict(X_v)
pred_xgb_t = xgb_best.predict(X_te)

# NN with 5-seed averaging using best config
nn_preds_v, nn_preds_t = [], []
nn_train_losses_all, nn_val_losses_all = [], []
for seed_i in range(5):
    torch.manual_seed(SEED + seed_i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED + seed_i)
    nn_m, _, _, tl, vl = train_nn(
        X_tr, y_tr, X_v, y_v, len(used_feats),
        hidden_dims=best_nn_cfg['hidden_dims'], lr=best_nn_cfg['lr'],
        dropout=best_nn_cfg['dropout'], weight_decay=best_nn_cfg['weight_decay'],
        batch_size=best_nn_cfg['batch_size'], patience=25
    )
    nn_preds_v.append(predict_nn(nn_m, X_v))
    nn_preds_t.append(predict_nn(nn_m, X_te))
    nn_train_losses_all.append(tl)
    nn_val_losses_all.append(vl)

pred_nn_v = np.mean(nn_preds_v, axis=0)
pred_nn_t = np.mean(nn_preds_t, axis=0)

print(f"  Ridge test:   {eval_metrics(y_te, pred_ridge_t)}")
print(f"  XGBoost test: {eval_metrics(y_te, pred_xgb_t)}")
print(f"  NN test:      {eval_metrics(y_te, pred_nn_t)}")

# Sweep ensemble weights
best_ens_mae = float('inf')
best_w = None
for w1 in np.arange(0, 1.05, 0.05):
    for w2 in np.arange(0, 1.05 - w1, 0.05):
        w3 = 1.0 - w1 - w2
        if w3 < -0.01:
            continue
        pred_v = w1 * pred_ridge_v + w2 * pred_xgb_v + w3 * pred_nn_v
        mae_v = mean_absolute_error(y_v, pred_v)
        if mae_v < best_ens_mae:
            best_ens_mae = mae_v
            best_w = (w1, w2, w3)

print(f"\n  Best ensemble weights: Ridge={best_w[0]:.2f}, XGB={best_w[1]:.2f}, NN={best_w[2]:.2f}")
pred_ens_t = best_w[0] * pred_ridge_t + best_w[1] * pred_xgb_t + best_w[2] * pred_nn_t
ens_m = eval_metrics(y_te, pred_ens_t)
print(f"  Ensemble test: MAE={ens_m['mae']:.3f}, R²={ens_m['r2']:.3f}")

# Also test L1+L5 ensemble for fair comparison
print("\n  ── L1+L5 baseline (for comparison) ──")
X_tr_l5, y_tr_l5, X_v_l5, y_v_l5, X_te_l5, y_te_l5, used_feats_l5, _ = split_and_scale(
    df_all, FEAT_L1_L5)
ridge_l5 = RidgeCV(alphas=np.logspace(-2, 4, 50))
ridge_l5.fit(X_tr_l5, y_tr_l5)
pred_l5 = ridge_l5.predict(X_te_l5)
l5_m = eval_metrics(y_te_l5, pred_l5)
print(f"  L1+L5 Ridge: MAE={l5_m['mae']:.3f}, R²={l5_m['r2']:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5: FEATURE IMPORTANCE & SELECTION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 5: FEATURE IMPORTANCE & SELECTION")
print("=" * 70)

importances = xgb_best.feature_importances_
feat_imp = sorted(zip(used_feats, importances), key=lambda x: x[1], reverse=True)
print("\n  Top 25 features (XGBoost):")
for i, (f, imp) in enumerate(feat_imp[:25]):
    print(f"    {i+1:2d}. {f:40s} {imp:.4f}")

# Test feature selection on Ridge
for top_k in [15, 20, 25, 30, 40, 50]:
    top_feats = [f for f, _ in feat_imp[:top_k]]
    X_tr_k, y_tr_k, X_v_k, y_v_k, X_te_k, y_te_k, _, _ = split_and_scale(df_all, top_feats)
    ridge_k = RidgeCV(alphas=np.logspace(-2, 4, 50))
    ridge_k.fit(X_tr_k, y_tr_k)
    pred_k = ridge_k.predict(X_te_k)
    m_k = eval_metrics(y_te_k, pred_k)
    print(f"  Top-{top_k:2d} Ridge: MAE={m_k['mae']:.3f}, R²={m_k['r2']:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6: VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 6: GENERATING VISUALIZATIONS")
print("=" * 70)

# ── 1. MAE Comparison Bar Chart ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
df_bar = df_results.copy()
pivot = df_bar.pivot(index='feature_set', columns='model', values='mae')
# Order feature sets logically
order = ['L6_only', 'L1_only', 'L1+L6', 'L1+L5', 'L1+L5+L6', 'L1+L5+L6_enh']
pivot = pivot.reindex([o for o in order if o in pivot.index])
pivot.plot(kind='bar', ax=ax, width=0.7)
ax.set_ylabel('Test MAE (points)')
ax.set_title('MAE by Feature Set and Model Type')
ax.set_xlabel('Feature Set')
ax.legend(title='Model')
ax.axhline(y=l5_m['mae'], color='red', linestyle='--', alpha=0.7, label=f'L1+L5 Ridge ({l5_m["mae"]:.3f})')
ax.legend()
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('results/lineup_mae_comparison.png', dpi=150)
plt.close()
print("  Saved results/lineup_mae_comparison.png")

# ── 2. MAE Heatmap ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
hm_data = pivot.values
sns.heatmap(hm_data, annot=True, fmt='.3f', cmap='RdYlGn_r',
            xticklabels=pivot.columns, yticklabels=pivot.index, ax=ax)
ax.set_title('MAE Heatmap: Feature Sets × Models')
plt.tight_layout()
plt.savefig('results/lineup_mae_heatmap.png', dpi=150)
plt.close()
print("  Saved results/lineup_mae_heatmap.png")

# ── 3. Feature Importance Plot ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
top20 = feat_imp[:20]
names = [f for f, _ in top20]
vals = [v for _, v in top20]
colors = []
for n in names:
    if any(x in n for x in ['CORE', 'STAR', 'MISSING', 'ROSTER', 'ACTIVE', 'UNIQUE', 'RECENT']):
        colors.append('#e74c3c')  # L6 features in red
    elif any(x in n for x in ['REST', 'B2B', 'STREAK', 'FATIGUE', 'CONSECUTIVE']):
        colors.append('#3498db')  # L5 features in blue
    else:
        colors.append('#2ecc71')  # L1 features in green
ax.barh(range(len(names)), vals, color=colors)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Feature Importance (XGBoost gain)')
ax.set_title('Top 20 Features — L1+L5+L6 Combined Model')
# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2ecc71', label='L1 (Season Stats)'),
                   Patch(facecolor='#3498db', label='L5 (Schedule)'),
                   Patch(facecolor='#e74c3c', label='L6 (Lineup)')]
ax.legend(handles=legend_elements, loc='lower right')
plt.tight_layout()
plt.savefig('results/lineup_feature_importance.png', dpi=150)
plt.close()
print("  Saved results/lineup_feature_importance.png")

# ── 4. Residual Plots ───────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

predictions = {
    'Ridge': pred_ridge_t,
    'XGBoost': pred_xgb_t,
    'NN (5-seed)': pred_nn_t,
    'Ensemble': pred_ens_t
}
for ax, (name, pred) in zip(axes.flatten(), predictions.items()):
    m = eval_metrics(y_te, pred)
    ax.scatter(pred, y_te, alpha=0.15, s=8, color='steelblue')
    lims = [-50, 50]
    ax.plot(lims, lims, 'r--', linewidth=1, alpha=0.7)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'{name}\nMAE={m["mae"]:.2f}, R²={m["r2"]:.3f}')
    ax.set_aspect('equal')
fig.suptitle('L1+L5+L6 Enhanced — Predicted vs Actual Point Differential', fontsize=13)
plt.tight_layout()
plt.savefig('results/lineup_residual_plots.png', dpi=150)
plt.close()
print("  Saved results/lineup_residual_plots.png")

# ── 5. NN Learning Curves ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
for i, (tl, vl) in enumerate(zip(nn_train_losses_all, nn_val_losses_all)):
    ax.plot(tl, alpha=0.3, color='blue', label='Train' if i == 0 else None)
    ax.plot(vl, alpha=0.3, color='orange', label='Val' if i == 0 else None)
# Average
max_len = max(len(tl) for tl in nn_train_losses_all)
avg_train = np.array([np.mean([tl[e] for tl in nn_train_losses_all if e < len(tl)])
                       for e in range(max_len)])
avg_val = np.array([np.mean([vl[e] for vl in nn_val_losses_all if e < len(vl)])
                     for e in range(max_len)])
ax.plot(avg_train, color='blue', linewidth=2, label='Train (avg)')
ax.plot(avg_val, color='orange', linewidth=2, label='Val (avg)')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('NN Learning Curves — L1+L5+L6 Enhanced (5 seeds)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/lineup_nn_learning_curves.png', dpi=150)
plt.close()
print("  Saved results/lineup_nn_learning_curves.png")

# ── 6. Residual Distribution ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
residuals = y_te - pred_ens_t
axes[0].hist(residuals, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].axvline(x=0, color='red', linestyle='--')
axes[0].set_xlabel('Residual (Actual - Predicted)')
axes[0].set_ylabel('Count')
axes[0].set_title(f'Ensemble Residual Distribution\nMean={residuals.mean():.2f}, Std={residuals.std():.2f}')

# QQ plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot (Ensemble Residuals)')
plt.tight_layout()
plt.savefig('results/lineup_residual_distribution.png', dpi=150)
plt.close()
print("  Saved results/lineup_residual_distribution.png")

# ── 7. Comparison with L1+L5 (the key plot) ──────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
compare_data = {
    'L1 Ridge\n(baseline)': 10.610,
    'L1+L5 Ridge': l5_m['mae'],
    'L1+L5+L6 Ridge': [r for r in results if r['feature_set'] == 'L1+L5+L6_enh' and r['model'] == 'Ridge'][0]['mae'],
    'L1+L5+L6 XGBoost': best_xgb_test['mae'],
    'L1+L5+L6 NN': eval_metrics(y_te, pred_nn_t)['mae'],
    'L1+L5+L6 Ensemble': ens_m['mae'],
    'L1+L5 Ensemble\n(prev best)': 10.480,
}
names = list(compare_data.keys())
vals = list(compare_data.values())
colors = ['#95a5a6', '#3498db', '#e74c3c', '#e74c3c', '#e74c3c', '#e74c3c', '#3498db']
bars = ax.bar(range(len(names)), vals, color=colors, edgecolor='white', width=0.6)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, fontsize=9, rotation=20, ha='right')
ax.set_ylabel('Test MAE (points)')
ax.set_title('MAE Comparison: Does Lineup Data Help?')
ax.axhline(y=10.480, color='blue', linestyle='--', alpha=0.5, label='Previous best (10.480)')
for i, v in enumerate(vals):
    ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
ax.set_ylim(min(vals) - 0.3, max(vals) + 0.3)
ax.legend()
plt.tight_layout()
plt.savefig('results/lineup_vs_l5_comparison.png', dpi=150)
plt.close()
print("  Saved results/lineup_vs_l5_comparison.png")

# ── 8. Lineup Impact Analysis ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 8a. MAE when star is out vs in
non_null = df_all.dropna(subset=['HOME_STAR1_OUT'])
test_mask = non_null['SEASON'].isin(TEST_SEASONS)
test_data = non_null[test_mask]

# Split by star status
star_out = test_data[test_data['HOME_STAR1_OUT'] == 1]
star_in = test_data[test_data['HOME_STAR1_OUT'] == 0]
axes[0].bar(['Star Playing', 'Star Out'],
            [star_in['POINT_DIFF'].std(), star_out['POINT_DIFF'].std()],
            color=['#2ecc71', '#e74c3c'])
axes[0].set_ylabel('Point Diff Std Dev')
axes[0].set_title(f'Variance by Star Status\n(Out: {len(star_out)} games, In: {len(star_in)} games)')

# 8b. Point diff by core availability
test_data_copy = test_data.copy()
test_data_copy['core_bucket'] = pd.cut(test_data_copy['HOME_CORE8_AVAILABLE_PCT'],
                                        bins=[0, 0.6, 0.75, 0.875, 1.0],
                                        labels=['≤60%', '62-75%', '75-87%', '87-100%'])
grouped = test_data_copy.groupby('core_bucket')['POINT_DIFF'].mean()
axes[1].bar(grouped.index.astype(str), grouped.values, color='steelblue')
axes[1].set_ylabel('Mean Home Point Diff')
axes[1].set_title('Home Point Diff by Core Roster Availability')
axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# 8c. Distribution of core8 missing
axes[2].hist(test_data['HOME_CORE8_MISSING'].values, bins=range(0, 9),
             color='steelblue', edgecolor='white', alpha=0.8, align='left')
axes[2].set_xlabel('Number of Core-8 Players Missing')
axes[2].set_ylabel('Count')
axes[2].set_title('Distribution of Missing Core Players (Home Team)')

plt.tight_layout()
plt.savefig('results/lineup_impact_analysis.png', dpi=150)
plt.close()
print("  Saved results/lineup_impact_analysis.png")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print("\n  All baseline results:")
print(df_results[['feature_set', 'model', 'mae', 'r2', 'n_features']].to_string(index=False))

print(f"\n  Best XGBoost (tuned): MAE={best_xgb_test['mae']:.3f}, R²={best_xgb_test['r2']:.3f}")
print(f"  Best NN (tuned):      MAE={best_nn_test['mae']:.3f}, R²={best_nn_test['r2']:.3f}")
print(f"  Ensemble:             MAE={ens_m['mae']:.3f}, R²={ens_m['r2']:.3f}")

print(f"\n  ── Comparison to previous best ──")
print(f"  L1 Ridge baseline:    MAE=10.610")
print(f"  L1+L5 best ensemble:  MAE=10.480")
print(f"  L1+L5+L6 ensemble:    MAE={ens_m['mae']:.3f}")
improvement = (10.480 - ens_m['mae']) / 10.480 * 100
print(f"  Improvement over L1+L5: {improvement:+.2f}%")

print("\n  Plots saved to results/:")
print("    lineup_mae_comparison.png")
print("    lineup_mae_heatmap.png")
print("    lineup_feature_importance.png")
print("    lineup_residual_plots.png")
print("    lineup_nn_learning_curves.png")
print("    lineup_residual_distribution.png")
print("    lineup_vs_l5_comparison.png")
print("    lineup_impact_analysis.png")

print("\nDone!")
