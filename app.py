#!/usr/bin/env python3
# Bench different models for: “UP/DOWN ≥ X% in next T days with ≥70% confidence”.
# Backends:
#   - quantile_lgbm: LightGBM quantile regressors per T
#   - clf_lgbm:      LightGBM classifiers per T for UP/DOWN events with probability thresholds

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import os
import joblib
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb

# =========================
# CONFIG (adjust in Colab)
# =========================
PANEL_WIDE = Path("/home/parham/myproject/panel")   # path to your parquet (folder or file)
OUT_DIR = Path("/home/parham/myproject")

TRAIN_FRACTION = 0.80
HORIZONS = [1, 2, 3, 4, 5, 6, 7]
THRESH_PER_DAY = 0.005   # 0.5% per day

CONF_UP = 0.75           # probability threshold for UP (relaxed for coverage)
CONF_DN = 0.65           # probability threshold for DOWN (relaxed for coverage)

WIN = 40                 # number of lags
USE_GPU = True           # set True if GPU-enabled libs are available

# ---------- Utility trade-off between probability (Z) and move size (X)
# Score = Z * X^gamma. Higher gamma favors larger X; lower gamma favors Z.
USE_UTILITY_FOR_COMBINE = True
UTILITY_GAMMA = 1.0

# Calibration on the head of OOS for quantile backends
CALIBRATE = True
VAL_FRAC_OOS = 0.35
# Wider quantile grids for better selection
UP_QGRID = [0.10, 0.20, 0.25, 0.30, 0.35, 0.40]
DN_QGRID = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

# ---------- Tighter coverage (fewer signals)
COVERAGE_TUNE = True
TARGET_COVERAGE = 0.20        # was 0.33
TARGET_COVERAGE_CLS = 0.15    # was 0.25
MIN_PRECISION_CLS = 0.90      # was 0.85

SAVE_MODELS = False
SAVE_TREES = False
SAVE_TREES_COUNT = 1

# Feature importance & contributions (global + SHAP-style)
COMPUTE_FEATURE_IMPORTANCE = True        # LightGBM gain/weight importances
EXPORT_FEATURE_CONTRIB = True            # SHAP-style mean |contrib| via pred_contrib
TOP_FEATURES_TO_PRINT = 20               # console summary length
COMPUTE_GROUP_IMPORTANCE = False         # grouped by family (uses gain importances)
COMPUTE_PERMUTATION_IMPORTANCE = False   # family-level permutation (heavier)

# Sanity-check toggles (disabled by default)
SANITY_RANDOMIZE_YTEST = False
SANITY_PERMUTE_XTE = False
SANITY_SEED = 42
MARGIN_EPS = 1e-9

# =====================
# Load panel + features
# =====================
EXCLUDE_FAMILIES: List[str] = []

def base_from_lag(col: str) -> str:
    if "_lag" in col:
        return col.split("_lag")[0]
    return col

def family_for(base: str) -> str:
    s = base
    if s.startswith("target_vol"): return "vol"
    if s == "target": return "target"
    if s.startswith("nav_premium") or s.startswith("nav_redeem"): return "nav"
    if s.startswith("trade_volume") or s.startswith("trade_value") or s.startswith("trade_count"): return "flows"
    if s.startswith("usd_irr"): return "fx"
    if s.startswith("global_gold"): return "global"
    if s.startswith("gold_diff") or s.startswith("diff_change") or s.startswith("refah_coin_spread"): return "basis"
    if s.startswith("gold_today") or s.startswith("gold_tmrw"): return "gold"
    if s.startswith("refah") or s.startswith("coin"): return "coin"
    return "other"

def load_panel(path: Path) -> pd.DataFrame:
    # Accept directory or file path
    p = path
    if p.is_dir():
        # find first parquet inside
        files = sorted(list(p.glob("*.parquet")))
        if not files:
            raise FileNotFoundError(f"No parquet found under {p}")
        p = files[0]
    df = pd.read_parquet(p).copy().sort_values("tehran_date").reset_index(drop=True)
    df["tehran_date"] = pd.to_datetime(df["tehran_date"])
    return df

def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    d = df.copy()
    def has(c): return c in d.columns
    feats: List[str] = []

    # target is yesterday's target_next (to build lags)
    d["target"] = d.get("target_next", pd.Series(np.nan, index=d.index)).shift(1)

    # --------------------- KEEPED FEATURES ---------------------
    if has("diff_change"): feats += ["diff_change"]

    if has("refah"):
        d["refah_ret1"] = d["refah"].pct_change(); feats += ["refah_ret1"]

    if has("coin_today"):
        d["coin_ret1"] = d["coin_today"].pct_change(); feats += ["coin_ret1"]

    if has("refah") and has("coin_today"):
        d["refah_coin_spread"] = d["refah"] - d["coin_today"]
        roll = d["refah_coin_spread"].rolling(60)
        d["refah_coin_spread_z60"] = (d["refah_coin_spread"] - roll.mean()) / (roll.std().replace(0, np.nan))
        feats += ["refah_coin_spread", "refah_coin_spread_z60"]

    if has("global_gold_usd"):
        d["global_gold_ret1"] = d["global_gold_usd"].pct_change(); feats += ["global_gold_ret1"]
        d["global_gold_ret5"] = d["global_gold_usd"].pct_change(5); feats += ["global_gold_ret5"]
        roll = d["global_gold_usd"].rolling(20)
        d["global_gold_z20"] = (d["global_gold_usd"] - roll.mean()) / (roll.std().replace(0, np.nan)); feats += ["global_gold_z20"]

    if has("usd_irr"):
        d["usd_irr_ret1"] = d["usd_irr"].pct_change(); feats += ["usd_irr_ret1"]
        d["usd_irr_ret5"] = d["usd_irr"].pct_change(5); feats += ["usd_irr_ret5"]
        roll = d["usd_irr"].rolling(20)
        d["usd_irr_z20"] = (d["usd_irr"] - roll.mean()) / (roll.std().replace(0, np.nan)); feats += ["usd_irr_z20"]

    # NAV premium features if available from panel
    if has("nav_premium"):
        d["nav_premium_ret1"] = d["nav_premium"].pct_change(); feats += ["nav_premium_ret1"]
        d["nav_premium_abs"] = d["nav_premium"].abs(); feats += ["nav_premium_abs"]
        d["nav_premium_diff5"] = d["nav_premium"].diff(5); feats += ["nav_premium_diff5"]
        roll = d["nav_premium"].rolling(60)
        d["nav_premium_z60"] = (d["nav_premium"] - roll.mean()) / (roll.std().replace(0, np.nan))
        feats += ["nav_premium_z60"]
        roll20 = d["nav_premium"].rolling(20)
        d["nav_premium_z20"] = (d["nav_premium"] - roll20.mean()) / (roll20.std().replace(0, np.nan))
        feats += ["nav_premium_z20"]

    # ETF flow metrics: keep trade_value / trade_count only (drop trade_volume)
    if has("trade_value"):
        d["trade_value_ret1"] = d["trade_value"].pct_change(); feats += ["trade_value_ret1"]
    if has("trade_count"):
        d["trade_count_ret1"] = d["trade_count"].pct_change(); feats += ["trade_count_ret1"]

    # target realized vol
    d["target_vol10"] = d["target"].rolling(10).std(); feats += ["target_vol10"]

    # --------------------- REMOVED FEATURES ---------------------
    # Removed gold_today/gold_tomorrow section:
    #   gold_today_ret1, gold_tmrw_ret1, gold_diff, gold_diff_z60
    # Removed nav_redeem_price -> nav_redeem_ret1
    # Removed trade_volume -> trade_volume_ret1, trade_volume_z20

    KEEP_BASES = [
        "target_vol10",
        "refah_coin_spread", "refah_coin_spread_z60",
        "diff_change",
        "refah_ret1", "coin_ret1",
        # Global gold / USD_IRR
        "global_gold_ret1", "global_gold_ret5", "global_gold_z20",
        "usd_irr_ret1", "usd_irr_ret5", "usd_irr_z20",
        # NAV premium
        "nav_premium_ret1", "nav_premium_abs", "nav_premium_diff5", "nav_premium_z20", "nav_premium_z60",
        # ETF flows (no trade_volume)
        "trade_value_ret1", "trade_count_ret1",
    ]
    feats = [f for f in feats if f in KEEP_BASES]

    # Apply ablation by families if requested
    if EXCLUDE_FAMILIES:
        feats = [f for f in feats if family_for(f) not in set(EXCLUDE_FAMILIES)]
    return d, feats

def make_lagged(work: pd.DataFrame, bases: List[str], win: int) -> Tuple[pd.DataFrame, List[str]]:
    out = work.copy()
    lag_frames: List[pd.DataFrame] = []
    lag_cols: List[str] = []

    for base in bases:
        s = out[base]
        cols = []
        for k in range(1, win + 1):
            ss = s.shift(k)
            ss.name = f"{base}_lag{k}"
            cols.append(ss)
            lag_cols.append(ss.name)
        if cols:
            lag_frames.append(pd.concat(cols, axis=1))

    # Target lags
    t = out["target"]
    tcols = []
    for k in range(1, win + 1):
        ts = t.shift(k)
        ts.name = f"target_lag{k}"
        tcols.append(ts)
        lag_cols.append(ts.name)
    if tcols:
        lag_frames.append(pd.concat(tcols, axis=1))

    if lag_frames:
        out = pd.concat([out] + lag_frames, axis=1)
    return out, lag_cols

def build_future_compound(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    s = np.log1p(df["target_next"].astype(float))
    sf = s.shift(-1)
    for T in horizons:
        out[f"y_T{T}"] = np.expm1(sf.rolling(T).sum())
    return out

# =============================
# Common eval and aggregation
# =============================
def decide_for_T(q_low_up, q_high_dn, T: int, thr_per_day: float) -> Tuple[np.ndarray, np.ndarray]:
    thr = thr_per_day * T
    margin_up = q_low_up - (+thr)
    margin_dn = (-thr) - q_high_dn
    side = np.where(margin_up > 0, +1, np.where(margin_dn > 0, -1, 0))
    margin = np.maximum(margin_up, margin_dn)
    return side, margin

def combine_across_T(all_T_sides: Dict[int, np.ndarray], all_T_margins: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Ts = sorted(all_T_sides.keys())
    n = len(next(iter(all_T_sides.values())))
    best_side = np.zeros(n, dtype=int)
    best_T = np.zeros(n, dtype=int)
    best_margin = np.full(n, -np.inf)
    for T in Ts:
        s = all_T_sides[T]
        m = all_T_margins[T]
        improve = m > best_margin
        best_margin[improve] = m[improve]
        best_side[improve] = s[improve]
        best_T[improve] = T
    best_side[best_margin <= 0] = 0
    best_T[best_margin <= 0] = 0
    return best_side, best_T, best_margin

def metrics(signals_side, signals_T, y_true_by_T: Dict[int, np.ndarray], thr_per_day: float) -> Dict[str, float]:
    n = len(signals_side)
    n_sig = int((signals_side != 0).sum())
    if n_sig == 0:
        return dict(coverage=0.0, precision=np.nan, recall_up=np.nan, recall_dn=np.nan)
    hits = 0
    up_hits = dn_hits = 0
    up_true = dn_true = 0
    for i in range(n):
        T = int(signals_T[i]) if signals_T[i] != 0 else 0
        if T <= 0:
            continue
        thr = thr_per_day * T
        yT = y_true_by_T[T][i]
        s = int(signals_side[i])
        if yT >= +thr: up_true += 1
        if yT <= -thr: dn_true += 1
        if s == +1 and yT >= +thr: hits += 1; up_hits += 1
        elif s == -1 and yT <= -thr: hits += 1; dn_hits += 1
    precision = hits / n_sig if n_sig else np.nan
    recall_up = (up_hits / up_true) if up_true else np.nan
    recall_dn = (dn_hits / dn_true) if dn_true else np.nan
    return dict(coverage=n_sig / n, precision=precision, recall_up=recall_up, recall_dn=recall_dn)

# ======================
# Backend: Quantile LGBM
# ======================
def lgbm_quantile(alpha: float) -> lgb.LGBMRegressor:
    params = dict(objective="quantile", alpha=alpha, n_estimators=600,
                  learning_rate=0.06, num_leaves=63, max_depth=-1,
                  subsample=0.8, colsample_bytree=0.7, min_child_samples=40,
                  random_state=42, n_jobs=-1, verbosity=-1)
    if USE_GPU:
        params.update(dict(device="gpu", gpu_platform_id=0, gpu_device_id=0, max_bin=255))
    return lgb.LGBMRegressor(**params)

def _qkey(q: float) -> str:
    return f"{q:.2f}"

def _pred_for_q(pred_map: Dict[str, np.ndarray], q: float) -> np.ndarray:
    k = _qkey(q)
    if k in pred_map:
        return pred_map[k]
    keys = list(pred_map.keys())
    q_vals = [float(x) for x in keys]
    idx = int(np.argmin(np.abs(np.array(q_vals) - q)))
    return pred_map[keys[idx]]

def backend_quantile_lgbm(Xtr, ytr, Xte, yte, T: int, feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    up_list = list(set(UP_QGRID + [1.0 - CONF_UP]))
    dn_list = list(set(DN_QGRID + [CONF_DN]))

    models_up: Dict[float, lgb.LGBMRegressor] = {}
    models_dn: Dict[float, lgb.LGBMRegressor] = {}

    ntr = len(ytr)
    nval = max(50, int(0.15 * ntr))
    Xfit, yfit = Xtr.iloc[:-nval], ytr[:-nval]
    Xval, yval = Xtr.iloc[-nval:], ytr[-nval:]

    for q in sorted(up_list):
        m = lgbm_quantile(q)
        try:
            m.fit(Xfit, yfit, eval_set=[(Xval, yval)],
                  eval_metric="quantile",
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        except Exception:
            m.fit(Xtr, ytr)
        models_up[q] = m

    for q in sorted(dn_list):
        m = lgbm_quantile(q)
        try:
            m.fit(Xfit, yfit, eval_set=[(Xval, yval)],
                  eval_metric="quantile",
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        except Exception:
            m.fit(Xtr, ytr)
        models_dn[q] = m

    preds_up = {_qkey(q): m.predict(Xte) for q, m in models_up.items()}
    preds_dn = {_qkey(q): m.predict(Xte) for q, m in models_dn.items()}

    n_test = len(yte)
    n_val = max(20, int(VAL_FRAC_OOS * n_test)) if CALIBRATE else 0
    thr = THRESH_PER_DAY * T
    sel_up, sel_dn = round(1.0 - CONF_UP, 2), round(CONF_DN, 2)
    best_up = (0.0, sel_up); best_dn = (0.0, sel_dn)

    if CALIBRATE and n_val > 0:
        for q in sorted(up_list):
            pred = _pred_for_q(preds_up, q)[:n_val]
            side = pred > (+thr)
            tp = int(((yte[:n_val] >= +thr) & side).sum())
            fp = int(((yte[:n_val] <  +thr) & side).sum())
            fn = int(((yte[:n_val] >= +thr) & (~side)).sum())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec  = tp / (tp + fn) if (tp + fn) else 0.0
            f1   = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
            if f1 > best_up[0]: best_up = (f1, q)

        for q in sorted(dn_list):
            pred = _pred_for_q(preds_dn, q)[:n_val]
            side = pred < (-thr)
            tp = int(((yte[:n_val] <= -thr) & side).sum())
            fp = int(((yte[:n_val] >  -thr) & side).sum())
            fn = int(((yte[:n_val] <= -thr) & (~side)).sum())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec  = tp / (tp + fn) if (tp + fn) else 0.0
            f1   = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
            if f1 > best_dn[0]: best_dn = (f1, q)

    sel_up, sel_dn = best_up[1], best_dn[1]

    q_low_up  = _pred_for_q(preds_up, sel_up)
    q_high_dn = _pred_for_q(preds_dn, sel_dn)
    side, margin = decide_for_T(q_low_up, q_high_dn, T, THRESH_PER_DAY)

    info: Dict[str, float] = {"q_up_sel": float(sel_up), "q_dn_sel": float(sel_dn)}

    # Feature importances & contributions for selected quantile models
    if COMPUTE_FEATURE_IMPORTANCE and feature_names is not None:
        try:
            m_up = models_up[sel_up]
            m_dn = models_dn[sel_dn]
            fi_up = dict(zip(feature_names, m_up.feature_importances_.tolist()))
            fi_dn = dict(zip(feature_names, m_dn.feature_importances_.tolist()))
            info.update({"fi_up": fi_up, "fi_dn": fi_dn})
            if EXPORT_FEATURE_CONTRIB:
                try:
                    contrib_up = m_up.predict(Xte, pred_contrib=True)
                    contrib_dn = m_dn.predict(Xte, pred_contrib=True)
                    contrib_up = np.abs(contrib_up[:, :-1])
                    contrib_dn = np.abs(contrib_dn[:, :-1])
                    mean_up = contrib_up.mean(axis=0)
                    mean_dn = contrib_dn.mean(axis=0)
                    info.update({
                        "fc_up": dict(zip(feature_names, mean_up.tolist())),
                        "fc_dn": dict(zip(feature_names, mean_dn.tolist())),
                    })
                except Exception:
                    pass
        except Exception:
            pass

    if SAVE_MODELS:
        (OUT_DIR / "models" / "quantile_lgbm").mkdir(parents=True, exist_ok=True)
        try:
            models_up[sel_up].booster_.save_model(str(OUT_DIR / "models" / "quantile_lgbm" / f"up_T{T}_q{sel_up:.2f}.txt"))
            models_dn[sel_dn].booster_.save_model(str(OUT_DIR / "models" / "quantile_lgbm" / f"dn_T{T}_q{sel_dn:.2f}.txt"))
        except Exception:
            pass
        try:
            meta = dict(T=T, backend="quantile_lgbm", q_up=float(sel_up), q_dn=float(sel_dn), features=feature_names or [])
            with open(OUT_DIR / "models" / "quantile_lgbm" / f"meta_T{T}.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    return side, margin, info

# =========================
# Backend: LGBM Classifiers
# =========================
def lgbm_classifier() -> lgb.LGBMClassifier:
    params = dict(n_estimators=600, learning_rate=0.06, num_leaves=63, max_depth=-1,
                  subsample=0.8, colsample_bytree=0.7, min_child_samples=40,
                  random_state=42, n_jobs=-1, class_weight="balanced", verbosity=-1)
    if USE_GPU:
        params.update(dict(device="gpu", gpu_platform_id=0, gpu_device_id=0, max_bin=255))
    return lgb.LGBMClassifier(**params)

def backend_clf_lgbm(Xtr, ytr, Xte, yte, T: int, feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    thr_move = THRESH_PER_DAY * T
    y_up = (ytr >= +thr_move).astype(int)
    y_dn = (ytr <= -thr_move).astype(int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf_up = CalibratedClassifierCV(lgbm_classifier(), method="isotonic", cv=3)
        clf_dn = CalibratedClassifierCV(lgbm_classifier(), method="isotonic", cv=3)
        clf_up.fit(Xtr, y_up)
        clf_dn.fit(Xtr, y_dn)

    proba_up = clf_up.predict_proba(Xte)[:, 1]
    proba_dn = clf_dn.predict_proba(Xte)[:, 1]

    n_test = len(yte)
    n_val = max(20, int(VAL_FRAC_OOS * n_test)) if CALIBRATE else 0
    sel_up, sel_dn = CONF_UP, CONF_DN

    if n_val > 0:
        pu_v = proba_up[:n_val]
        pd_v = proba_dn[:n_val]
        yu_v = (yte[:n_val] >= +thr_move).astype(int)
        yd_v = (yte[:n_val] <= -thr_move).astype(int)

        cov_up = max(0.0, min(1.0, TARGET_COVERAGE_CLS / 2.0))
        cov_dn = max(0.0, min(1.0, TARGET_COVERAGE_CLS / 2.0))

        def cov_thr(arr, cov):
            if len(arr) == 0 or cov <= 0:
                return 1.0
            q = 1.0 - cov
            return float(np.quantile(arr, q))

        def prec_thr(arr, ytrue_bin, prec_floor):
            if len(arr) == 0 or np.unique(ytrue_bin).size < 2:
                return None
            qs = np.linspace(0.50, 0.95, 10)
            cands = sorted({float(np.quantile(arr, 1.0 - q)) for q in qs})
            best = None
            for t in cands:
                pred = arr >= t
                tp = int(((ytrue_bin == 1) & pred).sum()); fp = int(((ytrue_bin == 0) & pred).sum())
                prec = tp / (tp + fp) if (tp + fp) else 0.0
                cov  = pred.mean()
                if prec >= prec_floor:
                    if best is None or cov > best[1]:
                        best = (t, cov)
            return best[0] if best is not None else None

        t_up = prec_thr(pu_v, yu_v, MIN_PRECISION_CLS)
        t_dn = prec_thr(pd_v, yd_v, MIN_PRECISION_CLS)
        if t_up is None: t_up = cov_thr(pu_v, cov_up)
        if t_dn is None: t_dn = cov_thr(pd_v, cov_dn)
        sel_up = float(np.clip(t_up, 0.50, 0.99))
        sel_dn = float(np.clip(t_dn, 0.50, 0.99))

    # Utility-based scoring (Z × X^gamma)
    if USE_UTILITY_FOR_COMBINE:
        pu_eff = np.where(proba_up >= sel_up, proba_up, 0.0)
        pd_eff = np.where(proba_dn >= sel_dn, proba_dn, 0.0)
        score_up = pu_eff * (thr_move ** UTILITY_GAMMA)
        score_dn = pd_eff * (thr_move ** UTILITY_GAMMA)
        score = np.maximum(score_up, score_dn)
        side = np.where(score_up >= score_dn, +1, -1)
        side = np.where(score > 0, side, 0)
        margin = score  # unnormalized; run_backend divides by thr per T
    else:
        margin_up = proba_up - sel_up
        margin_dn = proba_dn - sel_dn
        side = np.where(margin_up > 0, +1, np.where(margin_dn > 0, -1, 0))
        margin = np.maximum(margin_up, margin_dn)

    info: Dict[str, float] = {
        "thr_up": float(sel_up), "thr_dn": float(sel_dn),
        "use_utility": float(1.0 if USE_UTILITY_FOR_COMBINE else 0.0),
        "utility_gamma": float(UTILITY_GAMMA),
    }

    if COMPUTE_FEATURE_IMPORTANCE and feature_names is not None:
        try:
            base_up = lgbm_classifier().fit(Xtr, y_up)
            base_dn = lgbm_classifier().fit(Xtr, y_dn)
            fi_up = dict(zip(feature_names, base_up.feature_importances_.tolist()))
            fi_dn = dict(zip(feature_names, base_dn.feature_importances_.tolist()))
            info.update({"fi_up": fi_up, "fi_dn": fi_dn})
            if EXPORT_FEATURE_CONTRIB:
                try:
                    cu = base_up.predict(Xte, pred_contrib=True)
                    cd = base_dn.predict(Xte, pred_contrib=True)
                    cu = np.abs(cu[:, :-1]); cd = np.abs(cd[:, :-1])
                    info.update({
                        "fc_up": dict(zip(feature_names, cu.mean(axis=0).tolist())),
                        "fc_dn": dict(zip(feature_names, cd.mean(axis=0).tolist())),
                    })
                except Exception:
                    pass
        except Exception:
            pass

    if SAVE_MODELS:
        (OUT_DIR / "models" / "clf_lgbm").mkdir(parents=True, exist_ok=True)
        try:
            joblib.dump(clf_up, OUT_DIR / "models" / "clf_lgbm" / f"up_T{T}.joblib")
            joblib.dump(clf_dn, OUT_DIR / "models" / "clf_lgbm" / f"dn_T{T}.joblib")
            meta = dict(T=T, backend="clf_lgbm", thr_up=float(sel_up), thr_dn=float(sel_dn), features=feature_names or [])
            with open(OUT_DIR / "models" / "clf_lgbm" / f"meta_T{T}.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return side, margin, info

# ======
# Main
# ======
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude-families", type=str, default="", help="comma-separated families to exclude (e.g., nav,flows,fx,basis,global,gold,coin,vol,target)")
    ap.add_argument("--sanity-randomize-ytest", action="store_true", help="shuffle test labels to verify collapse of metrics")
    ap.add_argument("--sanity-permute-xte", action="store_true", help="permute test features to break alignment")
    ap.add_argument("--wf-enable", action="store_true", help="enable walk-forward evaluation")
    ap.add_argument("--wf-train-frac", type=float, default=None)
    ap.add_argument("--wf-val-frac", type=float, default=None)
    ap.add_argument("--wf-test-frac", type=float, default=None)
    args, _ = ap.parse_known_args()

    global EXCLUDE_FAMILIES
    if args.exclude_families:
        EXCLUDE_FAMILIES = [x.strip() for x in args.exclude_families.split(",") if x.strip()]
    global SANITY_RANDOMIZE_YTEST, SANITY_PERMUTE_XTE
    SANITY_RANDOMIZE_YTEST = bool(args.sanity_randomize_ytest)
    SANITY_PERMUTE_XTE = bool(args.sanity_permute_xte)

    print("Loading panel:", PANEL_WIDE)
    df = load_panel(PANEL_WIDE)
    work, bases = build_features(df)

    work_lag, lag_cols = make_lagged(work, bases, WIN)
    Y = build_future_compound(df, HORIZONS)
    work_lag = pd.concat([work_lag, Y], axis=1)

    supervised = work_lag.dropna(subset=[f"y_T{T}" for T in HORIZONS]).reset_index(drop=True)

    split_idx = int(TRAIN_FRACTION * len(supervised))
    Xdf = supervised[lag_cols].astype("float32")
    Xdf = Xdf.replace([np.inf, -np.inf], np.nan)

    Xtr = Xdf.iloc[:split_idx, :].copy()
    Xte = Xdf.iloc[split_idx:, :].copy()

    med = Xtr.median()
    Xtr = Xtr.fillna(med)
    Xte = Xte.fillna(med)

    dates = pd.to_datetime(supervised["tehran_date"]).to_numpy()
    dates_te = dates[split_idx:]

    y_true_by_T: Dict[int, np.ndarray] = {}
    for T in HORIZONS:
        y = supervised[f"y_T{T}"].astype("float32").values
        y_true_by_T[T] = y[split_idx:]

    if SANITY_RANDOMIZE_YTEST:
        rng = np.random.default_rng(SANITY_SEED)
        for T in HORIZONS:
            if len(y_true_by_T[T]) > 1:
                perm = rng.permutation(len(y_true_by_T[T]))
                y_true_by_T[T] = y_true_by_T[T][perm]
        print("[SANITY] Randomized y_test labels for all horizons.")

    WF_EVAL_TAIL_SIZE: Optional[int] = None

    def run_backend(name: str, write_signals: bool = True, Xte_override: Optional[pd.DataFrame] = None, out_suffix: str = ""):
        print(f"\n=== Backend: {name} ===")
        all_T_sides: Dict[int, np.ndarray] = {}
        all_T_margins: Dict[int, np.ndarray] = {}
        extra_info: Dict[int, Dict[str, float]] = {}

        for T in HORIZONS:
            y = supervised[f"y_T{T}"].astype("float32").values
            ytr, yte = y[:split_idx], y[split_idx:]
            Xte_use = Xte_override if Xte_override is not None else Xte
            if name == "quantile_lgbm":
                side, margin, info = backend_quantile_lgbm(Xtr, ytr, Xte_use, yte, T, feature_names=lag_cols)
                scale = max(1e-9, THRESH_PER_DAY * T)
                margin = margin / scale
            elif name == "clf_lgbm":
                side, margin, info = backend_clf_lgbm(Xtr, ytr, Xte_use, yte, T, feature_names=lag_cols)
                scale = max(1e-9, THRESH_PER_DAY * T)
                margin = margin / scale
            else:
                raise ValueError(name)
            all_T_sides[T] = side
            all_T_margins[T] = margin
            extra_info[T] = info

        best_side, best_T, best_margin = combine_across_T(all_T_sides, all_T_margins)

        # Tune margin thresholds on the validation head of OOS to approach target coverage
        margin_thr = 0.0
        margin_thr_up = 0.0
        margin_thr_dn = 0.0
        if COVERAGE_TUNE:
            n_test = len(best_side)
            n_val = max(20, int(VAL_FRAC_OOS * n_test))
            sides_v = best_side[:n_val]
            margins_v = best_margin[:n_val]
            want = int(round(TARGET_COVERAGE * n_val))
            pos_margins = margins_v[sides_v != 0]
            ups = margins_v[sides_v == +1]
            dns = margins_v[sides_v == -1]
            w_up = len(ups); w_dn = len(dns); w_tot = max(1, w_up + w_dn)
            want_up = int(round(want * (w_up / w_tot)))
            want_dn = max(0, want - want_up)

            def thr_for(arr, want_n):
                if want_n <= 0 or len(arr) == 0:
                    return 0.0
                if want_n >= len(arr):
                    return 0.0
                sm = np.sort(arr)[::-1]
                idx = max(0, min(len(sm)-1, want_n-1))
                return float(sm[idx] + MARGIN_EPS)

            margin_thr_up = thr_for(ups, want_up)
            margin_thr_dn = thr_for(dns, want_dn)

            if want > 0 and len(pos_margins) > 0:
                if want < len(pos_margins):
                    sm = np.sort(pos_margins)[::-1]
                    idx = max(0, min(len(sm)-1, want-1))
                    margin_thr = float(sm[idx] + MARGIN_EPS)
                else:
                    margin_thr = 0.0

            keep_up = (best_side == +1) & (best_margin >= margin_thr_up)
            keep_dn = (best_side == -1) & (best_margin >= margin_thr_dn)
            keep = keep_up | keep_dn
            best_side = np.where(keep, best_side, 0)
            best_T = np.where(keep, best_T, 0)

        # Evaluate on tail of OOS (exclude head used for tuning)
        n_test_tot = len(best_side)
        if WF_EVAL_TAIL_SIZE is not None and WF_EVAL_TAIL_SIZE > 0 and WF_EVAL_TAIL_SIZE <= n_test_tot:
            eval_start = n_test_tot - WF_EVAL_TAIL_SIZE
        else:
            eval_start = max(20, int(VAL_FRAC_OOS * n_test_tot)) if CALIBRATE else 0
            if eval_start >= n_test_tot:
                eval_start = 0

        eval_side = best_side[eval_start:]
        eval_T = best_T[eval_start:]
        y_true_eval = {T: arr[eval_start:] for T, arr in y_true_by_T.items()}
        dates_eval = dates_te[eval_start:]

        m = metrics(eval_side, eval_T, y_true_eval, THRESH_PER_DAY)
        print(f"Coverage: {m['coverage']:.2%} | Precision: {m['precision']:.3f} | RecallUP: {m['recall_up']:.3f} | RecallDN: {m['recall_dn']:.3f}")
        print(f"Breakdown: UP={(eval_side==1).sum()} | DOWN={(eval_side==-1).sum()} | NONE={(eval_side==0).sum()}")

        if write_signals:
            rows = []
            n_eval = len(eval_side)
            for i in range(n_eval):
                T = int(eval_T[i]); side = int(eval_side[i])
                row = {"date": dates_eval[i], "signal_T": T, "signal_side": ("UP" if side==1 else ("DOWN" if side==-1 else "NONE"))}
                if T > 0:
                    thr = THRESH_PER_DAY * T
                    row["threshold"] = thr
                    row["margin"] = float(best_margin[eval_start + i])
                    if COVERAGE_TUNE:
                        row["margin_thr"] = float(margin_thr)
                    ytrue = y_true_eval[T][i]
                    row["y_true"] = float(ytrue)
                    if side == 1:
                        row["met"] = bool(ytrue >= +thr)
                    elif side == -1:
                        row["met"] = bool(ytrue <= -thr)
                    else:
                        row["met"] = False
                    info = extra_info.get(T, {})
                    for k, v in info.items():
                        row[k] = v
                rows.append(row)
            out = pd.DataFrame(rows)
            try:
                if not out.empty:
                    out['date'] = pd.to_datetime(out['date'])
                    is_sig = (out['signal_side'] != 'NONE').astype(int)
                    abs_margin = out.get('margin', pd.Series([0]*len(out))).abs().fillna(0)
                    out = out.assign(__is_sig=is_sig, __abs_margin=abs_margin)
                    out = out.sort_values(['date','__is_sig','__abs_margin'], ascending=[True, False, False])
                    out = out.drop_duplicates(subset=['date'], keep='first')
                    out = out.drop(columns=['__is_sig','__abs_margin'])
            except Exception as e:
                print(f"Could not deduplicate by date: {e}")
            out_path = OUT_DIR / f"signals_{name}{out_suffix}.csv"
            out.to_csv(out_path, index=False)
            print(f"Saved signals (eval tail) to {out_path} (rows={len(out)})")
        return m, extra_info

    results: Dict[str, Optional[Dict[str, float]]] = {}
    backend_infos: Dict[str, Dict[int, Dict[str, float]]] = {}

    wf_enable = False
    wf_train_frac = 0.70
    wf_val_frac = 0.10
    wf_test_frac = 0.20
    try:
        wf_enable = bool(os.getenv("WF_ENABLE", "false").lower() in ("1","true","yes"))
    except Exception:
        wf_enable = False

    if getattr(args, 'wf_enable', False):
        wf_enable = True
    if getattr(args, 'wf_train_frac', None) is not None:
        wf_train_frac = float(args.wf_train_frac)
    if getattr(args, 'wf_val_frac', None) is not None:
        wf_val_frac = float(args.wf_val_frac)
    if getattr(args, 'wf_test_frac', None) is not None:
        wf_test_frac = float(args.wf_test_frac)

    if wf_enable:
        n_total = len(supervised)
        min_train = max(50, int(wf_train_frac * n_total))
        n_val     = max(10, int(wf_val_frac * n_total))
        n_test    = max(20, int(wf_test_frac * n_total))
        if (min_train + n_val + n_test) > n_total:
            n_test = max(20, n_total - (min_train + n_val))
        fold = 0
        agg_rows = []
        start_oos = min_train
        while (start_oos + n_val + n_test) <= n_total:
            end_oos = start_oos + n_val + n_test
            X_all = supervised[lag_cols].astype("float32").replace([np.inf, -np.inf], np.nan)
            Xtr_f = X_all.iloc[:start_oos, :].copy()
            Xte_f = X_all.iloc[start_oos:end_oos, :].copy()
            med2 = Xtr_f.median()
            Xtr_f = Xtr_f.fillna(med2)
            Xte_f = Xte_f.fillna(med2)
            dates_f = pd.to_datetime(supervised["tehran_date"]).to_numpy()
            dates_te_f = dates_f[start_oos:end_oos]
            y_true_by_T_f: Dict[int, np.ndarray] = {}
            for T in HORIZONS:
                yv = supervised[f"y_T{T}"].astype("float32").values
                y_true_by_T_f[T] = yv[start_oos:end_oos]
            Xtr, Xte, dates_te = Xtr_f, Xte_f, dates_te_f
            y_true_by_T = y_true_by_T_f
            split_idx = len(Xtr_f)
            Xte_override = None
            if SANITY_PERMUTE_XTE:
                Xte_override = Xte.sample(frac=1.0, random_state=SANITY_SEED).reset_index(drop=True)
                print(f"[SANITY] Permuted Xte rows for fold {fold}.")
            WF_EVAL_TAIL_SIZE = n_test
            for backend in ["quantile_lgbm", "clf_lgbm"]:
                try:
                    m, info_map = run_backend(backend, write_signals=True, Xte_override=Xte_override, out_suffix=f"_fold{fold}")
                    agg_rows.append({
                        "fold": fold, "backend": backend,
                        "coverage": m.get("coverage", float('nan')),
                        "precision": m.get("precision", float('nan')),
                        "recall_up": m.get("recall_up", float('nan')),
                        "recall_dn": m.get("recall_dn", float('nan')),
                    })
                    backend_infos[backend] = info_map
                except Exception as e:
                    print(f"Backend {backend} failed on fold {fold}: {e}")
            fold += 1
            start_oos += n_test
        if agg_rows:
            dfw = pd.DataFrame(agg_rows)
            summ = dfw.groupby("backend").agg(
                coverage=("coverage","mean"), precision=("precision","mean"),
                recall_up=("recall_up","mean"), recall_dn=("recall_dn","mean")
            ).reset_index()
            print("\n=== Walk-Forward Summary (mean across folds) ===")
            print(summ.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        return
    else:
        for backend in ["quantile_lgbm", "clf_lgbm"]:
            try:
                Xte_override = None
                if SANITY_PERMUTE_XTE:
                    Xte_override = Xte.sample(frac=1.0, random_state=SANITY_SEED).reset_index(drop=True)
                    print(f"[SANITY] Permuted Xte rows for backend {backend}.")
                m, info_map = run_backend(backend, write_signals=True, Xte_override=Xte_override)
                results[backend] = m
                backend_infos[backend] = info_map
            except Exception as e:
                print(f"Backend {backend} failed: {e}")
                results[backend] = None

    rows = []
    for name, m in results.items():
        if m is None: continue
        rows.append({
            "backend": name,
            "coverage": m["coverage"],
            "precision": m["precision"],
            "recall_up": m["recall_up"],
            "recall_dn": m["recall_dn"],
        })
    if rows:
        summ = pd.DataFrame(rows).sort_values(["precision", "coverage"], ascending=[False, False])
        print("\n=== Summary ===")
        print(summ.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

        if COMPUTE_FEATURE_IMPORTANCE or COMPUTE_GROUP_IMPORTANCE or EXPORT_FEATURE_CONTRIB:
            try:
                for name, info_map in backend_infos.items():
                    agg_gain: Dict[str, float] = {}
                    agg_contrib: Dict[str, float] = {}
                    for T, info in info_map.items():
                        for key in ("fi_up", "fi_dn"):
                            fi = info.get(key)
                            if isinstance(fi, dict):
                                for k, v in fi.items():
                                    agg_gain[k] = agg_gain.get(k, 0.0) + float(v)
                        for key in ("fc_up", "fc_dn"):
                            fc = info.get(key)
                            if isinstance(fc, dict):
                                for k, v in fc.items():
                                    agg_contrib[k] = agg_contrib.get(k, 0.0) + float(v)

                    if COMPUTE_FEATURE_IMPORTANCE and agg_gain:
                        fi_df = pd.DataFrame(sorted(agg_gain.items(), key=lambda x: -x[1]), columns=["feature","importance"])
                        outp = OUT_DIR / f"feature_importance_gain_{name}.csv"
                        fi_df.to_csv(outp, index=False)
                        print(f"Saved gain/weight feature importances to {outp}")
                        print(f"\nTop {TOP_FEATURES_TO_PRINT} features by gain ({name}):")
                        print(fi_df.head(TOP_FEATURES_TO_PRINT).to_string(index=False))

                    if EXPORT_FEATURE_CONTRIB and agg_contrib:
                        fc_df = pd.DataFrame(sorted(agg_contrib.items(), key=lambda x: -x[1]), columns=["feature","mean_abs_contrib"])
                        outp2 = OUT_DIR / f"feature_importance_contrib_{name}.csv"
                        fc_df.to_csv(outp2, index=False)
                        print(f"Saved SHAP-style contributions to {outp2}")
                        print(f"\nTop {TOP_FEATURES_TO_PRINT} features by |contrib| ({name}):")
                        print(fc_df.head(TOP_FEATURES_TO_PRINT).to_string(index=False))

                    if COMPUTE_GROUP_IMPORTANCE and agg_gain:
                        fi_df_full = pd.DataFrame(sorted(agg_gain.items(), key=lambda x: -x[1]), columns=["feature","importance"])
                        fi_df_full["base"] = fi_df_full["feature"].map(base_from_lag)
                        fi_df_full["family"] = fi_df_full["base"].map(family_for)
                        fam = fi_df_full.groupby("family")["importance"].sum().sort_values(ascending=False).reset_index()
                        outp3 = OUT_DIR / f"feature_importance_groups_{name}.csv"
                        fam.to_csv(outp3, index=False)
                        print(f"Saved grouped (family) importances to {outp3}")
            except Exception as e:
                print(f"Could not save/print feature importances: {e}")

        try:
            p_q = OUT_DIR / "signals_quantile_lgbm.csv"
            p_c = OUT_DIR / "signals_clf_lgbm.csv"
            if p_q.exists() and p_c.exists():
                dq = pd.read_csv(p_q)
                dc = pd.read_csv(p_c)
                merged = dq.merge(dc, on="date", suffixes=("_q", "_c"))
                agree = merged[(merged["signal_side_q"] != "NONE") & (merged["signal_side_q"] == merged["signal_side_c"])]
                if not agree.empty:
                    out_cols = [
                        "date",
                        "signal_side_q",
                        "signal_T_q", "signal_T_c",
                        "margin_q", "margin_c",
                        "y_true_q", "y_true_c",
                        "met_q", "met_c",
                    ]
                    agree[out_cols].rename(columns={"signal_side_q": "signal_side"}).to_csv(OUT_DIR / "signals_agree.csv", index=False)
                    print(f"Saved agreement-only signals to {OUT_DIR / 'signals_agree.csv'} (rows={len(agree)})")
                    n_test = len(dq)
                    cov_agree = len(agree) / n_test if n_test else 0.0
                    prec_agree = float(((agree["met_q"] == True) & (agree["met_c"] == True)).mean())
                    print(f"Agreement coverage: {cov_agree:.2%} | Agreement precision: {prec_agree:.3f}")
                    try:
                        mq75 = float(np.nanpercentile(agree["margin_q"].to_numpy(), 75))
                        mc75 = float(np.nanpercentile(agree["margin_c"].to_numpy(), 75))
                        strong = agree[(agree["margin_q"] >= mq75) & (agree["margin_c"] >= mc75)]
                        if not strong.empty:
                            strong[out_cols].rename(columns={"signal_side_q": "signal_side"}).to_csv(OUT_DIR / "signals_agree_strong.csv", index=False)
                            cov_str = len(strong) / n_test if n_test else 0.0
                            prec_str = float(((strong["met_q"] == True) & (strong["met_c"] == True)).mean())
                            print(f"Saved strong-agreement signals to {OUT_DIR / 'signals_agree_strong.csv'} (rows={len(strong)})")
                            print(f"Strong agreement coverage: {cov_str:.2%} | precision: {prec_str:.3f}")
                    except Exception as e:
                        print(f"Could not compute strong agreement: {e}")
        except Exception as e:
            print(f"Could not create agreement CSV: {e}")

        if COMPUTE_PERMUTATION_IMPORTANCE:
            try:
                fam_map: Dict[str, List[int]] = {}
                for idx, col in enumerate(lag_cols):
                    fam = family_for(base_from_lag(col))
                    fam_map.setdefault(fam, []).append(idx)
                rng = np.random.default_rng(42)
                for name in ["quantile_lgbm", "clf_lgbm"]:
                    if name not in results or results[name] is None:
                        continue
                    base_prec = float(results[name]["precision"]) if results[name]["precision"] is not None else np.nan
                    rows_pi = []
                    for fam, idxs in fam_map.items():
                        Xperm = Xte.copy()
                        for j in idxs:
                            vals = Xperm.iloc[:, j].to_numpy(copy=True)
                            rng.shuffle(vals)
                            Xperm.iloc[:, j] = vals
                        m_perm, _ = run_backend(name, write_signals=False, Xte_override=Xperm)
                        rows_pi.append({
                            "family": fam,
                            "precision": m_perm.get("precision", np.nan),
                            "coverage": m_perm.get("coverage", np.nan),
                            "delta_precision": base_prec - m_perm.get("precision", np.nan),
                        })
                    if rows_pi:
                        dfp = pd.DataFrame(rows_pi).sort_values("delta_precision", ascending=False)
                        outp = OUT_DIR / f"permutation_importance_{name}.csv"
                        dfp.to_csv(outp, index=False)
                        print(f"Saved permutation importance to {outp}")
            except Exception as e:
                print(f"Could not compute permutation importance: {e}")
    else:
        print("No successful backends to summarize.")

if __name__ == "__main__":
    main()
