# =============================================================================
# FILE 6: 6_baselines.py
# Dynamic Pricing MTL Project — Baseline Models Comparison
# =============================================================================
# Trains XGBoost, Random Forest, and Logistic Regression on the SAME test data.
# Compares their metrics against the MTL model from evaluation_results.json.
# Outputs: baseline_results.json  (paste into paper Table IV)
# =============================================================================

import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              r2_score, f1_score, roc_auc_score,
                              precision_score, recall_score)
from xgboost import XGBRegressor, XGBClassifier

PROCESSED_DIR = "processed"
MTL_RESULTS   = "evaluation_results.json"
OUTPUT_PATH   = "baseline_results.json"

# ── Load data ─────────────────────────────────────────────────────────────────
print("── Loading data ────────────────────────────────────────")
X_train = np.load(f"{PROCESSED_DIR}/X_train.npy")
X_test  = np.load(f"{PROCESSED_DIR}/X_test.npy")
y_train = np.load(f"{PROCESSED_DIR}/y_train.npy")
y_test  = np.load(f"{PROCESSED_DIR}/y_test.npy")

# Separate targets
y_train_vol, y_test_vol   = y_train[:,0], y_test[:,0]
y_train_tru, y_test_tru   = y_train[:,1], y_test[:,1]
y_train_col, y_test_col   = y_train[:,2].astype(int), y_test[:,2].astype(int)

print(f"   Train: {X_train.shape}  |  Test: {X_test.shape}")

# ── Helper ────────────────────────────────────────────────────────────────────
def reg_metrics(true, pred):
    return {
        "MAE":  round(mean_absolute_error(true, pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(true, pred)), 4),
        "R2":   round(r2_score(true, pred), 4),
    }

def cls_metrics(true, prob):
    pred = (prob >= 0.5).astype(int)
    return {
        "F1":        round(f1_score(true, pred, zero_division=0), 4),
        "AUC":       round(roc_auc_score(true, prob), 4),
        "Precision": round(precision_score(true, pred, zero_division=0), 4),
        "Recall":    round(recall_score(true, pred, zero_division=0), 4),
    }

results = {}

# ── Baseline 1: Random Forest ─────────────────────────────────────────────────
print("\n── Baseline 1: Random Forest ───────────────────────────")

rf_vol = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_vol.fit(X_train, y_train_vol)
pv = rf_vol.predict(X_test)
print(f"   Volatility done")

rf_tru = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_tru.fit(X_train, y_train_tru)
pt = rf_tru.predict(X_test)
print(f"   Trust done")

rf_col = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_col.fit(X_train, y_train_col)
pc = rf_col.predict_proba(X_test)[:,1]
print(f"   Collusion done")

results['RandomForest'] = {
    'volatility': reg_metrics(y_test_vol, pv),
    'trust':      reg_metrics(y_test_tru, pt),
    'collusion':  cls_metrics(y_test_col, pc),
}

# ── Baseline 2: XGBoost ───────────────────────────────────────────────────────
print("\n── Baseline 2: XGBoost ─────────────────────────────────")

xgb_vol = XGBRegressor(n_estimators=100, random_state=42,
                        verbosity=0, eval_metric='rmse')
xgb_vol.fit(X_train, y_train_vol)
pv = xgb_vol.predict(X_test)
print(f"   Volatility done")

xgb_tru = XGBRegressor(n_estimators=100, random_state=42,
                        verbosity=0, eval_metric='rmse')
xgb_tru.fit(X_train, y_train_tru)
pt = xgb_tru.predict(X_test)
print(f"   Trust done")

xgb_col = XGBClassifier(n_estimators=100, random_state=42,
                         verbosity=0, eval_metric='logloss',
                         use_label_encoder=False)
xgb_col.fit(X_train, y_train_col)
pc = xgb_col.predict_proba(X_test)[:,1]
print(f"   Collusion done")

results['XGBoost'] = {
    'volatility': reg_metrics(y_test_vol, pv),
    'trust':      reg_metrics(y_test_tru, pt),
    'collusion':  cls_metrics(y_test_col, pc),
}

# ── Baseline 3: Linear / Logistic Regression ─────────────────────────────────
print("\n── Baseline 3: Linear + Logistic Regression ────────────")

lr_vol = LinearRegression()
lr_vol.fit(X_train, y_train_vol)
pv = lr_vol.predict(X_test)
print(f"   Volatility done")

lr_tru = LinearRegression()
lr_tru.fit(X_train, y_train_tru)
pt = lr_tru.predict(X_test)
print(f"   Trust done")

lr_col = LogisticRegression(max_iter=1000, random_state=42)
lr_col.fit(X_train, y_train_col)
pc = lr_col.predict_proba(X_test)[:,1]
print(f"   Collusion done")

results['LinearLogistic'] = {
    'volatility': reg_metrics(y_test_vol, pv),
    'trust':      reg_metrics(y_test_tru, pt),
    'collusion':  cls_metrics(y_test_col, pc),
}

# ── Load MTL results and add to comparison ────────────────────────────────────
print("\n── Loading MTL results ─────────────────────────────────")
with open(MTL_RESULTS) as f:
    mtl = json.load(f)

results['MTLModel'] = {
    'volatility': mtl['volatility'],
    'trust':      mtl['trust'],
    'collusion':  mtl['collusion'],
}

# ── Print comparison table ────────────────────────────────────────────────────
print("\n╔══════════════════════════════════════════════════════════════════════╗")
print("║                    COMPARISON TABLE                                 ║")
print("╠═══════════════════╦════════════════════╦════════════════════╦═══════╣")
print("║ Model             ║ Vol MAE / R²       ║ Trust MAE / R²     ║Col F1 ║")
print("╠═══════════════════╬════════════════════╬════════════════════╬═══════╣")
for model_name, m in results.items():
    v  = m['volatility']
    t  = m['trust']
    c  = m['collusion']
    print(f"║ {model_name:<17} ║ {v['MAE']:.4f} / {v['R2']:.4f}     ║ "
          f"{t['MAE']:.4f} / {t['R2']:.4f}     ║ {c['F1']:.4f}║")
print("╚═══════════════════╩════════════════════╩════════════════════╩═══════╝")

print("\nFull AUC comparison (Collusion detection):")
for model_name, m in results.items():
    print(f"  {model_name:<20}: AUC={m['collusion']['AUC']:.4f}  "
          f"Precision={m['collusion']['Precision']:.4f}  "
          f"Recall={m['collusion']['Recall']:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅  Baseline results saved → {OUTPUT_PATH}")
print(f"\n→  Run 7_plots.py next")
