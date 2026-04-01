# =============================================================================
# FILE 5: 5_evaluate.py
# Dynamic Pricing MTL Project — Model Evaluation
# =============================================================================
# Loads best_model.pt, runs on test set, computes all metrics for the paper.
# Outputs: evaluation_results.json  (paste numbers directly into paper)
# =============================================================================

import numpy as np
import torch
import json, importlib.util
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              r2_score, f1_score, roc_auc_score,
                              precision_score, recall_score,
                              classification_report, confusion_matrix)

# ── Import model ──────────────────────────────────────────────────────────────
def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mtl             = load_module("mtl.py", "mtl_model")
MTLPricingModel = mtl.MTLPricingModel

PROCESSED_DIR   = "processed"
MODEL_PATH      = "best_model.pt"
RESULTS_PATH    = "evaluation_results.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load test data ────────────────────────────────────────────────────────────
print("── Loading test data ───────────────────────────────────")
X_test = torch.tensor(np.load(f"{PROCESSED_DIR}/X_test.npy"), dtype=torch.float32)
y_test = np.load(f"{PROCESSED_DIR}/y_test.npy")
print(f"   Test set: {X_test.shape[0]} rows")

# ── Load model ────────────────────────────────────────────────────────────────
print("── Loading model ───────────────────────────────────────")
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model      = MTLPricingModel(input_dim=checkpoint['input_dim']).to(device)
model.load_state_dict(checkpoint['model_state'])
model.eval()
print(f"   Loaded from epoch {checkpoint['epoch']}  (val_loss={checkpoint['val_loss']:.5f})")

# ── Get predictions ───────────────────────────────────────────────────────────
print("── Running predictions ─────────────────────────────────")
with torch.no_grad():
    pv, pt, pc = model(X_test.to(device))

pred_vol      = pv.cpu().numpy().flatten()
pred_trust    = pt.cpu().numpy().flatten()
pred_col_prob = pc.cpu().numpy().flatten()
pred_col_bin  = (pred_col_prob >= 0.5).astype(int)

true_vol   = y_test[:, 0]
true_trust = y_test[:, 1]
true_col   = y_test[:, 2].astype(int)

# ── Compute metrics ───────────────────────────────────────────────────────────
print("── Computing metrics ───────────────────────────────────\n")

# Head 1: Volatility Score
vol_mae  = mean_absolute_error(true_vol, pred_vol)
vol_rmse = np.sqrt(mean_squared_error(true_vol, pred_vol))
vol_r2   = r2_score(true_vol, pred_vol)

# Head 2: Trust Score
tru_mae  = mean_absolute_error(true_trust, pred_trust)
tru_rmse = np.sqrt(mean_squared_error(true_trust, pred_trust))
tru_r2   = r2_score(true_trust, pred_trust)

# Head 3: Collusion Flag
col_f1   = f1_score(true_col, pred_col_bin, zero_division=0)
col_auc  = roc_auc_score(true_col, pred_col_prob)
col_prec = precision_score(true_col, pred_col_bin, zero_division=0)
col_rec  = recall_score(true_col, pred_col_bin, zero_division=0)

# ── Print results table ───────────────────────────────────────────────────────
print("╔══════════════════════════════════════════════════════╗")
print("║          MTL MODEL — TEST SET RESULTS               ║")
print("╠══════════════════════════════════════════════════════╣")
print(f"║  HEAD 1 — Volatility Score (regression)             ║")
print(f"║    MAE  : {vol_mae:.4f}                                  ║")
print(f"║    RMSE : {vol_rmse:.4f}                                  ║")
print(f"║    R²   : {vol_r2:.4f}                                  ║")
print("╠══════════════════════════════════════════════════════╣")
print(f"║  HEAD 2 — Trust Score (regression)                  ║")
print(f"║    MAE  : {tru_mae:.4f}                                  ║")
print(f"║    RMSE : {tru_rmse:.4f}                                  ║")
print(f"║    R²   : {tru_r2:.4f}                                  ║")
print("╠══════════════════════════════════════════════════════╣")
print(f"║  HEAD 3 — Collusion Flag (classification)           ║")
print(f"║    F1        : {col_f1:.4f}                              ║")
print(f"║    AUC-ROC   : {col_auc:.4f}                              ║")
print(f"║    Precision : {col_prec:.4f}                              ║")
print(f"║    Recall    : {col_rec:.4f}                              ║")
print("╚══════════════════════════════════════════════════════╝")

print("\n── Collusion Classification Report ─────────────────────")
print(classification_report(true_col, pred_col_bin,
                              target_names=['No Collusion','Collusion']))

print("── Confusion Matrix (Collusion) ────────────────────────")
cm = confusion_matrix(true_col, pred_col_bin)
print(f"   TN={cm[0,0]}  FP={cm[0,1]}")
print(f"   FN={cm[1,0]}  TP={cm[1,1]}")

# ── Save results ──────────────────────────────────────────────────────────────
results = {
    "model": "MTLPricingModel",
    "test_samples": int(X_test.shape[0]),
    "volatility": {"MAE": round(vol_mae,4), "RMSE": round(vol_rmse,4), "R2": round(vol_r2,4)},
    "trust":      {"MAE": round(tru_mae,4), "RMSE": round(tru_rmse,4), "R2": round(tru_r2,4)},
    "collusion":  {"F1":  round(col_f1,4),  "AUC":  round(col_auc,4),
                   "Precision": round(col_prec,4), "Recall": round(col_rec,4)},
    # arrays for plotting
    "pred_vol":   pred_vol.tolist(),
    "true_vol":   true_vol.tolist(),
    "pred_trust": pred_trust.tolist(),
    "true_trust": true_trust.tolist(),
    "pred_col_prob": pred_col_prob.tolist(),
    "true_col":   true_col.tolist(),
}

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f)

print(f"\n✅  Results saved → {RESULTS_PATH}")
print(f"\n→  Run 6_baselines.py next, then 7_plots.py")
