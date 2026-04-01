# =============================================================================
# FILE 7: 7_plots.py
# Dynamic Pricing MTL Project — All Paper Figures
# =============================================================================
# Generates all figures needed for the conference paper.
# Run AFTER 5_evaluate.py and 6_baselines.py
# Outputs: plots/ folder with PNG files
# =============================================================================

import numpy as np
import json, os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Load results ──────────────────────────────────────────────────────────────
print("── Loading results ─────────────────────────────────────")
with open("training_history.json")  as f: history  = json.load(f)
with open("evaluation_results.json") as f: eval_res = json.load(f)
with open("baseline_results.json")  as f: baselines = json.load(f)
print("   All result files loaded")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1: Training & Validation Loss Curves
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Plot 1: Loss curves ─────────────────────────────────")
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("MTL Model — Training History", fontsize=13, fontweight='bold')

epochs = range(1, len(history['train_loss']) + 1)

plots_cfg = [
    ('train_loss', 'val_loss',   'Combined Loss',    '#2196F3', '#F44336'),
    ('val_vol',    None,          'Volatility Loss',  '#4CAF50', None),
    ('val_trust',  None,          'Trust Loss',       '#FF9800', None),
    ('val_col',    None,          'Collusion Loss',   '#9C27B0', None),
]

for ax, (train_key, val_key, title, c1, c2) in zip(axes, plots_cfg):
    ax.plot(epochs, history[train_key], color=c1, label='Train', linewidth=2)
    if val_key:
        ax.plot(epochs, history[val_key], color=c2, label='Val',
                linewidth=2, linestyle='--')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
path = f"{PLOTS_DIR}/plot1_training_curves.png"
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved → {path}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2: ROC Curve — Collusion Detection
# ─────────────────────────────────────────────────────────────────────────────
print("── Plot 2: ROC curve ───────────────────────────────────")
fig, ax = plt.subplots(figsize=(6, 6))

true_col  = np.array(eval_res['true_col'])
prob_col  = np.array(eval_res['pred_col_prob'])
fpr, tpr, _ = roc_curve(true_col, prob_col)
roc_auc_val = auc(fpr, tpr)

ax.plot(fpr, tpr, color='#2196F3', lw=2,
        label=f'MTL Model (AUC = {roc_auc_val:.4f})')
ax.plot([0,1],[0,1], 'k--', lw=1, label='Random classifier')
ax.fill_between(fpr, tpr, alpha=0.08, color='#2196F3')

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve — Collusion Detection', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

path = f"{PLOTS_DIR}/plot2_roc_curve.png"
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved → {path}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3: Predicted vs Actual — Volatility & Trust
# ─────────────────────────────────────────────────────────────────────────────
print("── Plot 3: Predicted vs Actual ─────────────────────────")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("MTL Model — Predicted vs Actual Values", fontsize=13, fontweight='bold')

for ax, pred_key, true_key, title, color, r2_val in [
    (axes[0], 'pred_vol',   'true_vol',   'Volatility Score',
     '#4CAF50', eval_res['volatility']['R2']),
    (axes[1], 'pred_trust', 'true_trust', 'Trust Score',
     '#FF9800', eval_res['trust']['R2']),
]:
    pred = np.array(eval_res[pred_key])
    true = np.array(eval_res[true_key])

    ax.scatter(true, pred, alpha=0.3, s=10, color=color)
    lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
    ax.plot(lims, lims, 'r--', lw=1.5, label='Perfect prediction')
    ax.set_xlabel(f'Actual {title}', fontsize=11)
    ax.set_ylabel(f'Predicted {title}', fontsize=11)
    ax.set_title(f'{title}  (R² = {r2_val:.4f})', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
path = f"{PLOTS_DIR}/plot3_pred_vs_actual.png"
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved → {path}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4: Baseline Comparison Bar Chart
# ─────────────────────────────────────────────────────────────────────────────
print("── Plot 4: Baseline comparison ─────────────────────────")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("MTL Model vs Baselines", fontsize=13, fontweight='bold')

model_names = list(baselines.keys())
colors      = ['#9E9E9E','#9E9E9E','#9E9E9E','#2196F3']
# Make MTL bar stand out
bar_colors  = ['#BBDEFB','#90CAF9','#64B5F6','#1565C0']

metrics_cfg = [
    # (task, metric_key, ylabel, lower_is_better, title)
    ('volatility', 'MAE',  'MAE (lower=better)',  True,  'Volatility — MAE'),
    ('trust',      'MAE',  'MAE (lower=better)',  True,  'Trust Score — MAE'),
    ('collusion',  'F1',   'F1 (higher=better)',  False, 'Collusion — F1 Score'),
]

for ax, (task, metric, ylabel, lower_better, title) in zip(axes, metrics_cfg):
    vals = [baselines[m][task][metric] for m in model_names]

    bars = ax.bar(model_names, vals, color=bar_colors, edgecolor='white', linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticklabels(model_names, rotation=15, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Highlight best bar
    best_idx = vals.index(min(vals) if lower_better else max(vals))
    bars[best_idx].set_edgecolor('#F44336')
    bars[best_idx].set_linewidth(2.5)

plt.tight_layout()
path = f"{PLOTS_DIR}/plot4_baseline_comparison.png"
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved → {path}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5: Residual Distribution
# ─────────────────────────────────────────────────────────────────────────────
print("── Plot 5: Residual distributions ─────────────────────")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("MTL Model — Prediction Residuals", fontsize=13, fontweight='bold')

for ax, pred_key, true_key, title, color in [
    (axes[0], 'pred_vol',   'true_vol',   'Volatility Score', '#4CAF50'),
    (axes[1], 'pred_trust', 'true_trust', 'Trust Score',      '#FF9800'),
]:
    pred = np.array(eval_res[pred_key])
    true = np.array(eval_res[true_key])
    resid = true - pred

    ax.hist(resid, bins=50, color=color, alpha=0.7, edgecolor='white')
    ax.axvline(0, color='red', linestyle='--', lw=1.5, label='Zero error')
    ax.set_xlabel('Residual (Actual − Predicted)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'{title} Residuals\n'
                 f'mean={resid.mean():.4f}  std={resid.std():.4f}', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
path = f"{PLOTS_DIR}/plot5_residuals.png"
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved → {path}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 6: Full Metrics Summary (Paper-Ready Table as Figure)
# ─────────────────────────────────────────────────────────────────────────────
print("── Plot 6: Metrics summary table ───────────────────────")
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')

model_list = list(baselines.keys())
col_labels = ['Model',
              'Vol MAE','Vol RMSE','Vol R²',
              'Trust MAE','Trust RMSE','Trust R²',
              'Col F1','Col AUC']

table_data = []
for m in model_list:
    v = baselines[m]['volatility']
    t = baselines[m]['trust']
    c = baselines[m]['collusion']
    table_data.append([
        m,
        f"{v['MAE']:.4f}", f"{v['RMSE']:.4f}", f"{v['R2']:.4f}",
        f"{t['MAE']:.4f}", f"{t['RMSE']:.4f}", f"{t['R2']:.4f}",
        f"{c['F1']:.4f}",  f"{c['AUC']:.4f}",
    ])

tbl = ax.table(cellText=table_data, colLabels=col_labels,
               loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.2, 2.0)

# Highlight header
for j in range(len(col_labels)):
    tbl[0,j].set_facecolor('#1565C0')
    tbl[0,j].set_text_props(color='white', fontweight='bold')

# Highlight MTL row
mtl_idx = model_list.index('MTLModel') + 1
for j in range(len(col_labels)):
    tbl[mtl_idx, j].set_facecolor('#E3F2FD')
    tbl[mtl_idx, j].set_text_props(fontweight='bold')

ax.set_title('Model Comparison — All Metrics', fontsize=13,
             fontweight='bold', pad=20)

path = f"{PLOTS_DIR}/plot6_metrics_table.png"
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved → {path}")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n✅  All plots saved to /{PLOTS_DIR}/")
print(f"\nFiles generated:")
for f in sorted(os.listdir(PLOTS_DIR)):
    print(f"   {PLOTS_DIR}/{f}")
print(f"\nUse these directly in your IEEE paper figures.")
