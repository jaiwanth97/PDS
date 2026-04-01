# =============================================================================
# FILE 8: 8_shap_analysis.py
# Dynamic Pricing MTL Project — SHAP Feature Importance
# =============================================================================

import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle, json, os, importlib.util, warnings
warnings.filterwarnings('ignore')

PLOTS_DIR     = "plots"
PROCESSED_DIR = "processed"
MODEL_PATH    = "best_model.pt"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Import model ──────────────────────────────────────────────────────────────
def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mtl             = load_module("mtl.py", "mtl_model")
MTLPricingModel = mtl.MTLPricingModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load feature names ────────────────────────────────────────────────────────
print("── Loading data ────────────────────────────────────────")
with open(f"{PROCESSED_DIR}/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

X_test  = np.load(f"{PROCESSED_DIR}/X_test.npy").astype(np.float32)
X_train = np.load(f"{PROCESSED_DIR}/X_train.npy").astype(np.float32)

print(f"   Test samples  : {X_test.shape[0]}")
print(f"   Features      : {len(feature_names)}")

# ── Load model ────────────────────────────────────────────────────────────────
print("── Loading model ───────────────────────────────────────")
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model      = MTLPricingModel(input_dim=checkpoint['input_dim']).to(device)
model.load_state_dict(checkpoint['model_state'])
model.eval()
print(f"   Loaded from epoch {checkpoint['epoch']}")

# ── Create wrapper functions for each head ────────────────────────────────────
def predict_volatility(x):
    t = torch.tensor(x, dtype=torch.float32).to(device)
    with torch.no_grad():
        vol, _, _ = model(t)
    return vol.cpu().numpy().reshape(-1, 1)

def predict_trust(x):
    t = torch.tensor(x, dtype=torch.float32).to(device)
    with torch.no_grad():
        _, trust, _ = model(t)
    return trust.cpu().numpy().reshape(-1, 1)

def predict_collusion(x):
    t = torch.tensor(x, dtype=torch.float32).to(device)
    with torch.no_grad():
        _, _, col = model(t)
    return col.cpu().numpy().reshape(-1, 1)

# ── SHAP background sample ────────────────────────────────────────────────────
print("\n── Setting up SHAP explainer ───────────────────────────")
np.random.seed(42)
bg_idx      = np.random.choice(len(X_train), size=100, replace=False)
background  = X_train[bg_idx]

explain_idx = np.random.choice(len(X_test), size=200, replace=False)
X_explain   = X_test[explain_idx]

print(f"   Background samples : {background.shape[0]}")
print(f"   Explain samples    : {X_explain.shape[0]}")
print(f"   This may take 2-5 minutes...")

# ── Compute SHAP values for each head ─────────────────────────────────────────
heads = [
    ("Volatility Score", predict_volatility, "#4CAF50", "vol"),
    ("Trust Score",      predict_trust,      "#FF9800", "trust"),
    ("Collusion Flag",   predict_collusion,  "#9C27B0", "col"),
]

all_shap_values = {}

for head_name, predict_fn, color, key in heads:
    print(f"\n── Computing SHAP: {head_name} ──────────────────────────")
    explainer   = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_explain, nsamples=50, l1_reg='num_features(10)')
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.array(shap_values)
    # Squeeze to 2D (200, 28) regardless of output shape
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]
    all_shap_values[key] = shap_values
    print(f"   Done. Shape: {shap_values.shape}")

# ── Clean up feature names for display ───────────────────────────────────────
def clean_name(name):
    replacements = {
        'PriceChangeFreq':    'Price Change Freq',
        'PriceStdDev':        'Price Std Dev',
        'PriceRange':         'Price Range',
        'DemandIndicator':    'Demand Indicator',
        'CompetitorPrice':    'Competitor Price',
        'PriceSyncScore':     'Price Sync Score',
        'ReactivePricingFlag':'Reactive Pricing',
        'SearchDuration':     'Search Duration',
        'PurchaseDelay':      'Purchase Delay',
        'NumSearches':        'Num Searches',
        'ADPExposure':        'ADP Exposure',
        'DidPurchase':        'Did Purchase',
        'MitigationApplied':  'Mitigation Applied',
        'PriceMatchingFlag':  'Price Matching',
        'NumRivals':          'Num Rivals',
        'UnitPrice':          'Unit Price',
        'BasePrice':          'Base Price',
        'Quantity':           'Quantity',
        'Category_bags':      'Category: Bags',
        'Category_clothing':  'Category: Clothing',
        'Category_home_decor':'Category: Home Decor',
        'Category_kitchen':   'Category: Kitchen',
        'Category_seasonal':  'Category: Seasonal',
        'Category_toys':      'Category: Toys',
        'Season_autumn':      'Season: Autumn',
        'Season_spring':      'Season: Spring',
        'Season_summer':      'Season: Summer',
        'Season_winter':      'Season: Winter',
    }
    return replacements.get(name, name)

display_names = [clean_name(n) for n in feature_names]

# ─────────────────────────────────────────────────────────────────────────────
# PLOT A: Mean |SHAP| bar chart — Top 12 features per head
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Generating SHAP plots ───────────────────────────────")

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle("SHAP Feature Importance — MTL Model\n"
             "(Mean |SHAP value| = average impact on model output)",
             fontsize=13, fontweight='bold', y=1.01)

TOP_N = 12

for ax, (head_name, _, color, key) in zip(axes, heads):
    sv         = all_shap_values[key]                          # (200, 28)
    mean_abs   = np.abs(sv).mean(axis=0)                       # (28,)
    sorted_idx = np.argsort(mean_abs)[::-1][:TOP_N][::-1]     # top N, ascending for barh
    idx_list   = sorted_idx.tolist()                           # plain Python ints

    vals  = [float(mean_abs[i]) for i in idx_list]            # plain Python floats
    names = [display_names[i]   for i in idx_list]

    bars = ax.barh(names, vals, color=color, alpha=0.8, edgecolor='white', linewidth=0.8)

    for bar, val in zip(bars, vals):
        ax.text(val + 0.0002, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=8)

    ax.set_title(f'{head_name}', fontsize=12, fontweight='bold', color=color)
    ax.set_xlabel('Mean |SHAP value|', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
path = f"{PLOTS_DIR}/shap_feature_importance.png"
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved → {path}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT B: SHAP Summary Dot Plot
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle("SHAP Summary — Feature Impact Direction\n"
             "(Right = pushes prediction higher  |  Left = pushes prediction lower)",
             fontsize=13, fontweight='bold', y=1.01)

for ax, (head_name, _, color, key) in zip(axes, heads):
    sv       = all_shap_values[key]                        # (200, 28)
    mean_abs = np.abs(sv).mean(axis=0)                     # (28,)
    top_idx  = np.argsort(mean_abs)[::-1][:TOP_N][::-1]   # ascending for plot
    idx_list = top_idx.tolist()

    for i, feat_idx in enumerate(idx_list):
        shap_vals = sv[:, feat_idx].flatten()              # (200,)
        feat_vals = X_explain[:, feat_idx].flatten()       # (200,)

        ax.scatter(
            shap_vals,
            np.full(len(shap_vals), i) + np.random.normal(0, 0.07, len(shap_vals)),
            c=feat_vals, cmap='RdYlBu_r',
            alpha=0.5, s=12, linewidth=0
        )

    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_yticks(range(TOP_N))
    ax.set_yticklabels([display_names[i] for i in idx_list], fontsize=9)
    ax.set_xlabel('SHAP value (impact on output)', fontsize=10)
    ax.set_title(f'{head_name}', fontsize=12, fontweight='bold', color=color)
    ax.grid(True, alpha=0.2, axis='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap='RdYlBu_r',
                                norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Feature value\n(low → high)', fontsize=8)

plt.tight_layout()
path = f"{PLOTS_DIR}/shap_summary_dots.png"
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved → {path}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT C: Unified heatmap
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 9))

all_mean_abs = np.stack([
    np.abs(all_shap_values['vol']).mean(axis=0),
    np.abs(all_shap_values['trust']).mean(axis=0),
    np.abs(all_shap_values['col']).mean(axis=0),
])
avg_importance = all_mean_abs.mean(axis=0)
top15_idx      = np.argsort(avg_importance)[::-1][:15].tolist()

heatmap_data = all_mean_abs[:, top15_idx]
feat_labels  = [display_names[i] for i in top15_idx]
head_labels  = ['Volatility', 'Trust Score', 'Collusion']

heatmap_norm = heatmap_data / (heatmap_data.max(axis=1, keepdims=True) + 1e-9)

im = ax.imshow(heatmap_norm, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

ax.set_xticks(range(len(feat_labels)))
ax.set_xticklabels(feat_labels, rotation=45, ha='right', fontsize=10)
ax.set_yticks(range(len(head_labels)))
ax.set_yticklabels(head_labels, fontsize=11, fontweight='bold')

for i in range(len(head_labels)):
    for j in range(len(feat_labels)):
        val  = float(heatmap_data[i, j])
        norm = float(heatmap_norm[i, j])
        ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                fontsize=7.5, color='black' if norm < 0.6 else 'white')

plt.colorbar(im, ax=ax, label='Relative Feature Importance (row-normalized)')
ax.set_title("SHAP Importance Heatmap — Top 15 Features × All 3 Tasks\n"
             "(Shows which features are important across the entire MTL model)",
             fontsize=12, fontweight='bold', pad=15)

plt.tight_layout()
path = f"{PLOTS_DIR}/shap_heatmap.png"
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved → {path}")

# ─────────────────────────────────────────────────────────────────────────────
# PRINT TEXT SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n╔══════════════════════════════════════════════════════════════╗")
print("║          SHAP FINDINGS — PASTE INTO PAPER DISCUSSION        ║")
print("╠══════════════════════════════════════════════════════════════╣")

for head_name, _, color, key in heads:
    sv       = all_shap_values[key]
    mean_abs = np.abs(sv).mean(axis=0)
    top5_idx = np.argsort(mean_abs)[::-1][:5].tolist()
    print(f"\n  {head_name} — Top 5 drivers:")
    for rank, idx in enumerate(top5_idx, 1):
        direction = "↑ increases" if float(sv[:, idx].mean()) > 0 else "↓ decreases"
        print(f"    {rank}. {display_names[idx]:<25} "
              f"(mean|SHAP|={float(mean_abs[idx]):.4f}, {direction})")

print("\n╚══════════════════════════════════════════════════════════════╝")

np.save(f"{PROCESSED_DIR}/shap_vol.npy",   all_shap_values['vol'])
np.save(f"{PROCESSED_DIR}/shap_trust.npy", all_shap_values['trust'])
np.save(f"{PROCESSED_DIR}/shap_col.npy",   all_shap_values['col'])

print(f"\n✅  SHAP analysis complete!")
print(f"\nPlots saved:")
print(f"   plots/shap_feature_importance.png  → Use as Fig 3 in paper")
print(f"   plots/shap_summary_dots.png        → Use as Fig 4 in paper")
print(f"   plots/shap_heatmap.png             → Use as Fig 5 in paper")
print(f"\nSHAP values saved to processed/shap_*.npy for further analysis")