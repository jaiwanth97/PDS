# =============================================================================
# predict.py — Single Row Inference Script
# Run this after you've trained the model (best_model.pt must exist)
# Usage: python predict.py
# =============================================================================

import torch
import numpy as np
import pickle
import importlib.util

# ── Load the model class from mtl.py ─────────────────────────────────────────
def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mtl             = load_module("mtl.py", "mtl_model")
MTLPricingModel = mtl.MTLPricingModel

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH    = "best_model.pt"
PROCESSED_DIR = "processed"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load scaler and feature names ─────────────────────────────────────────────
with open(f"{PROCESSED_DIR}/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open(f"{PROCESSED_DIR}/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# ── Load model ────────────────────────────────────────────────────────────────
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model      = MTLPricingModel(input_dim=checkpoint['input_dim']).to(device)
model.load_state_dict(checkpoint['model_state'])
model.eval()

print(f"✅ Model loaded (trained for {checkpoint['epoch']} epochs)")
print(f"   Features expected: {len(feature_names)}")
print(f"   Feature list: {feature_names}\n")


# =============================================================================
# ✏️  EDIT THIS SECTION — Put your input values here
# =============================================================================

# ── Raw numeric features ──────────────────────────────────────────────────────
# These are the non-encoded columns. Change values as needed.
raw_input = {
    "UnitPrice":           9.95,    # price of the item
    "Quantity":            3,       # how many units
    "BasePrice":           8.50,    # base/reference price
    "PriceChangeFreq":     4,       # how often price changed (count)
    "PriceStdDev":         1.2,     # std deviation of price over time
    "PriceRange":          3.0,     # max - min price observed
    "DemandIndicator":     0.7,     # proxy for demand (0 to 1)
    "CompetitorPrice":     10.50,   # competitor's price for same item
    "PriceSyncScore":      0.65,    # how in-sync with competitors (0 to 1)
    "ReactivePricingFlag": 1,       # 1 = reacts to competitor changes, 0 = no
    "SearchDuration":      45,      # seconds user spent searching
    "PurchaseDelay":       2,       # days between first view and purchase
    "NumSearches":         5,       # number of searches before buying
    "ADPExposure":         1,       # 1 = exposed to algorithmic dynamic pricing
    "DidPurchase":         1,       # 1 = purchased, 0 = did not
    "MitigationApplied":   0,       # 1 = price mitigation was applied
    "PriceMatchingFlag":   1,       # 1 = price matching active
    "NumRivals":           3,       # number of competing sellers
}

# ── Categorical features ──────────────────────────────────────────────────────
# Category: one of → "home_decor", "clothing", "toys", "bags", "kitchen", "seasonal"
# Season:   one of → "winter", "spring", "summer", "autumn"

CATEGORY = "clothing"
SEASON   = "winter"

# =============================================================================
# 🔧 Below this line — don't need to edit anything
# =============================================================================

def build_feature_vector(raw_input, category, season, feature_names):
    """
    Converts raw input + categoricals into the exact feature vector
    that the model expects (same order as training).
    """
    # All possible one-hot columns (must match what get_dummies created)
    category_options = ["bags", "clothing", "home_decor", "kitchen", "seasonal", "toys"]
    season_options   = ["autumn", "spring", "summer", "winter"]

    # Build the one-hot encoded dict
    cat_encoded = {f"Category_{c}": int(c == category) for c in category_options}
    sea_encoded = {f"Season_{s}":   int(s == season)   for s in season_options}

    # Merge everything
    full_input = {**raw_input, **cat_encoded, **sea_encoded}

    # Build vector in the exact order of feature_names
    vector = []
    missing = []
    for feat in feature_names:
        if feat in full_input:
            vector.append(float(full_input[feat]))
        else:
            vector.append(0.0)
            missing.append(feat)

    if missing:
        print(f"⚠️  Warning: these features were missing and set to 0: {missing}")

    return np.array(vector, dtype=np.float32)


def predict(raw_input, category, season):
    # Build feature vector
    x_raw = build_feature_vector(raw_input, category, season, feature_names)

    # Scale using the same scaler from training
    x_scaled = scaler.transform(x_raw.reshape(1, -1))

    # Convert to tensor
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(device)

    # Run model
    with torch.no_grad():
        vol, trust, col = model(x_tensor)

    volatility  = float(vol.item())
    trust_score = float(trust.item())
    col_prob    = float(col.item())
    col_flag    = 1 if col_prob >= 0.5 else 0

    return {
        "VolatilityScore": volatility,
        "TrustScore":      trust_score,
        "CollusionFlag":   col_flag,
        "CollusionProb":   col_prob,
    }


# ── Run prediction ────────────────────────────────────────────────────────────
print("=" * 52)
print("         INPUT SUMMARY")
print("=" * 52)
print(f"  Category : {CATEGORY}")
print(f"  Season   : {SEASON}")
for k, v in raw_input.items():
    print(f"  {k:<25}: {v}")

print("\n" + "=" * 52)
print("         PREDICTIONS")
print("=" * 52)

result = predict(raw_input, CATEGORY, SEASON)

vol  = result['VolatilityScore']
tru  = result['TrustScore']
cp   = result['CollusionProb']
cf   = result['CollusionFlag']

print(f"\n  📊 Volatility Score : {vol:.4f}  {'🔴 HIGH' if vol > 0.6 else '🟡 MEDIUM' if vol > 0.3 else '🟢 LOW'}")
print(f"  ⭐ Trust Score      : {tru:.4f}  {'🟢 HIGH TRUST' if tru > 3.5 else '🟡 MEDIUM' if tru > 2.0 else '🔴 LOW TRUST'}")
print(f"  🚨 Collusion Flag   : {'YES 🔴' if cf == 1 else 'NO  🟢'}  (probability: {cp:.4f})")

print("\n" + "=" * 52)
print("  INTERPRETATION")
print("=" * 52)
print(f"""
  Volatility ({vol:.2f}):
    {'Price is highly unstable — possible manipulation.' if vol > 0.6
     else 'Price is moderately volatile.' if vol > 0.3
     else 'Price is stable and predictable.'}

  Trust ({tru:.2f}/5):
    {'Consumers likely trust this pricing.' if tru > 3.5
     else 'Consumer trust is moderate.' if tru > 2.0
     else 'Low consumer trust — pricing may seem unfair.'}

  Collusion ({cp:.2f}):
    {'⚠️  High likelihood of algorithmic price collusion detected!' if cf == 1
     else '✅ No collusion pattern detected.'}
""")
