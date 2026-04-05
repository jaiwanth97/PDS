# =============================================================================
# FILE 2: 2_preprocessing.py  (REVISED — handles missing data properly)
# Dynamic Pricing MTL Project — Preprocessing Pipeline
# =============================================================================
# CHANGE FROM V1: Instead of dropping rows with missing values (which would
# lose ~20% of data), we now impute them using median + missing indicator flags.
# This is the correct approach for real-world messy data and is standard
# practice in production ML systems.
# =============================================================================

import pandas as pd
import numpy as np
import pickle, os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

INPUT_PATH  = "dataset.csv"
OUTPUT_DIR  = "processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("── Step 1: Load ────────────────────────────────────────")
df = pd.read_csv(INPUT_PATH)
print(f"   Loaded {df.shape[0]} rows × {df.shape[1]} columns")

# ── 2. Drop identifier columns ────────────────────────────────────────────────
print("── Step 2: Drop identifiers ────────────────────────────")
df = df.drop(columns=['InvoiceNo','StockCode','Description',
                       'CustomerID','Country','InvoiceDate'])
print(f"   Remaining: {df.shape[1]} columns")

# ── 3. Remove exact duplicates only (keep NaN rows — they are real) ───────────
print("── Step 3: Remove exact duplicates ─────────────────────")
before = len(df)
df = df.drop_duplicates()
print(f"   Removed {before-len(df)} duplicate rows  |  Remaining: {len(df)}")

# ── 4. Report missing data ────────────────────────────────────────────────────
print("── Step 4: Missing data report ─────────────────────────")
missing = df.isnull().sum()
missing = missing[missing > 0]
for col, cnt in missing.items():
    print(f"   {col:<25} : {cnt} missing ({100*cnt/len(df):.1f}%)")

# ── 5. Impute missing values (do NOT drop rows) ───────────────────────────────
print("── Step 5: Impute missing values ───────────────────────")
# For each column with missing values:
# a) Fill with median of that column
# b) Add a binary flag column indicating WHERE it was missing
# This way the model learns TWO things: the imputed value AND the missingness pattern
COLS_WITH_MISSING = ['CompetitorPrice', 'SearchDuration', 'PriceSyncScore']

for col in COLS_WITH_MISSING:
    if col in df.columns and df[col].isnull().any():
        # Add missingness indicator flag
        flag_col = f"{col}_missing"
        df[flag_col] = df[col].isnull().astype(int)
        # Impute with median
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"   {col:<25}: imputed with median={median_val:.3f}, "
              f"added flag column '{flag_col}'")

# Any remaining nulls — fill with median
remaining_nulls = df.isnull().sum()
remaining_nulls = remaining_nulls[remaining_nulls > 0]
if len(remaining_nulls) > 0:
    for col in remaining_nulls.index:
        df[col] = df[col].fillna(df[col].median())
        print(f"   {col}: filled remaining nulls with median")

print(f"   Total nulls remaining: {df.isnull().sum().sum()}")
print(f"   Columns now: {df.shape[1]}")

# ── 6. One-hot encode categoricals ───────────────────────────────────────────
print("── Step 6: One-hot encode ──────────────────────────────")
df = pd.get_dummies(df, columns=['Category','Season'], drop_first=False)
print(f"   Columns after encoding: {df.shape[1]}")

# ── 7. Separate features and targets ─────────────────────────────────────────
print("── Step 7: Separate X / y ──────────────────────────────")
TARGET_COLS  = ['VolatilityScore','TrustScore','CollusionFlag']
FEATURE_COLS = [c for c in df.columns if c not in TARGET_COLS]

X = df[FEATURE_COLS].values.astype(np.float32)
y = df[TARGET_COLS].values.astype(np.float32)
print(f"   X: {X.shape}  |  y: {y.shape}")
print(f"   Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")

# ── 8. Normalize features ────────────────────────────────────────────────────
print("── Step 8: Normalize → [0,1] ───────────────────────────")
scaler   = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
with open(f"{OUTPUT_DIR}/scaler.pkl","wb") as f:       pickle.dump(scaler,f)
with open(f"{OUTPUT_DIR}/feature_names.pkl","wb") as f: pickle.dump(FEATURE_COLS,f)
print(f"   Scaler saved → {OUTPUT_DIR}/scaler.pkl")

# ── 9. Train / Val / Test split (70 / 15 / 15) ───────────────────────────────
print("── Step 9: Train/Val/Test split ────────────────────────")
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X_scaled, y,
                                              test_size=0.30, random_state=42)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp,
                                              test_size=0.50, random_state=42)
print(f"   Train: {len(X_tr)}  Val: {len(X_val)}  Test: {len(X_te)}")

# ── 10. Save splits ───────────────────────────────────────────────────────────
print("── Step 10: Save ───────────────────────────────────────")
for name, arr in [("X_train",X_tr),("X_val",X_val),("X_test",X_te),
                  ("y_train",y_tr),("y_val",y_val),("y_test",y_te)]:
    np.save(f"{OUTPUT_DIR}/{name}.npy", arr)
    print(f"   Saved → {OUTPUT_DIR}/{name}.npy")

print(f"\n✅  Preprocessing complete!")
print(f"   Final feature count = {X_tr.shape[1]}")
print(f"   (Use this as INPUT_DIM — train.py auto-detects it)")