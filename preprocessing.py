# =============================================================================
# FILE 2: preprocessing.py
# =============================================================================
# Loads dataset.csv, cleans it, encodes categoricals, normalizes numerics,
# and splits into train / validation / test sets.
# Outputs: X_train, X_val, X_test, y_train, y_val, y_test as .npy files
#          + scaler object as scaler.pkl
# =============================================================================

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

INPUT_PATH  = "dataset.csv"
OUTPUT_DIR  = "processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────────────────
print("── Step 1: Loading dataset ─────────────────────────────")
df = pd.read_csv(INPUT_PATH)
print(f"   Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────────────────────────────────────────
# 2. DROP COLUMNS NOT USED IN MODELING
# (InvoiceNo, StockCode, Description, CustomerID, Country, InvoiceDate
#  are real UCI identifiers — useful for EDA but not model features)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Step 2: Dropping identifier columns ─────────────────")
DROP_COLS = ['InvoiceNo', 'StockCode', 'Description',
             'CustomerID', 'Country', 'InvoiceDate']
df = df.drop(columns=DROP_COLS)
print(f"   Remaining columns: {df.shape[1]}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. HANDLE MISSING VALUES
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Step 3: Handling missing values ─────────────────────")
before = len(df)
df = df.drop_duplicates()
df = df.dropna()
after = len(df)
print(f"   Removed {before - after} duplicate/null rows")
print(f"   Remaining rows: {after}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. ONE-HOT ENCODE CATEGORICALS
# (Category: home_decor, clothing, toys, bags, kitchen, seasonal)
# (Season: winter, spring, summer, autumn)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Step 4: One-hot encoding categoricals ───────────────")
df = pd.get_dummies(df, columns=['Category', 'Season'], drop_first=False)
print(f"   Columns after encoding: {df.shape[1]}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. SEPARATE FEATURES AND TARGETS
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Step 5: Separating features and targets ─────────────")
TARGET_COLS = ['VolatilityScore', 'TrustScore', 'CollusionFlag']
FEATURE_COLS = [c for c in df.columns if c not in TARGET_COLS]

X = df[FEATURE_COLS].values.astype(np.float32)
y = df[TARGET_COLS].values.astype(np.float32)

print(f"   Features (X): {X.shape}")
print(f"   Targets  (y): {y.shape}")
print(f"   Feature names: {FEATURE_COLS}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. NORMALIZE FEATURES (MinMaxScaler → all values 0 to 1)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Step 6: Normalizing features ────────────────────────")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print(f"   All features scaled to [0, 1]")

# Save scaler for use during inference later
with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
print(f"   Scaler saved → {OUTPUT_DIR}/scaler.pkl")

# Also save feature names
with open(os.path.join(OUTPUT_DIR, "feature_names.pkl"), "wb") as f:
    pickle.dump(FEATURE_COLS, f)

# ─────────────────────────────────────────────────────────────────────────────
# 7. TRAIN / VAL / TEST SPLIT  (70 / 15 / 15)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Step 7: Splitting into train / val / test ───────────")
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42)

print(f"   Train : {X_train.shape[0]} rows  ({100*len(X_train)/len(X_scaled):.0f}%)")
print(f"   Val   : {X_val.shape[0]} rows  ({100*len(X_val)/len(X_scaled):.0f}%)")
print(f"   Test  : {X_test.shape[0]} rows  ({100*len(X_test)/len(X_scaled):.0f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 8. SAVE ALL SPLITS AS .npy FILES
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Step 8: Saving splits ───────────────────────────────")
np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "X_val.npy"),   X_val)
np.save(os.path.join(OUTPUT_DIR, "X_test.npy"),  X_test)
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "y_val.npy"),   y_val)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"),  y_test)

for fname in ["X_train","X_val","X_test","y_train","y_val","y_test"]:
    print(f"   Saved → {OUTPUT_DIR}/{fname}.npy")

# ─────────────────────────────────────────────────────────────────────────────
# 9. QUICK SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Sanity check ────────────────────────────────────────")
print(f"   X_train range: [{X_train.min():.3f}, {X_train.max():.3f}]")
print(f"   y targets (train):")
print(f"     VolatilityScore  mean={y_train[:,0].mean():.3f}  std={y_train[:,0].std():.3f}")
print(f"     TrustScore       mean={y_train[:,1].mean():.3f}  std={y_train[:,1].std():.3f}")
print(f"     CollusionFlag    positive rate={y_train[:,2].mean():.3f}")
print(f"\n✅ Preprocessing complete. All files saved to /{OUTPUT_DIR}/")