# =============================================================================
# FILE 4: train.py
# =============================================================================
# Trains the MTL model. Saves best model weights and training history.
# Run this after 1_generate_dataset.py, 2_preprocessing.py, 3_mtl_model.py
# =============================================================================

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os, json, time

import importlib.util, sys as _sys
_spec = importlib.util.spec_from_file_location("mtl_model", "3_mtl_model.py")
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MTLPricingModel = _mod.MTLPricingModel
MTLLoss         = _mod.MTLLoss

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PROCESSED_DIR   = "processed"
MODEL_SAVE_PATH = "best_model.pt"
HISTORY_PATH    = "training_history.json"

INPUT_DIM       = 28
BATCH_SIZE      = 64
LEARNING_RATE   = 0.001
MAX_EPOCHS      = 100
PATIENCE        = 10          # early stopping patience
DROPOUT         = 0.3

W_VOLATILITY    = 0.4
W_TRUST         = 0.4
W_COLLUSION     = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Loading processed data ──────────────────────────────")
X_train = torch.tensor(np.load(f"{PROCESSED_DIR}/X_train.npy"), dtype=torch.float32)
X_val   = torch.tensor(np.load(f"{PROCESSED_DIR}/X_val.npy"),   dtype=torch.float32)
y_train = torch.tensor(np.load(f"{PROCESSED_DIR}/y_train.npy"), dtype=torch.float32)
y_val   = torch.tensor(np.load(f"{PROCESSED_DIR}/y_val.npy"),   dtype=torch.float32)

print(f"   Train: {X_train.shape}  Val: {X_val.shape}")

# DataLoaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset   = TensorDataset(X_val,   y_val)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

# ─────────────────────────────────────────────────────────────────────────────
# 2. INITIALIZE MODEL, LOSS, OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Initializing model ──────────────────────────────────")
model     = MTLPricingModel(input_dim=INPUT_DIM, dropout_rate=DROPOUT).to(device)
criterion = MTLLoss(w_volatility=W_VOLATILITY, w_trust=W_TRUST, w_collusion=W_COLLUSION)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=False)

total_params = sum(p.numel() for p in model.parameters())
print(f"   Total parameters: {total_params:,}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n── Training ({MAX_EPOCHS} epochs, early stopping patience={PATIENCE}) ──")
print(f"{'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>9}  "
      f"{'Vol':>7}  {'Trust':>7}  {'Col':>7}  {'LR':>8}")
print("─" * 70)

history = {
    'train_loss': [], 'val_loss': [],
    'val_vol': [], 'val_trust': [], 'val_col': []
}

best_val_loss   = float('inf')
patience_counter = 0
start_time      = time.time()

for epoch in range(1, MAX_EPOCHS + 1):

    # ── TRAIN ────────────────────────────────────────────────────────────────
    model.train()
    train_losses = []

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_vol   = y_batch[:, 0:1].to(device)
        y_trust = y_batch[:, 1:2].to(device)
        y_col   = y_batch[:, 2:3].to(device)

        optimizer.zero_grad()
        pred_vol, pred_trust, pred_col = model(X_batch)

        loss, lv, lt, lc = criterion(pred_vol, pred_trust, pred_col,
                                     y_vol, y_trust, y_col)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_losses.append(loss.item())

    avg_train_loss = np.mean(train_losses)

    # ── VALIDATE ─────────────────────────────────────────────────────────────
    model.eval()
    val_losses, val_vols, val_trusts, val_cols = [], [], [], []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_vol   = y_batch[:, 0:1].to(device)
            y_trust = y_batch[:, 1:2].to(device)
            y_col   = y_batch[:, 2:3].to(device)

            pred_vol, pred_trust, pred_col = model(X_batch)
            loss, lv, lt, lc = criterion(pred_vol, pred_trust, pred_col,
                                         y_vol, y_trust, y_col)
            val_losses.append(loss.item())
            val_vols.append(lv.item())
            val_trusts.append(lt.item())
            val_cols.append(lc.item())

    avg_val_loss  = np.mean(val_losses)
    avg_val_vol   = np.mean(val_vols)
    avg_val_trust = np.mean(val_trusts)
    avg_val_col   = np.mean(val_cols)
    current_lr    = optimizer.param_groups[0]['lr']

    scheduler.step(avg_val_loss)

    # Log history
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['val_vol'].append(avg_val_vol)
    history['val_trust'].append(avg_val_trust)
    history['val_col'].append(avg_val_col)

    # Print every 5 epochs + first + last
    if epoch % 5 == 0 or epoch == 1:
        marker = " ✓ BEST" if avg_val_loss < best_val_loss else ""
        print(f"{epoch:>6}  {avg_train_loss:>11.5f}  {avg_val_loss:>9.5f}  "
              f"{avg_val_vol:>7.5f}  {avg_val_trust:>7.5f}  "
              f"{avg_val_col:>7.5f}  {current_lr:>8.6f}{marker}")

    # ── EARLY STOPPING + SAVE BEST ───────────────────────────────────────────
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save({
            'epoch':           epoch,
            'model_state':     model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'val_loss':        best_val_loss,
            'input_dim':       INPUT_DIM,
        }, MODEL_SAVE_PATH)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n   Early stopping triggered at epoch {epoch}")
            print(f"   Best val loss: {best_val_loss:.5f}")
            break

elapsed = time.time() - start_time
print(f"\n── Training complete ───────────────────────────────────")
print(f"   Time elapsed   : {elapsed:.1f}s")
print(f"   Best val loss  : {best_val_loss:.5f}")
print(f"   Model saved    → {MODEL_SAVE_PATH}")

# Save history
with open(HISTORY_PATH, 'w') as f:
    json.dump(history, f)
print(f"   History saved  → {HISTORY_PATH}")
print(f"\n✅ Training done. Run 5_evaluate.py next.")