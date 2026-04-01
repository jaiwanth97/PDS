# =============================================================================
# FILE 3: mtl_model.py
# =============================================================================
# Defines the Multi-Task Learning (MTL) neural network.
# Architecture: Shared Encoder → 3 Task-Specific Heads
#   Head 1: Volatility Score  (regression)
#   Head 2: Trust Score       (regression)
#   Head 3: Collusion Flag    (binary classification)
# =============================================================================

import torch
import torch.nn as nn

class MTLPricingModel(nn.Module):
    """
    Multi-Task Learning model for Algorithmic Dynamic Pricing analysis.

    Architecture:
        Input → Shared Encoder (128→64) → 3 Heads
        Head 1: Volatility Score  (0–1, regression)
        Head 2: Trust Score       (1–5, regression)
        Head 3: Collusion Flag    (0/1, binary classification)

    The shared encoder forces the model to learn a unified representation
    of pricing dynamics, which empirically validates the paper's thesis
    that volatility, trust, and collusion are deeply interconnected.
    """

    def __init__(self, input_dim: int, dropout_rate: float = 0.3):
        super(MTLPricingModel, self).__init__()

        self.input_dim = input_dim

        # ── Shared Encoder ───────────────────────────────────────────────────
        # Two dense layers with BatchNorm + ReLU + Dropout
        # Learns shared representations across all three tasks
        self.shared_encoder = nn.Sequential(
            # Layer 1: input → 128
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Layer 2: 128 → 64
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # ── Head 1: Volatility Score ─────────────────────────────────────────
        # Regression: predicts price manipulation/instability (0 to 1)
        self.volatility_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()        # output constrained to [0, 1]
        )

        # ── Head 2: Trust Score ──────────────────────────────────────────────
        # Regression: predicts consumer trust level (1 to 5)
        # No sigmoid here — raw output scaled externally, or we use MSE loss
        # and let the model learn the 1-5 range naturally
        self.trust_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # ── Head 3: Collusion Flag ───────────────────────────────────────────
        # Binary classification: predicts algorithmic collusion (0 or 1)
        self.collusion_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()        # output = probability of collusion
        )

        # Initialize weights using Kaiming uniform (good for ReLU networks)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: tensor of shape (batch_size, input_dim)
        Returns:
            volatility: (batch_size, 1) — values in [0, 1]
            trust:      (batch_size, 1) — values approximately in [1, 5]
            collusion:  (batch_size, 1) — probability in [0, 1]
        """
        # Pass through shared encoder
        shared_repr = self.shared_encoder(x)

        # Each head gets the same shared representation
        volatility = self.volatility_head(shared_repr)
        trust      = self.trust_head(shared_repr)
        collusion  = self.collusion_head(shared_repr)

        return volatility, trust, collusion

    def get_shared_representation(self, x):
        """
        Returns the shared encoder output (64-dim vector).
        Used for SHAP analysis and feature importance visualization.
        """
        return self.shared_encoder(x)


class MTLLoss(nn.Module):
    """
    Combined weighted loss for the three tasks.

    Total Loss = w1 * MSE(volatility)
               + w2 * MSE(trust)
               + w3 * BCE(collusion)

    Weights reflect the paper's priority:
      - Volatility and Trust are primary research focuses (w=0.4 each)
      - Collusion is secondary/supporting (w=0.2)
    """

    def __init__(self, w_volatility=0.4, w_trust=0.4, w_collusion=0.2):
        super(MTLLoss, self).__init__()
        self.w_vol = w_volatility
        self.w_tru = w_trust
        self.w_col = w_collusion

        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def forward(self, pred_vol, pred_trust, pred_col,
                      true_vol, true_trust, true_col):
        """
        Args:
            pred_*: model predictions (batch_size, 1)
            true_*: ground truth targets (batch_size, 1)
        Returns:
            total_loss, loss_vol, loss_trust, loss_col
        """
        loss_vol   = self.mse(pred_vol,   true_vol)
        loss_trust = self.mse(pred_trust, true_trust)
        loss_col   = self.bce(pred_col,   true_col)

        total = (self.w_vol * loss_vol +
                 self.w_tru * loss_trust +
                 self.w_col * loss_col)

        return total, loss_vol, loss_trust, loss_col


# ─────────────────────────────────────────────────────────────────────────────
# Quick architecture test (run this file directly to verify)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    INPUT_DIM = 28   # matches preprocessing output

    model = MTLPricingModel(input_dim=INPUT_DIM, dropout_rate=0.3)
    criterion = MTLLoss(w_volatility=0.4, w_trust=0.4, w_collusion=0.2)

    print("── Model Architecture ──────────────────────────────────")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n── Parameter Count ─────────────────────────────────────")
    print(f"   Total params    : {total_params:,}")
    print(f"   Trainable params: {trainable:,}")

    # Test forward pass with dummy batch
    dummy_input = torch.randn(64, INPUT_DIM)
    model.eval()
    with torch.no_grad():
        vol, trust, col = model(dummy_input)

    print(f"\n── Forward Pass Test (batch_size=64) ───────────────────")
    print(f"   Input shape      : {dummy_input.shape}")
    print(f"   Volatility output: {vol.shape}  range [{vol.min():.3f}, {vol.max():.3f}]")
    print(f"   Trust output     : {trust.shape}  range [{trust.min():.3f}, {trust.max():.3f}]")
    print(f"   Collusion output : {col.shape}  range [{col.min():.3f}, {col.max():.3f}]")

    print(f"\n✅ Model architecture verified successfully")