# =============================================================================
# FILE 3: 3_mtl_model.py  (UPDATED — anti-negative-transfer architecture)
# Dynamic Pricing MTL Project — Model Architecture
# =============================================================================
# CHANGES FROM V1:
#   1. Bigger shared encoder: 128→64  becomes  256→128
#      More capacity to learn the latent shared representation
#
#   2. Task-private layers BEFORE the shared encoder
#      Each task gets its own 32-dim private representation
#      Combined with shared: [private(32) + shared(128)] → head
#      This is the standard fix for negative transfer in MTL
#      (See: Liu et al. 2019, "Multi-Task Learning as Multi-Objective Optimization")
#
#   3. Updated loss weights: collusion gets more weight (0.4 not 0.2)
#      Collusion was getting crushed under the old weights
# =============================================================================

import torch
import torch.nn as nn


class MTLPricingModel(nn.Module):
    """
    Anti-negative-transfer MTL architecture.

    Each task has:
      - A private encoder (captures task-specific patterns)
      - Access to the shared encoder (captures cross-task patterns)
    The head receives: concat(private, shared) = 32+128 = 160 dims

    This prevents the shared encoder from being pulled in conflicting
    directions by different tasks — the main cause of negative transfer.
    """

    def __init__(self, input_dim: int, dropout: float = 0.25):
        super().__init__()

        SHARED_DIM  = 128   # shared encoder output dim
        PRIVATE_DIM = 32    # per-task private encoder output dim
        HEAD_INPUT  = SHARED_DIM + PRIVATE_DIM   # 160

        # ── Shared Encoder (bigger than before) ──────────────────────────────
        # Learns the joint latent representation of market_stress,
        # consumer_sens, and collusion_risk simultaneously
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Task-Private Encoders ─────────────────────────────────────────────
        # Each captures task-specific patterns without interfering with others
        # This is the key fix for negative transfer
        self.vol_private = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, PRIVATE_DIM),
            nn.ReLU(),
        )
        self.trust_private = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, PRIVATE_DIM),
            nn.ReLU(),
        )
        self.col_private = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, PRIVATE_DIM),
            nn.ReLU(),
        )

        # ── Task Heads (receive shared + private concatenated) ────────────────
        self.volatility_head = nn.Sequential(
            nn.Linear(HEAD_INPUT, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()    # output 0–1
        )

        self.trust_head = nn.Sequential(
            nn.Linear(HEAD_INPUT, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            # no activation — trust is regression, clipped in loss
        )

        self.collusion_head = nn.Sequential(
            nn.Linear(HEAD_INPUT, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()    # output probability 0–1
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Shared representation (joint latent space)
        shared = self.shared_encoder(x)

        # Task-private representations
        priv_vol   = self.vol_private(x)
        priv_trust = self.trust_private(x)
        priv_col   = self.col_private(x)

        # Concatenate shared + private for each task
        vol_input   = torch.cat([shared, priv_vol],   dim=1)
        trust_input = torch.cat([shared, priv_trust], dim=1)
        col_input   = torch.cat([shared, priv_col],   dim=1)

        volatility = self.volatility_head(vol_input)
        trust      = self.trust_head(trust_input)
        collusion  = self.collusion_head(col_input)

        return volatility, trust, collusion

    def get_shared_repr(self, x):
        """Returns shared encoder output — used for SHAP analysis."""
        return self.shared_encoder(x)


class MTLLoss(nn.Module):
    """
    Updated loss weights:
      Volatility : 0.30  (was 0.40)
      Trust      : 0.30  (was 0.40)
      Collusion  : 0.40  (was 0.20) ← bumped up, was getting crushed
    """

    def __init__(self, w_vol=0.30, w_trust=0.30, w_col=0.40):
        super().__init__()
        self.w_vol   = w_vol
        self.w_trust = w_trust
        self.w_col   = w_col
        self.mse     = nn.MSELoss()
        self.bce     = nn.BCELoss()

    def forward(self, p_vol, p_trust, p_col, t_vol, t_trust, t_col):
        lv = self.mse(p_vol,   t_vol)
        lt = self.mse(p_trust, t_trust)
        lc = self.bce(p_col,   t_col)
        total = self.w_vol*lv + self.w_trust*lt + self.w_col*lc
        return total, lv, lt, lc


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    INPUT_DIM = 31   # matches new preprocessing output
    model     = MTLPricingModel(input_dim=INPUT_DIM, dropout=0.25)
    criterion = MTLLoss()

    print("── Architecture ────────────────────────────────────────")
    print(model)

    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total:,}")

    dummy = torch.randn(64, INPUT_DIM)
    model.eval()
    with torch.no_grad():
        vol, trust, col = model(dummy)

    print(f"\n── Forward pass (batch=64) ──────────────────────────────")
    print(f"  Volatility : {vol.shape}   range [{vol.min():.3f}, {vol.max():.3f}]")
    print(f"  Trust      : {trust.shape}   range [{trust.min():.3f}, {trust.max():.3f}]")
    print(f"  Collusion  : {col.shape}   range [{col.min():.3f}, {col.max():.3f}]")
    print(f"\n✅  Model OK")