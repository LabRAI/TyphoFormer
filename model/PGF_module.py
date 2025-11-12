"""
pgf_module.py
--------------------------------------------------
Step 3 of TyphoFormer Implementation:
Prompt-aware Gating Fusion (PGF) Module

This module fuses numerical meteorological features (x_t)
and textual semantic embeddings (p_t) into a unified representation.

Author: Lincan Li
Date: 2025-06-02
"""

import torch
import torch.nn as nn
from utils import *

class PromptGatingFusion(nn.Module):
    """
    Implements the Prompt-aware Gating Fusion (PGF) mechanism from TyphoFormer.

    Equation:
        x_tilde = σ(W_g [x_t; p_t] + b_g) ⊙ x_t + (1 - σ(...)) ⊙ p_t
    """
    def __init__(self, d_num, d_text, d_out, debug=False):
        """
        Args:
            d_num:  Dimension of numerical input features (x_t)
            d_text: Dimension of text embedding features (p_t)
            d_out:  Output feature dimension (default = d_num)
        """
        super().__init__()
        self.d_out = d_out or d_num
        self.fc_gate = nn.Linear(d_num + d_text, self.d_out)
        self.proj_num = nn.Linear(d_num, self.d_out)
        self.proj_text = nn.Linear(d_text, self.d_out)
        self.sigmoid = nn.Sigmoid()
        self.debug = debug

    def forward(self, x_t: torch.Tensor, p_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: [B, T, d_num]  numerical features
            p_t: [B, T, d_text] textual embeddings (from Step 2)
        Returns:
            x_tilde: [B, T, 1, d_out] fused representation (for STAEformer)
        """
        if x_t.shape[:2] != p_t.shape[:2]:
            raise ValueError(f"Shape mismatch: x_t {x_t.shape}, p_t {p_t.shape}")

        fused_input = torch.cat([x_t, p_t], dim=-1)
        gate = self.sigmoid(self.fc_gate(fused_input))  # [B, T, d_out]

        x_proj = self.proj_num(x_t)
        p_proj = self.proj_text(p_t)
        x_tilde = gate * x_proj + (1 - gate) * p_proj  # [B, T, d_out]
        # Add spatial dimension for STAEformer (N=1)
        x_tilde = x_tilde.unsqueeze(2)  # [B, T, 1, d_out]
        if self.debug:
            print("\n[PGF] --- Forward ---")
            print_shape("fused_input", fused_input, 4)
            print_shape("gate", gate, 4)
            print_shape("fused_output", x_tilde, 4)

        #`gate` must be returned because we will use it as a panelty in our loss function to panelize too small `gate` value
        return x_tilde,gate



# Example usage test
if __name__ == "__main__":
    B, T, d_num, d_text = 4, 8, 14, 384
    x_t = torch.randn(B, T, d_num)
    p_t = torch.randn(B, T, d_text)

    pgf = PromptGatingFusion(d_num, d_text)
    fused = pgf(x_t, p_t)

    print("\n[DEBUG] PGF Output Verification:")
    print(f"  Input numerical: {x_t.shape}")
    print(f"  Input textual:    {p_t.shape}")
    print(f"  Fused output:     {fused.shape}  (Expected: [B, T, 1, d_out])")
