"""View-Aware Attention Fusion (VAAF) for CXR multi-view late fusion.

Theory
------
In clinical radiology, chest X-ray diagnosis follows a view-aware protocol:

  * Frontal (z_F) -- primary diagnostic anchor (heart, lungs, mediastinum)
  * Lateral (z_L) -- depth disambiguation (retrocardiac, posterior structures)
  * Cross-view (z_X) -- synergistic evidence (correlations across projections)

VAAF assigns each view a learnable *view embedding* that encodes its diagnostic
function, then uses C=14 disease query prototypes that attend over the 3
view-aware tokens via gated multi-head cross-attention.  The resulting
Disease-View Affinity Matrix alpha in [0,1]^{C x 3} tells us, for each
pathology, how much it relies on frontal vs lateral vs cross-view information.

References
----------
- Q2L (Liu et al., 2021)  -- label queries + cross-attention for multi-label
- Flamingo (Alayrac et al., NeurIPS 2022) -- gated cross-attention, zero-init gate
- CrossViT (Chen et al., ICCV 2021) -- CLS-token cross-attention between branches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# View indices (for readability and attention map access)
VIEW_FRONTAL = 0
VIEW_LATERAL = 1
VIEW_CROSS = 2
VIEW_NAMES = ["frontal", "lateral", "cross-view"]


class ViewAwareAttentionFusion(nn.Module):
    """CXR-specific View-Aware Attention Fusion (VAAF).

    Each of the C pathological labels has a learnable disease-query prototype
    that attends over 3 view-aware tokens (frontal, lateral,
    cross-view) via multi-head cross-attention with gating.

    Parameters
    ----------
    embed_dim : int
        Feature vector dimensionality (768 for ViT-B / EVA-X).
    num_labels : int
        Number of classification labels (14 for CXR).
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate for attention weights and output.
    gate_bias : float
        Initial bias for the gate projection.  Negative value -> model starts
        by relying on query priors and gradually learns to trust attention.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_labels: int = 14,
        num_heads: int = 4,
        dropout: float = 0.2,
        gate_bias: float = -2.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.embed_dim = embed_dim
        self.num_labels = num_labels
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_views = 3  # frontal, lateral, cross-view

        # ── View-aware embeddings ──────────────────────────────────────
        # Encode the diagnostic function of each view (analogous to the
        # view embeddings used in the multi-view token merging stage).
        self.view_embeddings = nn.Parameter(
            torch.randn(1, self.num_views, embed_dim) * 0.02
        )

        # ── Disease query prototypes ───────────────────────────────────
        # Each query learns to represent a specific pathology and to attend
        # to the views most informative for that disease.
        self.disease_queries = nn.Parameter(
            torch.randn(1, num_labels, embed_dim) * 0.02
        )

        # ── Multi-head cross-attention projections ─────────────────────
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

        # ── Gating mechanism ──────────────────────────────────────────
        self.gate_proj = nn.Linear(2 * embed_dim, embed_dim)
        nn.init.constant_(self.gate_proj.bias, gate_bias)

        # ── Layer normalization ────────────────────────────────────────
        self.ln_views = nn.LayerNorm(embed_dim)
        self.ln_queries = nn.LayerNorm(embed_dim)
        self.ln_out = nn.LayerNorm(embed_dim)

        # ── Per-label classifier ───────────────────────────────────────
        self.classifier = nn.Linear(embed_dim, 1)

        # ── Dropout ────────────────────────────────────────────────────
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # ── Stored attention map for visualization ─────────────────────
        self._disease_view_affinity: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        z_frontal: torch.Tensor,
        z_cross: torch.Tensor,
        z_lateral: torch.Tensor,
    ) -> torch.Tensor:
        """Compute disease prediction by attending over view-aware tokens.

        Args:
            z_frontal: [B, D] - frontal-view CLS feature (frozen encoder)
            z_cross:   [B, D] - cross-view fused feature (multi-view encoder)
            z_lateral: [B, D] - lateral-view CLS feature (frozen encoder)

        Returns:
            logits: [B, C] - per-label correction logits
        """
        B = z_frontal.shape[0]

        # 1. Stack view features and add view-aware embeddings  [B, 3, D]
        #    Order: [frontal, lateral, cross-view]
        views = torch.stack([z_frontal, z_lateral, z_cross], dim=1)
        view_emb = F.normalize(self.view_embeddings, dim=-1)
        views = views + view_emb  # view-aware tokens
        views = self.ln_views(views)

        # 2. Disease query prototypes  [B, C, D]
        queries = self.disease_queries.expand(B, -1, -1)
        queries = self.ln_queries(queries)

        # 3. Multi-head cross-attention: queries attend over views
        #    Q: [B, H, C, d_k]   K,V: [B, H, 3, d_k]
        q = (
            self.W_q(queries)
            .view(B, self.num_labels, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.W_k(views)
            .view(B, self.num_views, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.W_v(views)
            .view(B, self.num_views, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Attention weights: [B, H, C, 3]  (Disease-View Affinity Matrix)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Store Disease-View Affinity Matrix (averaged over heads)
        self._disease_view_affinity = attn_weights.mean(dim=1).detach()

        # Weighted aggregation: [B, H, C, d_k] -> [B, C, D]
        attn_out = torch.matmul(attn_weights, v)
        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(B, self.num_labels, self.embed_dim)
        )
        attn_out = self.W_o(attn_out)

        # 4. Gated residual fusion
        gate_input = torch.cat([queries, attn_out], dim=-1)  # [B, C, 2D]
        gate = torch.sigmoid(self.gate_proj(gate_input))     # [B, C, D]

        fused = gate * attn_out + (1 - gate) * queries       # [B, C, D]
        fused = self.ln_out(fused)
        fused = self.dropout(fused)

        # 5. Per-label prediction: [B, C] 
        logits = self.classifier(fused).squeeze(-1)

        return logits

    # ------------------------------------------------------------------
    # Visualization & Interpretability
    # ------------------------------------------------------------------

    def get_disease_view_affinity(self) -> torch.Tensor | None:
        """Return the Disease-View Affinity Matrix.

        Returns
        -------
        affinity : Tensor [B, 14, 3] or None
            Column order: [frontal, lateral, cross-view].
            Each row sums to 1 and shows how much each disease relies
            on each view for its prediction.
        """
        return self._disease_view_affinity

    @staticmethod
    def format_affinity_table(
        affinity: torch.Tensor,
        label_names: list[str] | None = None,
    ) -> str:
        """Pretty-print a single sample's Disease-View Affinity Matrix.

        Args:
            affinity: [C, 3] tensor (one sample, no batch dim).
            label_names: optional list of C label names.

        Returns:
            Formatted string table.
        """
        default_labels = [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
            "Enlarged Cardiomed.", "Fracture", "Lung Lesion", "Lung Opacity",
            "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
            "Pneumothorax", "Support Devices",
        ]
        names = label_names or default_labels
        header = f"{'Pathology':<22s} {'Frontal':>8s} {'Lateral':>8s} {'Cross-V':>8s}"
        sep = "─" * len(header)
        rows = [sep, header, sep]
        for i, name in enumerate(names):
            f_val = affinity[i, VIEW_FRONTAL].item()
            l_val = affinity[i, VIEW_LATERAL].item()
            x_val = affinity[i, VIEW_CROSS].item()
            rows.append(f"{name:<22s} {f_val:>8.3f} {l_val:>8.3f} {x_val:>8.3f}")
        rows.append(sep)
        return "\n".join(rows)
