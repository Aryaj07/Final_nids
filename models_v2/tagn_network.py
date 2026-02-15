"""
TAGN — Temporal Attention Graph Network (Stream B).
Algorithm Steps 9-11:
  9.  Stream B: Threat Classification (TAGN Supervised Model)
  10. Construct graph G from recent V; apply temporal encoding and graph attention
  11. Classify with TC → C  (Benign, DDoS, PortScan, WebAttack, Infiltration, Botnet, Probe)

This module implements the FULL TAGN architecture described in the paper:
  - Multi-scale temporal analysis via bidirectional LSTM + multi-head attention
  - Graph attention layer for communication-pattern modelling
  - Lightweight dynamic graph construction from flow features
  - 7-class threat classification with calibrated confidence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────

class TemporalEncoder(nn.Module):
    """Bidirectional LSTM encoder for sequential flow processing."""

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # x: (B, T, D)
        out, (h, c) = self.lstm(x)          # out: (B, T, 2*H)
        out = self.dropout(out)
        return out, (h, c)


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product multi-head self-attention."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.dk = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.scale = math.sqrt(self.dk)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.size()
        Q = self.Wq(x).view(B, T, self.n_heads, self.dk).transpose(1, 2)
        K = self.Wk(x).view(B, T, self.n_heads, self.dk).transpose(1, 2)
        V = self.Wv(x).view(B, T, self.n_heads, self.dk).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = self.drop(F.softmax(scores, dim=-1))
        ctx = torch.matmul(attn, V)                              # (B, H, T, dk)
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, D)     # (B, T, D)
        return self.Wo(ctx)


class GraphAttentionLayer(nn.Module):
    """Single-head graph attention (GAT) layer."""

    def __init__(self, in_dim: int, out_dim: int, alpha: float = 0.2):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.leaky = nn.LeakyReLU(alpha)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        h:   (B, N, in_dim)  — node features
        adj: (B, N, N)       — adjacency (>0 means edge exists)
        """
        Wh = self.W(h)                                          # (B, N, out)
        N = Wh.size(1)
        # pairwise concat
        Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1)           # (B, N, N, out)
        Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1)           # (B, N, N, out)
        e = self.leaky(self.a(torch.cat([Wh_i, Wh_j], dim=-1)).squeeze(-1))  # (B, N, N)

        neg_inf = -1e9 * torch.ones_like(e)
        e = torch.where(adj > 0, e, neg_inf)
        alpha = F.softmax(e, dim=-1)
        out = torch.matmul(alpha, Wh)                           # (B, N, out)
        return F.elu(out)


# ──────────────────────────────────────────────────────────────────────────────
# Multi-scale temporal analyser
# ──────────────────────────────────────────────────────────────────────────────

class MultiScaleTemporalAnalyser(nn.Module):
    """Process flows at short / medium / long time-scales, then fuse."""

    def __init__(self, input_dim: int, hidden_dim: int, n_heads: int = 4):
        super().__init__()
        h = hidden_dim
        # three temporal encoders — each outputs 2*h because bidirectional
        self.enc_short  = TemporalEncoder(input_dim, h // 2, n_layers=1)
        self.enc_medium = TemporalEncoder(input_dim, h // 2, n_layers=2)
        self.enc_long   = TemporalEncoder(input_dim, h // 2, n_layers=2)

        # attention over each scale (input dim = 2*(h//2) = h)
        self.attn_short  = MultiHeadSelfAttention(h, n_heads)
        self.attn_medium = MultiHeadSelfAttention(h, n_heads)
        self.attn_long   = MultiHeadSelfAttention(h, n_heads)

        # fuse three scales → single vector
        self.fusion = nn.Sequential(
            nn.Linear(h * 3, h),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(h, h),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, hidden_dim)"""
        T = x.size(1)
        short_len  = min(10, T)
        medium_len = min(50, T)

        s_out, _ = self.enc_short(x[:, -short_len:, :])
        s_out = self.attn_short(s_out).mean(dim=1)

        m_out, _ = self.enc_medium(x[:, -medium_len:, :])
        m_out = self.attn_medium(m_out).mean(dim=1)

        l_out, _ = self.enc_long(x)
        l_out = self.attn_long(l_out).mean(dim=1)

        fused = torch.cat([s_out, m_out, l_out], dim=-1)        # (B, 3*h)
        return self.fusion(fused)                                 # (B, h)


# ──────────────────────────────────────────────────────────────────────────────
# Lightweight graph builder
# ──────────────────────────────────────────────────────────────────────────────

class FlowGraphBuilder(nn.Module):
    """
    Build a small communication graph from a batch of flow-feature sequences.

    Instead of requiring explicit IP indices (which are unavailable during
    offline CSV-based training), we learn a soft graph:
      - Project each flow into a node-space
      - Compute pairwise similarity → adjacency
    This gives the GAT layer meaningful structure to attend over.
    """

    def __init__(self, flow_dim: int, node_dim: int, max_nodes: int = 32):
        super().__init__()
        self.max_nodes = max_nodes
        self.node_proj = nn.Sequential(
            nn.Linear(flow_dim, node_dim),
            nn.ReLU(),
        )

    def forward(self, flow_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        flow_seq: (B, T, D)
        Returns:
            node_feats: (B, N, node_dim)     N = min(T, max_nodes)
            adj:        (B, N, N)            soft adjacency
        """
        B, T, D = flow_seq.shape
        N = min(T, self.max_nodes)
        # Sub-sample or take first N flows as "nodes"
        nodes_in = flow_seq[:, :N, :]                            # (B, N, D)
        node_feats = self.node_proj(nodes_in)                    # (B, N, nd)

        # Soft adjacency via cosine similarity
        nf_norm = F.normalize(node_feats, dim=-1)
        adj = torch.bmm(nf_norm, nf_norm.transpose(1, 2))       # (B, N, N) in [-1,1]
        adj = F.relu(adj)                                        # keep positive edges
        return node_feats, adj


# ──────────────────────────────────────────────────────────────────────────────
# Complete TAGN Network
# ──────────────────────────────────────────────────────────────────────────────

# Label mapping used throughout the project
THREAT_LABELS = [
    "BENIGN",        # 0
    "DDoS",          # 1
    "PortScan",      # 2
    "Web Attack",    # 3
    "Infiltration",  # 4
    "Botnet",        # 5
    "Probe",         # 6
]

NUM_CLASSES = len(THREAT_LABELS)


class TAGNNetwork(nn.Module):
    """
    Full TAGN: temporal encoding + graph attention + multi-class classifier.

    Outputs dict with keys:
        classification.logits            (B, C)
        classification.class_probabilities (B, C)
        classification.predicted_class   (B,)
        classification.confidence_score  (B,)
        correlation_features             (B, 16)   — for the correlation engine
    """

    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 128,
        n_heads: int = 4,
        num_classes: int = NUM_CLASSES,
        max_graph_nodes: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # 1. Feature projection
        self.feat_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),      # applied per-flow below
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )

        # 2. Multi-scale temporal analyser
        self.temporal = MultiScaleTemporalAnalyser(hidden_dim, hidden_dim, n_heads)

        # 3. Graph builder + GAT
        node_dim = hidden_dim // 2
        self.graph_builder = FlowGraphBuilder(hidden_dim, node_dim, max_graph_nodes)
        self.gat = GraphAttentionLayer(node_dim, node_dim)

        # 4. Fusion of temporal + graph representations
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim + node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 5. Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        # 6. Correlation-feature extractor (16-d vector for downstream engine)
        self.corr_proj = nn.Linear(hidden_dim, 16)

        # 7. Confidence head (scalar)
        self.conf_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------
    def _project_features(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feat_proj with BatchNorm handling for 3-D input."""
        B, T, D = x.shape
        # flatten → BN → reshape
        flat = x.reshape(B * T, D)
        proj = self.feat_proj(flat)
        return proj.reshape(B, T, -1)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        x: (B, T, input_dim)   — sequence of flow feature vectors
        """
        # 1. project
        h = self._project_features(x)                            # (B, T, H)

        # 2. temporal path
        temporal_repr = self.temporal(h)                          # (B, H)

        # 3. graph path
        node_feats, adj = self.graph_builder(h)                  # (B, N, nd), (B, N, N)
        graph_out = self.gat(node_feats, adj)                    # (B, N, nd)
        graph_repr = graph_out.mean(dim=1)                       # (B, nd)

        # 4. fuse
        fused = self.fuse(torch.cat([temporal_repr, graph_repr], dim=-1))  # (B, H)

        # 5. classify
        logits = self.classifier(fused)                           # (B, C)
        probs  = F.softmax(logits, dim=-1)

        # 6. correlation features
        corr_feats = self.corr_proj(fused)                        # (B, 16)

        # 7. confidence
        conf = self.conf_head(fused).squeeze(-1)                  # (B,)

        return {
            "classification": {
                "logits": logits,
                "class_probabilities": probs,
                "predicted_class": torch.argmax(probs, dim=-1),
                "confidence_score": conf,
            },
            "correlation_features": corr_feats,
            "fused_representation": fused,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def create_tagn_model(
    input_dim: int = 80,
    hidden_dim: int = 128,
    n_heads: int = 4,
    num_classes: int = NUM_CLASSES,
) -> TAGNNetwork:
    return TAGNNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        num_classes=num_classes,
    )
