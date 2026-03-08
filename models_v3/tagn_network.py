"""
TAGN v3 — Temporal Attention Graph Network (Stream B).
HALO NIDS v3 — Algorithm 1 Steps 9-11:
  9.  Stream B: Threat Classification (TAGN Supervised Model)
  10. Construct hard k-NN graph G from recent V using Euclidean distance
      with tau-threshold and edge weights; apply temporal encoding + stacked GAT
  11. Classify with TC -> C  (15-class CICIDS2017 taxonomy)

Changes from v2:
  - NUM_CLASSES: 7 -> 15 (full CICIDS2017 granular taxonomy)
  - FlowGraphBuilder: soft cosine adj -> hard k-NN with tau-threshold + edge weights
  - GraphAttentionLayer: stacked 2-layer GAT for deeper structural reasoning
  - hidden_dim: 128 -> 256 for richer minority-class feature separation
  - n_heads: 4 -> 8
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple


# -----------------------------------------------------------------------------
# 15-class taxonomy  (matches GCN-2-Former Table 6 / CICIDS2017 raw labels)
# -----------------------------------------------------------------------------

THREAT_LABELS = [
    "BENIGN",            #  0
    "Bot",               #  1
    "DDoS",              #  2
    "DoS GoldenEye",     #  3
    "DoS Hulk",          #  4
    "DoS Slowhttptest",  #  5
    "DoS Slowloris",     #  6
    "FTP-Patator",       #  7
    "Heartbleed",        #  8
    "Infiltration",      #  9
    "PortScan",          # 10
    "SSH-Patator",       # 11
    "Brute Force",       # 12
    "SQL Injection",     # 13
    "XSS",               # 14
]

NUM_CLASSES = len(THREAT_LABELS)  # 15

# Attack super-family lookup (used by LLM / correlation engine for context)
ATTACK_FAMILY = {
    1:  "Botnet",
    2:  "DDoS", 3: "DDoS", 4: "DDoS", 5: "DDoS", 6: "DDoS",
    7:  "Brute-Force Probe", 11: "Brute-Force Probe",
    8:  "Exploitation", 9: "Infiltration",
    10: "Reconnaissance",
    12: "Web Attack", 13: "Web Attack", 14: "Web Attack",
}


# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------

class TemporalEncoder(nn.Module):
    """Bidirectional LSTM encoder for sequential flow processing."""
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        out, _ = self.lstm(x)
        return self.dropout(out)


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product multi-head self-attention."""
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
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
        attn = self.drop(F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / self.scale, dim=-1))
        ctx = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, D)
        return self.Wo(ctx)


# -----------------------------------------------------------------------------
# v3 NEW: Hard k-NN Graph Builder  (replaces soft cosine adjacency)
# Algorithm Step 10: E = {(i,j) : D(i,j) < tau}, omega_ij = 1 - D(i,j)
# -----------------------------------------------------------------------------

class HardKNNGraphBuilder(nn.Module):
    """
    Constructs a hard k-NN graph from flow feature sequences.

    v2 used soft cosine similarity -> adj = relu(V * V^T), which assigns
    non-zero edge weight to almost every node pair and washes out
    the topological signal for minority attack classes.

    v3 uses:
      1. Project flows into node space
      2. Compute pairwise Euclidean distance D(i,j) = ||v_i - v_j||_2
      3. Hard threshold: keep edge only if D(i,j) < tau  (sparse adjacency)
      4. Per-node top-k: retain only k nearest neighbours
      5. Edge weight: omega_ij = 1 - D(i,j)  (closer = stronger)

    Same construction strategy as GCN-2-Former (tau=0.5, k=10).
    """

    def __init__(self, flow_dim: int, node_dim: int,
                 max_nodes: int = 32, tau: float = 0.5, k: int = 10):
        super().__init__()
        self.max_nodes = max_nodes
        self.tau = tau
        self.k = k
        self.node_proj = nn.Sequential(
            nn.Linear(flow_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.ReLU(),
        )

    def forward(self, flow_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        flow_seq : (B, T, D)
        Returns
            node_feats : (B, N, node_dim)
            adj        : (B, N, N)   sparse weighted adjacency
        """
        B, T, D = flow_seq.shape
        N = min(T, self.max_nodes)
        nodes_in = flow_seq[:, :N, :]           # (B, N, D)
        node_feats = self.node_proj(nodes_in)   # (B, N, nd)

        # Pairwise Euclidean distance
        diff = node_feats.unsqueeze(2) - node_feats.unsqueeze(1)  # (B,N,N,nd)
        dist = torch.norm(diff, dim=-1)                            # (B,N,N)

        # Hard tau-threshold
        mask_tau = dist < self.tau                                 # (B,N,N) bool

        # Top-k nearest neighbours per node
        if N > self.k:
            _, topk_idx = torch.topk(dist, k=self.k, dim=-1, largest=False)
            mask_k = torch.zeros_like(dist, dtype=torch.bool)
            mask_k.scatter_(-1, topk_idx, True)
        else:
            mask_k = torch.ones_like(dist, dtype=torch.bool)

        # Combined mask: satisfy BOTH tau-threshold AND top-k; remove self-loops
        mask = mask_tau & mask_k
        eye = torch.eye(N, dtype=torch.bool, device=dist.device).unsqueeze(0)
        mask = mask & ~eye

        # Edge weight: omega_ij = 1 - D(i,j), clipped to [0, 1]
        omega = (1.0 - dist).clamp(0.0, 1.0)
        adj = omega * mask.float()              # (B, N, N)

        return node_feats, adj


# -----------------------------------------------------------------------------
# v3 NEW: Stacked 2-layer GAT  (replaces single-layer GAT)
# -----------------------------------------------------------------------------

class GraphAttentionLayer(nn.Module):
    """Single-head graph attention (GAT) layer."""
    def __init__(self, in_dim: int, out_dim: int, alpha: float = 0.2):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.leaky = nn.LeakyReLU(alpha)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        Wh = self.W(h)
        N = Wh.size(1)
        Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1)
        Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1)
        e = self.leaky(self.a(torch.cat([Wh_i, Wh_j], dim=-1)).squeeze(-1))
        e = torch.where(adj > 0, e, torch.full_like(e, -1e9))
        alpha = F.softmax(e, dim=-1)
        # Weight by edge weight omega from hard k-NN graph
        alpha = alpha * adj
        alpha = alpha / (alpha.sum(dim=-1, keepdim=True) + 1e-8)
        return F.elu(torch.matmul(alpha, Wh))


class StackedGAT(nn.Module):
    """Two GAT layers with residual connections for deeper graph reasoning."""
    def __init__(self, node_dim: int):
        super().__init__()
        self.gat1 = GraphAttentionLayer(node_dim, node_dim)
        self.gat2 = GraphAttentionLayer(node_dim, node_dim)
        self.norm1 = nn.LayerNorm(node_dim)
        self.norm2 = nn.LayerNorm(node_dim)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.norm1(h + self.gat1(h, adj))
        h = self.norm2(h + self.gat2(h, adj))
        return h


# -----------------------------------------------------------------------------
# Multi-scale temporal analyser
# -----------------------------------------------------------------------------

class MultiScaleTemporalAnalyser(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_heads: int = 8):
        super().__init__()
        h = hidden_dim
        self.enc_short  = TemporalEncoder(input_dim, h // 2, n_layers=1)
        self.enc_medium = TemporalEncoder(input_dim, h // 2, n_layers=2)
        self.enc_long   = TemporalEncoder(input_dim, h // 2, n_layers=2)
        self.attn_short  = MultiHeadSelfAttention(h, n_heads)
        self.attn_medium = MultiHeadSelfAttention(h, n_heads)
        self.attn_long   = MultiHeadSelfAttention(h, n_heads)
        self.fusion = nn.Sequential(
            nn.Linear(h * 3, h), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h, h),     nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        s = self.attn_short (self.enc_short (x[:, -min(10, T):, :])).mean(1)
        m = self.attn_medium(self.enc_medium(x[:, -min(50, T):, :])).mean(1)
        l = self.attn_long  (self.enc_long  (x)).mean(1)
        return self.fusion(torch.cat([s, m, l], dim=-1))


# -----------------------------------------------------------------------------
# Complete TAGN v3 Network
# -----------------------------------------------------------------------------

class TAGNNetwork(nn.Module):
    """
    TAGN v3: 15-class classifier with hard k-NN graph + stacked GAT.

    Key differences from v2:
      - NUM_CLASSES = 15
      - HardKNNGraphBuilder  (tau=0.5, k=10)
      - StackedGAT (2 layers with residuals)
      - hidden_dim = 256 (was 128)
      - n_heads    = 8   (was 4)
    """

    def __init__(
        self,
        input_dim:       int   = 80,
        hidden_dim:      int   = 256,
        n_heads:         int   = 8,
        num_classes:     int   = NUM_CLASSES,
        max_graph_nodes: int   = 32,
        tau:             float = 0.5,
        k_neighbours:    int   = 10,
        dropout:         float = 0.3,
    ):
        super().__init__()
        self.input_dim   = input_dim
        self.hidden_dim  = hidden_dim
        self.num_classes = num_classes
        node_dim = hidden_dim // 2

        # 1. Feature projection
        self.feat_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )

        # 2. Multi-scale temporal analyser
        self.temporal = MultiScaleTemporalAnalyser(hidden_dim, hidden_dim, n_heads)

        # 3. Hard k-NN graph builder + stacked GAT  [v3 change]
        self.graph_builder = HardKNNGraphBuilder(
            hidden_dim, node_dim, max_graph_nodes, tau, k_neighbours
        )
        self.gat = StackedGAT(node_dim)

        # 4. Fusion
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim + node_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )

        # 5. Classification head  (output = 15)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        # 6. Correlation feature extractor
        self.corr_proj = nn.Linear(hidden_dim, 16)

        # 7. Confidence head
        self.conf_head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(),
            nn.Linear(32, 1),          nn.Sigmoid(),
        )

    def _project_features(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        return self.feat_proj(x.reshape(B * T, D)).reshape(B, T, -1)

    def forward(self, x: torch.Tensor, **kwargs) -> Dict:
        h               = self._project_features(x)
        temporal_repr   = self.temporal(h)
        node_feats, adj = self.graph_builder(h)
        graph_repr      = self.gat(node_feats, adj).mean(dim=1)
        fused           = self.fuse(torch.cat([temporal_repr, graph_repr], dim=-1))
        logits          = self.classifier(fused)
        probs           = F.softmax(logits, dim=-1)
        return {
            "classification": {
                "logits":              logits,
                "class_probabilities": probs,
                "predicted_class":     torch.argmax(probs, dim=-1),
                "confidence_score":    self.conf_head(fused).squeeze(-1),
            },
            "correlation_features":  self.corr_proj(fused),
            "fused_representation":  fused,
        }


def create_tagn_model(
    input_dim:   int   = 80,
    hidden_dim:  int   = 256,
    n_heads:     int   = 8,
    num_classes: int   = NUM_CLASSES,
    dropout:     float = 0.3,
) -> TAGNNetwork:
    return TAGNNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        num_classes=num_classes,
        dropout=dropout,
    )
