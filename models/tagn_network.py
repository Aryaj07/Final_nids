"""
TAGN (Temporal Attention Graph Network) for Enhanced AGILE NIDS
Phase 1: Stream B Implementation - Threat Classification & Temporal Analysis

This module implements the core TAGN architecture for:
- Multi-scale temporal analysis of network flows
- Graph attention for communication pattern analysis
- 7-category threat classification with confidence intervals
- Real-time processing optimization for edge deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List
import math


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism for temporal pattern recognition."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        
        # Generate Q, K, V matrices
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        output = self.w_o(context)
        return output, attention_weights


class TemporalEncoder(nn.Module):
    """LSTM-based encoder for temporal sequence processing."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, (hidden, cell) = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        
        # Combine bidirectional outputs
        combined = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (batch_size, 2*hidden_dim)
        
        return lstm_out, (hidden, cell)


class GraphAttentionLayer(nn.Module):
    """Graph attention layer for network communication analysis."""
    
    def __init__(self, in_features: int, out_features: int, alpha: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leaky_relu = nn.LeakyReLU(alpha)
        
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # h: node features (batch_size, num_nodes, in_features)
        # adj: adjacency matrix (batch_size, num_nodes, num_nodes)
        
        Wh = torch.matmul(h, self.W)  # (batch_size, num_nodes, out_features)
        
        # Compute attention coefficients
        a_input = torch.cat([Wh.repeat(1, 1, Wh.size(1)).view(Wh.size(0), Wh.size(1), Wh.size(1), -1),
                            Wh.repeat(1, Wh.size(1), 1)], dim=-1)
        # Shape: (batch_size, num_nodes, num_nodes, 2*out_features)
        
        e = self.leaky_relu(torch.matmul(a_input, self.a).squeeze(-1))
        # Shape: (batch_size, num_nodes, num_nodes)
        
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        
        h_prime = torch.matmul(attention, Wh)
        
        return F.elu(h_prime)


class DynamicGraphBuilder(nn.Module):
    """Real-time construction of communication graphs from network flows."""
    
    def __init__(self, max_nodes: int = 100, feature_dim: int = 64):
        super().__init__()
        self.max_nodes = max_nodes
        self.feature_dim = feature_dim
        
        # Node embedding for IP addresses
        self.node_embedding = nn.Embedding(max_nodes, feature_dim)
        
        # Edge weight calculator
        self.edge_calculator = nn.Sequential(
            nn.Linear(2 * feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, flow_features: torch.Tensor, 
                src_ips: torch.Tensor, dst_ips: torch.Tensor,
                flow_volumes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            flow_features: (batch_size, seq_len, feature_dim)
            src_ips: (batch_size, seq_len) - source IP indices
            dst_ips: (batch_size, seq_len) - destination IP indices
            flow_volumes: (batch_size, seq_len) - flow volume/byte count
            
        Returns:
            node_features: (batch_size, max_nodes, feature_dim)
            adjacency_matrix: (batch_size, max_nodes, max_nodes)
        """
        batch_size, seq_len, _ = flow_features.shape
        
        # Create node features from flow aggregations
        node_features = torch.zeros(batch_size, self.max_nodes, self.feature_dim, device=flow_features.device)
        
        # Aggregate flow features by destination nodes (hosts being monitored)
        for b in range(batch_size):
            for i in range(seq_len):
                dst_idx = dst_ips[b, i].item()
                if dst_idx < self.max_nodes:
                    # Accumulate features for this node
                    node_features[b, dst_idx] += flow_features[b, i]
        
        # Normalize node features
        node_features = F.normalize(node_features, p=2, dim=-1)
        
        # Build adjacency matrix based on communication patterns
        adjacency = torch.zeros(batch_size, self.max_nodes, self.max_nodes, device=flow_features.device)
        
        for b in range(batch_size):
            for i in range(seq_len):
                src_idx = src_ips[b, i].item()
                dst_idx = dst_ips[b, i].item()
                if src_idx < self.max_nodes and dst_idx < self.max_nodes:
                    # Edge weight based on flow volume and feature similarity
                    src_emb = self.node_embedding(src_idx)
                    dst_emb = self.node_embedding(dst_idx)
                    
                    edge_weight_input = torch.cat([src_emb, dst_emb], dim=-1)
                    edge_weight = self.edge_calculator(edge_weight_input).squeeze(-1)
                    
                    # Scale by flow volume (normalized)
                    volume_weight = torch.sigmoid(flow_volumes[b, i] / 1000.0)  # Normalize volume
                    final_weight = edge_weight * volume_weight
                    
                    adjacency[b, src_idx, dst_idx] = final_weight
                    adjacency[b, dst_idx, src_idx] = final_weight  # Bidirectional
        
        return node_features, adjacency


class MultiScaleTemporalAnalyzer(nn.Module):
    """Processes network flows at multiple temporal scales and fuses patterns."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-scale temporal encoders
        self.short_term_encoder = TemporalEncoder(input_dim, hidden_dim//2, num_layers=1)
        self.medium_term_encoder = TemporalEncoder(input_dim, hidden_dim//2, num_layers=2)
        self.long_term_encoder = TemporalEncoder(input_dim, hidden_dim, num_layers=3)
        
        # Attention mechanisms for each scale
        self.short_attention = MultiHeadAttention(hidden_dim, num_heads)
        self.medium_attention = MultiHeadAttention(hidden_dim, num_heads)
        self.long_attention = MultiHeadAttention(hidden_dim, num_heads)
        
        # Fusion layer to combine multi-scale patterns
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor, sequence_lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (batch_size, seq_len, input_dim)
            sequence_lengths: Actual sequence lengths for padding (batch_size,)
            
        Returns:
            Multi-scale fused representations (batch_size, hidden_dim//2)
        """
        batch_size, seq_len, input_dim = x.size()
        
        # Pad sequences to consistent length
        max_len = seq_len
        padded_x = F.pad(x, (0, 0, 0, max_len - seq_len))
        
        # Split into temporal scales
        short_len = min(10, max_len)
        medium_len = min(50, max_len)
        
        # Short-term processing (last 10 flows)
        short_x = padded_x[:, -short_len:, :]
        short_out, _ = self.short_term_encoder(short_x)
        short_attn_out, _ = self.short_attention(short_out)
        short_repr = torch.mean(short_attn_out, dim=1)  # Global average pooling
        
        # Medium-term processing (last 50 flows)
        medium_x = padded_x[:, -medium_len:, :]
        medium_out, _ = self.medium_term_encoder(medium_x)
        medium_attn_out, _ = self.medium_attention(medium_out)
        medium_repr = torch.mean(medium_attn_out, dim=1)
        
        # Long-term processing (all flows)
        long_out, _ = self.long_term_encoder(padded_x)
        long_attn_out, _ = self.long_attention(long_out)
        long_repr = torch.mean(long_attn_out, dim=1)
        
        # Fuse multi-scale representations
        fused_repr = torch.cat([short_repr, medium_repr, long_repr], dim=-1)
        final_repr = self.fusion_layer(fused_repr)
        
        return final_repr


class ThreatClassifier(nn.Module):
    """7-category threat classification with confidence intervals."""
    
    def __init__(self, input_dim: int, num_classes: int = 7, confidence_dim: int = 64):
        super().__init__()
        self.num_classes = num_classes
        
        # Feature extractor for classification
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Classification head with uncertainty estimation
        self.classifier = nn.Linear(64, num_classes)
        self.confidence_head = nn.Linear(64, confidence_dim)
        
        # Confidence calibration layer
        self.confidence_calibrator = nn.Sequential(
            nn.Linear(confidence_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Threat categories mapping
        self.threat_categories = [
            "Normal", "DDoS", "PortScan", "WebAttack", 
            "Infiltration", "Botnet", "Probe"
        ]
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features (batch_size, input_dim)
            
        Returns:
            class_logits: Classification logits (batch_size, num_classes)
            class_probabilities: Softmax probabilities (batch_size, num_classes)
            confidence_scores: Calibrated confidence scores (batch_size, 1)
        """
        features = self.feature_extractor(x)
        
        # Classification logits
        class_logits = self.classifier(features)
        
        # Class probabilities
        class_probabilities = F.softmax(class_logits, dim=-1)
        
        # Confidence estimation
        confidence_features = self.confidence_head(features)
        confidence_scores = self.confidence_calibrator(confidence_features)
        
        return class_logits, class_probabilities, confidence_scores
    
    def predict_with_confidence(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate predictions with confidence intervals."""
        logits, probs, confidence = self.forward(x)
        
        predictions = {
            'predicted_class': torch.argmax(probs, dim=-1),
            'class_probabilities': probs,
            'confidence_score': confidence.squeeze(-1),
            'logits': logits
        }
        
        return predictions


class TAGNNetwork(nn.Module):
    """Complete TAGN (Temporal Attention Graph Network) implementation."""
    
    def __init__(self, 
                 input_dim: int = 78,
                 hidden_dim: int = 128,
                 num_heads: int = 8,
                 num_classes: int = 7,
                 max_nodes: int = 100):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Feature preprocessing
        self.feature_preprocessor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Multi-scale temporal analysis
        self.temporal_analyzer = MultiScaleTemporalAnalyzer(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads
        )
        
        # Dynamic graph construction and attention
        self.graph_builder = DynamicGraphBuilder(max_nodes=max_nodes, feature_dim=hidden_dim//2)
        self.graph_attention = GraphAttentionLayer(hidden_dim//2, hidden_dim//2)
        
        # Feature fusion
        self.temporal_fusion = nn.Sequential(
            nn.Linear(hidden_dim//2 + hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Threat classification
        self.threat_classifier = ThreatClassifier(hidden_dim, num_classes)
        
        # Priority scoring
        self.priority_scorer = nn.Sequential(
            nn.Linear(hidden_dim + num_classes, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # CRITICAL, HIGH, MEDIUM, LOW
            nn.LogSoftmax(dim=-1)
        )
        
    def forward(self, 
                flow_features: torch.Tensor,
                src_ips: Optional[torch.Tensor] = None,
                dst_ips: Optional[torch.Tensor] = None,
                flow_volumes: Optional[torch.Tensor] = None,
                sequence_lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Complete TAGN forward pass.
        
        Args:
            flow_features: Network flow features (batch_size, seq_len, input_dim)
            src_ips: Source IP indices (batch_size, seq_len)
            dst_ips: Destination IP indices (batch_size, seq_len)
            flow_volumes: Flow volumes (batch_size, seq_len)
            sequence_lengths: Actual sequence lengths (batch_size,)
            
        Returns:
            Dictionary containing all outputs
        """
        batch_size, seq_len, _ = flow_features.shape
        
        # Initialize sequence lengths if not provided
        if sequence_lengths is None:
            sequence_lengths = torch.full((batch_size,), seq_len, device=flow_features.device)
        
        # 1. Feature preprocessing
        processed_features = self.feature_preprocessor(flow_features)
        
        # 2. Multi-scale temporal analysis
        temporal_repr = self.temporal_analyzer(processed_features, sequence_lengths)
        
        # 3. Graph-based analysis (if IP information available)
        graph_repr = None
        if src_ips is not None and dst_ips is not None:
            node_features, adjacency = self.graph_builder(
                processed_features, src_ips, dst_ips, flow_volumes
            )
            graph_repr = self.graph_attention(node_features, adjacency)
            # Aggregate graph representation
            graph_repr = torch.mean(graph_repr, dim=1)  # (batch_size, hidden_dim//2)
        
        # 4. Feature fusion
        if graph_repr is not None:
            fused_repr = torch.cat([temporal_repr, graph_repr], dim=-1)
        else:
            fused_repr = temporal_repr
        
        fused_repr = self.temporal_fusion(fused_repr)
        
        # 5. Threat classification
        classification_results = self.threat_classifier.predict_with_confidence(fused_repr)
        
        # 6. Priority scoring
        priority_input = torch.cat([fused_repr, classification_results['class_probabilities']], dim=-1)
        priority_scores = self.priority_scorer(priority_input)
        
        # 7. Compile results
        results = {
            'temporal_representation': temporal_repr,
            'graph_representation': graph_repr,
            'fused_representation': fused_repr,
            'classification': classification_results,
            'priority_scores': priority_scores,
            'priority_levels': ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
        }
        
        return results


def create_tagn_model(input_dim: int = 78, 
                     hidden_dim: int = 128,
                     num_heads: int = 8,
                     num_classes: int = 7) -> TAGNNetwork:
    """
    Factory function to create a TAGN model with default configuration.
    
    Args:
        input_dim: Input feature dimension (default: 78 for CICIDS2017)
        hidden_dim: Hidden layer dimensions
        num_heads: Number of attention heads
        num_classes: Number of threat categories
        
    Returns:
        Configured TAGNNetwork instance
    """
    model = TAGNNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_classes=num_classes
    )
    
    return model


# Utility functions for model usage
def extract_ip_features(ip_addresses: List[str], max_nodes: int = 100) -> torch.Tensor:
    """
    Extract IP-based features for graph construction.
    
    Args:
        ip_addresses: List of IP addresses
        max_nodes: Maximum number of nodes to consider
        
    Returns:
        IP feature tensor
    """
    # Simple IP hashing for demonstration
    # In practice, you'd want more sophisticated IP feature engineering
    ip_features = []
    for ip in ip_addresses:
        # Convert IP to numeric representation
        parts = ip.split('.')
        if len(parts) == 4:
            # Simple hash based on IP components
            ip_hash = sum(int(part) * (256 ** (3 - i)) for i, part in enumerate(parts[:4]))
            normalized_ip = ip_hash % max_nodes
            ip_features.append(normalized_ip)
        else:
            ip_features.append(0)
    
    return torch.tensor(ip_features, dtype=torch.long)


if __name__ == "__main__":
    # Test the TAGN implementation
    print("🧪 Testing TAGN Network Implementation...")
    
    # Create sample data
    batch_size, seq_len, input_dim = 4, 50, 78
    
    # Sample flow features
    flow_features = torch.randn(batch_size, seq_len, input_dim)
    
    # Sample IP data
    src_ips = torch.randint(0, 50, (batch_size, seq_len))
    dst_ips = torch.randint(0, 50, (batch_size, seq_len))
    flow_volumes = torch.rand(batch_size, seq_len) * 1000
    sequence_lengths = torch.tensor([50, 45, 40, 35])
    
    # Create and test model
    model = create_tagn_model(input_dim=input_dim)
    
    print(f"📊 Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    results = model(flow_features, src_ips, dst_ips, flow_volumes, sequence_lengths)
    
    # Display results
    print(f"✅ Temporal representation shape: {results['temporal_representation'].shape}")
    print(f"✅ Fused representation shape: {results['fused_representation'].shape}")
    print(f"✅ Classification shape: {results['classification']['class_probabilities'].shape}")
    print(f"✅ Priority scores shape: {results['priority_scores'].shape}")
    
    # Show threat categories
    print(f"🎯 Threat categories: {model.threat_classifier.threat_categories}")
    
    print("🎉 TAGN Network test completed successfully!")