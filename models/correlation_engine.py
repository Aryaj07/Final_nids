"""
Correlation Engine for Enhanced AGILE NIDS
Phase 1: Dual-Stream Signal Fusion

This module implements the correlation engine that intelligently combines:
- Stream A: Autoencoder anomaly detection scores
- Stream B: TAGN network threat classification and confidence

Features:
- Weighted signal correlation for precise threat prioritization
- Dynamic weight adjustment based on detection performance
- Statistical confidence measures for all decisions
- Real-time optimization for edge deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
import math
from dataclasses import dataclass
from enum import Enum


class ThreatLevel(Enum):
    """Threat priority levels as defined in PRD."""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass
class CorrelationResult:
    """Result structure for correlation engine output."""
    correlation_score: float
    confidence_score: float
    threat_level: ThreatLevel
    stream_a_contribution: float
    stream_b_contribution: float
    uncertainty_estimate: float
    decision_rationale: str


class WeightedSignalCorrelator(nn.Module):
    """Intelligent combination of anomaly and classification signals."""
    
    def __init__(self, 
                 autoencoder_dim: int = 1,  # Anomaly score
                 tagn_dim: int = 16,         # 16-dim correlation features from TAGN
                 confidence_dim: int = 1,    # Confidence score
                 hidden_dim: int = 64):
        super().__init__()
        self.autoencoder_dim = autoencoder_dim
        self.tagn_dim = tagn_dim
        self.confidence_dim = confidence_dim
        
        # Input fusion layers
        self.input_fusion = nn.Sequential(
            nn.Linear(autoencoder_dim + tagn_dim + confidence_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Dynamic weight calculator
        self.weight_calculator = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # Weights for Stream A and Stream B
            nn.Softmax(dim=-1)
        )
        
        # Correlation score calculator
        self.correlation_calculator = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Uncertainty estimator
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, 
                anomaly_scores: torch.Tensor,
                threat_probabilities: torch.Tensor,
                confidence_scores: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            anomaly_scores: Autoencoder anomaly scores (batch_size, 1)
            threat_probabilities: TAGN correlation features (batch_size, 16)
            confidence_scores: TAGN confidence scores (batch_size, 1)
            
        Returns:
            Dictionary containing correlation results
        """
        # Combine all inputs
        combined_input = torch.cat([anomaly_scores, threat_probabilities, confidence_scores], dim=-1)
        
        # Fuse inputs
        fused_representation = self.input_fusion(combined_input)
        
        # Calculate dynamic weights
        weights = self.weight_calculator(fused_representation)  # (batch_size, 2)
        stream_a_weight = weights[:, 0:1]  # Autoencoder weight
        stream_b_weight = weights[:, 1:2]  # TAGN weight
        
        # Calculate correlation score
        correlation_score = self.correlation_calculator(fused_representation)
        
        # Calculate confidence
        confidence_score = self.confidence_estimator(fused_representation)
        
        # Calculate uncertainty
        uncertainty_score = self.uncertainty_estimator(fused_representation)
        
        # Calculate individual contributions
        max_threat_prob = torch.max(threat_probabilities, dim=-1, keepdim=True)[0]
        stream_a_contribution = anomaly_scores * stream_a_weight
        stream_b_contribution = max_threat_prob * stream_b_weight
        
        results = {
            'correlation_score': correlation_score,
            'confidence_score': confidence_score,
            'uncertainty_score': uncertainty_score,
            'stream_a_weight': stream_a_weight,
            'stream_b_weight': stream_b_weight,
            'stream_a_contribution': stream_a_contribution,
            'stream_b_contribution': stream_b_contribution,
            'combined_representation': fused_representation
        }
        
        return results


class AdaptiveThresholdManager(nn.Module):
    """Dynamic threshold adjustment based on false positive rates."""
    
    def __init__(self, 
                 initial_threshold: float = 0.5,
                 adaptation_rate: float = 0.01,
                 min_threshold: float = 0.1,
                 max_threshold: float = 0.9,
                 window_size: int = 100):
        super().__init__()
        self.initial_threshold = initial_threshold
        self.adaptation_rate = adaptation_rate
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.window_size = window_size
        
        # Current threshold (trainable parameter)
        self.current_threshold = nn.Parameter(torch.tensor(initial_threshold))
        
        # Performance tracking
        self.register_buffer('false_positive_history', torch.zeros(window_size))
        self.register_buffer('true_positive_history', torch.zeros(window_size))
        self.register_buffer('detection_rate_history', torch.zeros(window_size))
        self.register_buffer('current_index', torch.tensor(0))
        
        # Target performance metrics
        self.target_false_positive_rate = 0.05  # 5%
        self.target_detection_rate = 0.95       # 95%
        
    def update_performance_metrics(self, 
                                   predictions: torch.Tensor,
                                   labels: torch.Tensor,
                                   is_attack: torch.Tensor):
        """Update performance tracking with new batch results."""
        # Calculate batch metrics
        tp = ((predictions == 1) & (is_attack == 1)).sum().float()
        fp = ((predictions == 1) & (is_attack == 0)).sum().float()
        fn = ((predictions == 0) & (is_attack == 1)).sum().float()
        
        # Calculate rates
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        fpr = fp / ((is_attack == 0).sum().float() + 1e-6)
        
        # Update history buffers
        idx = int(self.current_index.item()) % self.window_size
        
        self.false_positive_history[idx] = fpr
        self.true_positive_history[idx] = precision
        self.detection_rate_history[idx] = recall
        
        self.current_index = (self.current_index + 1) % self.window_size
        
        return precision, recall, fpr
    
    def adaptive_adjustment(self) -> torch.Tensor:
        """Dynamically adjust threshold based on performance history."""
        # Calculate moving averages
        avg_fpr = torch.mean(self.false_positive_history)
        avg_recall = torch.mean(self.detection_rate_history)
        
        # Calculate adjustment factors
        fpr_error = self.target_false_positive_rate - avg_fpr
        recall_error = self.target_detection_rate - avg_recall
        
        # Combined adjustment signal
        adjustment_signal = fpr_error + 0.5 * recall_error
        
        # Apply adjustment with bounded learning rate
        threshold_update = self.adaptation_rate * adjustment_signal
        new_threshold = self.current_threshold + threshold_update
        
        # Clamp to valid range
        bounded_threshold = torch.clamp(new_threshold, self.min_threshold, self.max_threshold)
        
        # Update threshold (only if change is significant)
        if torch.abs(bounded_threshold - self.current_threshold) > 0.001:
            self.current_threshold.data = bounded_threshold
        
        return self.current_threshold
    
    def get_threshold(self) -> float:
        """Get current threshold value."""
        return self.current_threshold.item()


class ThreatPrioritizer(nn.Module):
    """CRITICAL/HIGH/MEDIUM/LOW threat prioritization system."""
    
    def __init__(self, num_priority_levels: int = 4):
        super().__init__()
        self.num_priority_levels = num_priority_levels
        
        # Priority classifier - FIX: Input is 3 features (correlation + confidence + uncertainty)
        self.priority_classifier = nn.Sequential(
            nn.Linear(3, 32),  # 3 input features: correlation_score + confidence + uncertainty
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_priority_levels),
            nn.LogSoftmax(dim=-1)
        )
        
        # Priority level mapping
        self.priority_levels = [
            ThreatLevel.CRITICAL,
            ThreatLevel.HIGH,
            ThreatLevel.MEDIUM,
            ThreatLevel.LOW
        ]
        
        # Priority thresholds (can be learned or fixed)
        self.register_buffer('priority_thresholds', 
                           torch.tensor([0.8, 0.6, 0.4, 0.0]))
    
    def calculate_priority(self, 
                          correlation_score: torch.Tensor,
                          confidence_score: torch.Tensor,
                          uncertainty_score: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate threat priority level.
        
        Args:
            correlation_score: Combined anomaly/classification score
            confidence_score: Model confidence in prediction
            uncertainty_score: Prediction uncertainty
            
        Returns:
            priority_logits: Log probabilities for priority levels
            priority_levels: Predicted priority levels
        """
        # Combine features for priority classification
        priority_input = torch.cat([
            correlation_score,
            confidence_score,
            uncertainty_score
        ], dim=-1)
        
        priority_logits = self.priority_classifier(priority_input)
        priority_levels = torch.argmax(priority_logits, dim=-1)
        
        return priority_logits, priority_levels
    
    def get_priority_description(self, priority_level: ThreatLevel) -> str:
        """Get human-readable description for priority level."""
        descriptions = {
            ThreatLevel.CRITICAL: "Immediate threat requiring urgent response",
            ThreatLevel.HIGH: "Significant threat requiring prompt attention",
            ThreatLevel.MODERATE: "Moderate threat requiring monitoring",
            ThreatLevel.LOW: "Low priority threat or false positive"
        }
        return descriptions.get(priority_level, "Unknown priority level")


class CorrelationEngine(nn.Module):
    """Complete correlation engine for dual-stream signal fusion."""
    
    def __init__(self,
                 autoencoder_dim: int = 1,
                 tagn_dim: int = 16,
                 confidence_dim: int = 1,
                 hidden_dim: int = 64,
                 initial_threshold: float = 0.5):
        super().__init__()
        
        # Core components
        self.signal_correlator = WeightedSignalCorrelator(
            autoencoder_dim, tagn_dim, confidence_dim, hidden_dim
        )
        
        self.threshold_manager = AdaptiveThresholdManager(
            initial_threshold=initial_threshold
        )
        
        self.threat_prioritizer = ThreatPrioritizer()
        
        # Decision rationale generator
        # FIX: Input is 64 (hidden_dim) + 3 (correlation + confidence + uncertainty) = 67
        self.rationale_generator = nn.Sequential(
            nn.Linear(67, 128),  # 64 (combined_representation) + 3 (scores)
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, 
                autoencoder_results: Dict[str, torch.Tensor],
                tagn_results: Dict[str, torch.Tensor],
                performance_feedback: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Complete correlation engine forward pass.
        
        Args:
            autoencoder_results: Results from Stream A (autoencoder)
                - 'anomaly_score': Reconstruction error
                - 'is_anomaly': Boolean anomaly flag
            tagn_results: Results from Stream B (TAGN network)
                - 'classification': Dict with class_probabilities and confidence_score
                - 'priority_scores': Priority level scores
            performance_feedback: Optional performance metrics for adaptation
            
        Returns:
            Dictionary containing all correlation results
        """
        # Extract Stream A features - ensure correct shape
        anomaly_scores = autoencoder_results['anomaly_score']
        if anomaly_scores.dim() == 1:
            anomaly_scores = anomaly_scores.unsqueeze(-1)  # (batch,) -> (batch, 1)
        elif anomaly_scores.dim() == 2 and anomaly_scores.size(1) != 1:
            anomaly_scores = anomaly_scores.mean(dim=1, keepdim=True)  # (batch, n) -> (batch, 1)
        
        # Extract Stream B features - ensure correct shape
        threat_probabilities = tagn_results['classification']['class_probabilities']
        if threat_probabilities.dim() == 3:
            threat_probabilities = threat_probabilities.squeeze(1)  # (batch, 1, 16) -> (batch, 16)
        
        confidence_scores = tagn_results['classification']['confidence_score']
        if confidence_scores.dim() == 1:
            confidence_scores = confidence_scores.unsqueeze(-1)  # (batch,) -> (batch, 1)
        elif confidence_scores.dim() == 3:
            confidence_scores = confidence_scores.squeeze(1)  # (batch, 1, 1) -> (batch, 1)
        
        # Update performance metrics if feedback provided
        if performance_feedback is not None:
            self.threshold_manager.update_performance_metrics(
                predictions=performance_feedback['predictions'],
                labels=performance_feedback['labels'],
                is_attack=performance_feedback['is_attack']
            )
        
        # Apply adaptive threshold
        current_threshold = self.threshold_manager.adaptive_adjustment()
        
        # Correlate signals
        correlation_results = self.signal_correlator(
            anomaly_scores, threat_probabilities, confidence_scores
        )
        
        # Calculate threat priority
        priority_logits, priority_levels = self.threat_prioritizer.calculate_priority(
            correlation_results['correlation_score'],
            correlation_results['confidence_score'],
            correlation_results['uncertainty_score']
        )
        
        # Generate decision rationale
        # FIX: Ensure all tensors have correct dimensions before concatenation
        combined_rep = correlation_results['combined_representation']  # (batch, 64)
        corr_score = correlation_results['correlation_score']  # (batch, 1)
        conf_score = correlation_results['confidence_score']  # (batch, 1)
        uncert_score = correlation_results['uncertainty_score']  # (batch, 1)
        
        # Total: 64 + 1 + 1 + 1 = 67 features
        rationale_input = torch.cat([
            combined_rep,
            corr_score,
            conf_score,
            uncert_score
        ], dim=-1)
        
        decision_rationale_score = self.rationale_generator(rationale_input)
        
        # Compile final results
        final_results = {
            # Core correlation metrics
            'correlation_score': correlation_results['correlation_score'],
            'confidence_score': correlation_results['confidence_score'],
            'uncertainty_score': correlation_results['uncertainty_score'],
            'current_threshold': current_threshold,
            
            # Stream contributions
            'stream_a_contribution': correlation_results['stream_a_contribution'],
            'stream_b_contribution': correlation_results['stream_b_contribution'],
            'stream_a_weight': correlation_results['stream_a_weight'],
            'stream_b_weight': correlation_results['stream_b_weight'],
            
            # Threat prioritization
            'priority_logits': priority_logits,
            'priority_levels': priority_levels,
            'priority_scores': tagn_results['priority_scores'],
            
            # Decision rationale
            'decision_rationale_score': decision_rationale_score,
            
            # Original inputs for reference
            'anomaly_scores': anomaly_scores.squeeze(-1),
            'threat_probabilities': threat_probabilities,
            'confidence_scores': confidence_scores,
            
            # Performance tracking
            'is_anomaly': correlation_results['correlation_score'] > current_threshold,
            'performance_metrics': {
                'target_fpr': self.threshold_manager.target_false_positive_rate,
                'target_detection_rate': self.threshold_manager.target_detection_rate,
                'adaptive_threshold': current_threshold.item()
            }
        }
        
        return final_results
    
    def generate_correlation_result(self, batch_index: int, results: Dict[str, torch.Tensor]) -> List[CorrelationResult]:
        """Generate structured correlation results for a batch."""
        batch_size = results['correlation_score'].size(0)
        correlation_results = []
        
        for i in range(batch_size):
            # Convert priority level to enum
            priority_level = self.threat_prioritizer.priority_levels[
                results['priority_levels'][i].item()
            ]
            
            # Generate rationale
            rationale = self._generate_rationale_text(
                results['correlation_score'][i].item(),
                results['priority_levels'][i].item(),
                results['decision_rationale_score'][i].item()
            )
            
            correlation_result = CorrelationResult(
                correlation_score=results['correlation_score'][i].item(),
                confidence_score=results['confidence_score'][i].item(),
                threat_level=priority_level,
                stream_a_contribution=results['stream_a_contribution'][i].item(),
                stream_b_contribution=results['stream_b_contribution'][i].item(),
                uncertainty_estimate=results['uncertainty_score'][i].item(),
                decision_rationale=rationale
            )
            
            correlation_results.append(correlation_result)
        
        return correlation_results
    
    def _generate_rationale_text(self, 
                                correlation_score: float,
                                priority_level: int,
                                rationale_score: float) -> str:
        """Generate human-readable rationale for the decision."""
        base_rationale = f"Correlation score: {correlation_score:.3f}"
        
        if priority_level == 0:  # CRITICAL
            return f"{base_rationale} - Critical threat detected. High correlation and confidence indicate immediate security incident."
        elif priority_level == 1:  # HIGH
            return f"{base_rationale} - High-priority threat. Strong anomaly patterns detected requiring prompt attention."
        elif priority_level == 2:  # MEDIUM
            return f"{base_rationale} - Medium-priority alert. Suspicious activity detected, monitor for escalation."
        else:  # LOW
            return f"{base_rationale} - Low-priority alert. Possible false positive or minor anomaly detected."
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get current performance statistics for monitoring."""
        avg_fpr = torch.mean(self.threshold_manager.false_positive_history).item()
        avg_recall = torch.mean(self.threshold_manager.detection_rate_history).item()
        
        return {
            'current_threshold': self.threshold_manager.get_threshold(),
            'avg_false_positive_rate': avg_fpr,
            'avg_detection_rate': avg_recall,
            'threshold_deviation_from_target': abs(
                self.threshold_manager.get_threshold() - self.threshold_manager.initial_threshold
            )
        }


def create_correlation_engine(autoencoder_dim: int = 1,
                             tagn_dim: int = 16,
                             confidence_dim: int = 1,
                             hidden_dim: int = 64) -> CorrelationEngine:
    """
    Factory function to create a correlation engine.
    
    Args:
        autoencoder_dim: Dimension of autoencoder output (usually 1 for anomaly score)
        tagn_dim: Dimension of TAGN correlation_features output (16-dim)
        confidence_dim: Dimension of confidence scores (usually 1)
        hidden_dim: Hidden layer dimensions for neural networks
        
    Returns:
        Configured CorrelationEngine instance
    """
    engine = CorrelationEngine(
        autoencoder_dim=autoencoder_dim,
        tagn_dim=tagn_dim,
        confidence_dim=confidence_dim,
        hidden_dim=hidden_dim
    )
    
    return engine


if __name__ == "__main__":
    # Test the correlation engine implementation
    print("🧪 Testing Correlation Engine Implementation...")
    
    # Create sample data
    batch_size = 4
    
    # Sample autoencoder results (Stream A)
    autoencoder_results = {
        'anomaly_score': torch.tensor([0.8, 0.3, 0.9, 0.2]),  # Reconstruction errors
        'is_anomaly': torch.tensor([True, False, True, False])
    }
    
    # Sample TAGN results (Stream B) - using correlation_features (16-dim)
    tagn_results = {
        'classification': {
            'class_probabilities': torch.randn(batch_size, 16),  # correlation_features
            'confidence_score': torch.tensor([0.85, 0.60, 0.90, 0.40])
        },
        'priority_scores': torch.randn(batch_size, 4)
    }
    
    # Sample performance feedback
    performance_feedback = {
        'predictions': torch.tensor([1, 0, 1, 0]),  # 1=attack, 0=benign
        'labels': torch.tensor([0, 0, 1, 0]),
        'is_attack': torch.tensor([False, False, True, False])
    }
    
    # Create and test correlation engine
    engine = create_correlation_engine()
    
    print(f"📊 Engine Parameters: {sum(p.numel() for p in engine.parameters()):,}")
    
    # Forward pass
    results = engine(autoencoder_results, tagn_results, performance_feedback)
    
    # Display results
    print(f"✅ Correlation scores: {results['correlation_score'].squeeze().numpy()}")
    print(f"✅ Confidence scores: {results['confidence_score'].squeeze().numpy()}")
    print(f"✅ Stream A weights: {results['stream_a_weight'].squeeze().numpy()}")
    print(f"✅ Stream B weights: {results['stream_b_weight'].squeeze().numpy()}")
    print(f"✅ Priority levels: {results['priority_levels'].numpy()}")
    
    # Generate correlation results
    correlation_results = engine.generate_correlation_result(0, results)
    print(f"🎯 Sample rationale: {correlation_results[0].decision_rationale}")
    
    # Show performance stats
    stats = engine.get_performance_stats()
    print(f"📈 Performance stats: {stats}")
    
    print("🎉 Correlation Engine test completed successfully!")