"""
Correlation Engine -- Phase 3: Correlation & Validation.
Algorithm Steps 12-21 (explicit rule-based priority assignment):

    if S > threshold  OR  C != BENIGN:
        if S high  AND  known_threat:   -> CRITICAL
        elif S high AND  C == BENIGN:   -> HIGH
        else:                           -> MEDIUM
    else:
        -> (no alert -- normal traffic)

v2.1 improvements:
  - Confidence gating: TAGN predictions below a confidence threshold are
    reverted to BENIGN before entering the priority rules. This drastically
    reduces false positives from low-confidence misclassifications.
  - TAGN-primary decision: the TAGN classification (with confidence gate)
    is the primary signal; anomaly score acts as a secondary amplifier
    for priority assignment, not for the binary attack/benign decision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from enum import IntEnum


# -----------------------------------------------------------------------
# Priority levels (algorithm lines 15-19)
# -----------------------------------------------------------------------

class Priority(IntEnum):
    CRITICAL = 0
    HIGH     = 1
    MEDIUM   = 2
    LOW      = 3      # implicit in algorithm -- normal traffic


# -----------------------------------------------------------------------
# Learned signal correlator (unchanged -- used for fusion score training)
# -----------------------------------------------------------------------

class SignalCorrelator(nn.Module):
    """
    Learns a single correlation/fusion score from:
        - anomaly_score  (1-d)        from Stream A
        - corr_features  (16-d)       from Stream B's correlation_features
        - confidence      (1-d)       from Stream B
    Total input = 18.
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(18, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        anomaly_score: torch.Tensor,       # (B, 1)
        corr_features: torch.Tensor,       # (B, 16)
        confidence: torch.Tensor,          # (B, 1)
    ) -> torch.Tensor:
        x = torch.cat([anomaly_score, corr_features, confidence], dim=-1)
        return self.net(x)


# -----------------------------------------------------------------------
# Rule-based priority assigner WITH confidence gating
# -----------------------------------------------------------------------

def assign_priority_rules(
    anomaly_score: torch.Tensor,           # (B,)
    predicted_class: torch.Tensor,         # (B,)  -- 0=BENIGN, 1..6=attacks
    confidence: torch.Tensor,             # (B,)  -- TAGN confidence score
    anomaly_threshold: float,
    confidence_threshold: float = 0.70,
    high_anomaly_factor: float = 2.0,
) -> tuple:
    """
    Rule-based priority assignment following Algorithm 1 lines 13-21,
    with confidence gating to reduce false positives.

    Confidence gate: if TAGN predicts attack but confidence < threshold,
    revert the prediction to BENIGN (class 0). This filters out the many
    low-confidence misclassifications that inflate FPR.

    Returns:
        priority: (B,) long tensor -- 0=CRITICAL, 1=HIGH, 2=MEDIUM, 3=LOW
        gated_class: (B,) long tensor -- class after confidence gating
    """
    B = anomaly_score.size(0)

    # --- Confidence gating ---
    # Revert low-confidence attack predictions to BENIGN
    gated_class = predicted_class.clone()
    low_conf_attack = (predicted_class != 0) & (confidence < confidence_threshold)
    gated_class[low_conf_attack] = 0

    # --- Priority assignment (algorithm lines 13-21) ---
    priority = torch.full((B,), Priority.LOW, dtype=torch.long,
                          device=anomaly_score.device)

    s_above  = anomaly_score > anomaly_threshold        # S > threshold
    c_attack = gated_class != 0                         # C non-benign (after gating)
    s_high   = anomaly_score > (anomaly_threshold * high_anomaly_factor)

    trigger = s_above | c_attack                        # line 13

    # line 14-15: S high AND known threat -> CRITICAL
    crit = trigger & s_high & c_attack
    priority[crit] = Priority.CRITICAL

    # line 16-17: S high AND C == Benign -> HIGH
    high = trigger & s_high & (~c_attack)
    priority[high] = Priority.HIGH

    # line 18-19: else (trigger but not high) -> MEDIUM
    med = trigger & (~s_high)
    priority[med] = Priority.MEDIUM

    # everything else stays LOW (no alert)
    return priority, gated_class


# -----------------------------------------------------------------------
# Complete Correlation Engine
# -----------------------------------------------------------------------

class CorrelationEngine(nn.Module):
    """
    Combines:
      - A learned SignalCorrelator (produces a fusion score for training)
      - The exact rule-based priority logic from the algorithm
      - Confidence gating to reduce false positives
    """

    def __init__(self, hidden: int = 64, anomaly_threshold: float = 0.5,
                 confidence_threshold: float = 0.70):
        super().__init__()
        self.correlator = SignalCorrelator(hidden)
        self.anomaly_threshold = anomaly_threshold
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def _ensure_2d(t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            return t.unsqueeze(-1)
        return t

    def forward(
        self,
        anomaly_score: torch.Tensor,       # (B,) or (B,1)
        corr_features: torch.Tensor,       # (B, 16)
        confidence: torch.Tensor,          # (B,) or (B,1)
        predicted_class: torch.Tensor,     # (B,)  long -- 0=BENIGN,1..6=attacks
    ) -> Dict[str, torch.Tensor]:

        a = self._ensure_2d(anomaly_score)        # (B, 1)
        c = self._ensure_2d(confidence)            # (B, 1)

        # learned fusion score
        fusion_score = self.correlator(a, corr_features, c)      # (B, 1)

        # rule-based priority with confidence gating
        a_flat = a.squeeze(-1)                                    # (B,)
        c_flat = c.squeeze(-1)                                    # (B,)
        priority, gated_class = assign_priority_rules(
            a_flat, predicted_class, c_flat,
            anomaly_threshold=self.anomaly_threshold,
            confidence_threshold=self.confidence_threshold,
        )

        is_alert = priority < Priority.LOW

        return {
            "fusion_score":    fusion_score,         # (B, 1) -- trainable
            "priority":        priority,             # (B,) long
            "is_alert":        is_alert,             # (B,) bool
            "anomaly_score":   a_flat,               # (B,)
            "gated_class":     gated_class,          # (B,) -- class after confidence gating
        }


def create_correlation_engine(hidden: int = 64, anomaly_threshold: float = 0.5,
                              confidence_threshold: float = 0.70):
    return CorrelationEngine(hidden=hidden, anomaly_threshold=anomaly_threshold,
                             confidence_threshold=confidence_threshold)
