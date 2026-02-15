"""
LLM-Powered Alert Generation — Phase 4.
Algorithm Steps 22-24:
  22. Phase 4: LLM-Powered Alert Generation
  23. Assemble prompt P using V, S, C, and context; pass to LLM
  24. Generate readable alert A with summary, impact, and recommended actions

This version:
  - Constructs an actual prompt string from flow data, anomaly score,
    classification result, and context.
  - Attempts to call an LLM API (OpenAI-compatible) if configured.
  - Falls back to a comprehensive rule-based engine when no API is available.
  - Produces structured JSON alerts with multi-class threat types.
"""

import json, time, os, logging, hashlib
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)

# Re-use the label list from TAGN
THREAT_LABELS = [
    "BENIGN", "DDoS", "PortScan", "Web Attack",
    "Infiltration", "Botnet", "Probe",
]


class PriorityLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"


# ──────────────────────────────────────────────────────────────────────────────
# Structured alert
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SecurityAlert:
    alert_id: str
    timestamp: str
    priority: str
    threat_type: str
    threat_class_index: int
    class_probabilities: Dict[str, float]
    anomaly_score: float
    confidence_score: float
    fusion_score: float
    summary: str
    impact: str
    recommended_actions: List[str]
    evidence: Dict[str, Any] = field(default_factory=dict)
    llm_enhanced: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# Rule-based knowledge base (fallback when no LLM available)
# ──────────────────────────────────────────────────────────────────────────────

_KB = {
    "DDoS": {
        "summary": "Distributed Denial of Service attack detected — high-volume traffic aimed at exhausting target resources.",
        "impact": "Service disruption, potential downtime, degraded performance for legitimate users.",
        "actions": [
            "Activate DDoS mitigation / rate-limiting on the target",
            "Block or null-route source IPs via firewall",
            "Engage upstream ISP scrubbing if volumetric",
            "Monitor bandwidth utilisation in real time",
            "Preserve flow logs for forensic analysis",
        ],
    },
    "PortScan": {
        "summary": "Systematic port-scanning / reconnaissance activity detected.",
        "impact": "Information leakage about open services; likely precursor to exploitation.",
        "actions": [
            "Block scanning source IPs",
            "Review exposed services and close unnecessary ports",
            "Update IDS/IPS signatures for follow-up attacks",
            "Monitor the source for escalation to exploitation",
        ],
    },
    "Web Attack": {
        "summary": "Web application attack detected (possible SQLi, XSS, or command injection).",
        "impact": "Risk of data exfiltration, session hijacking, or remote code execution.",
        "actions": [
            "Block attacker IP at WAF / reverse-proxy",
            "Review web-server access logs for IOCs",
            "Patch or virtually-patch the vulnerable endpoint",
            "Rotate any potentially compromised credentials",
        ],
    },
    "Infiltration": {
        "summary": "Network infiltration indicators detected — possible lateral movement or privilege escalation.",
        "impact": "Full system compromise, data theft, persistence establishment.",
        "actions": [
            "Isolate affected hosts immediately",
            "Reset credentials for compromised accounts",
            "Conduct memory / disk forensics",
            "Check for persistence mechanisms (scheduled tasks, services)",
        ],
    },
    "Botnet": {
        "summary": "Botnet command-and-control communication patterns detected.",
        "impact": "Compromised hosts may be used for spam, DDoS amplification, or data theft.",
        "actions": [
            "Block C2 domains / IPs at DNS and firewall",
            "Quarantine and re-image infected hosts",
            "Scan the network for additional infections",
            "Update endpoint protection signatures",
        ],
    },
    "Probe": {
        "summary": "Network reconnaissance / probing activity detected.",
        "impact": "Attacker gathering topology and service information for planned exploitation.",
        "actions": [
            "Monitor for follow-up exploitation attempts",
            "Verify network segmentation effectiveness",
            "Conduct a proactive vulnerability scan",
        ],
    },
    "BENIGN": {
        "summary": "Traffic classified as benign — no threat detected.",
        "impact": "None.",
        "actions": ["Continue routine monitoring."],
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Prompt builder (for real LLM calls)
# ──────────────────────────────────────────────────────────────────────────────

def build_llm_prompt(
    anomaly_score: float,
    threat_label: str,
    class_probs: Dict[str, float],
    confidence: float,
    priority: str,
    flow_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Assemble prompt P using V, S, C, and context (Algorithm step 23)."""
    ctx = json.dumps(flow_context, indent=2) if flow_context else "N/A"
    prob_str = ", ".join(f"{k}: {v:.3f}" for k, v in class_probs.items())

    prompt = (
        "You are a cybersecurity analyst AI. Given the following network intrusion "
        "detection output, produce a JSON object with keys: summary, impact, "
        "recommended_actions (list of strings).\n\n"
        f"Anomaly Score (reconstruction error): {anomaly_score:.6f}\n"
        f"Threat Classification: {threat_label}\n"
        f"Class Probabilities: {prob_str}\n"
        f"Model Confidence: {confidence:.4f}\n"
        f"Assigned Priority: {priority}\n"
        f"Flow Context:\n{ctx}\n\n"
        "Respond ONLY with a valid JSON object, no markdown fences."
    )
    return prompt


# ──────────────────────────────────────────────────────────────────────────────
# Alert generator
# ──────────────────────────────────────────────────────────────────────────────

_counter = 0

class AlertGenerator:
    """
    Generates SecurityAlert objects.

    If an OpenAI-compatible API key + endpoint are set, it will call the LLM
    to produce the summary / impact / actions.  Otherwise it falls back to
    the local knowledge base.
    """

    def __init__(
        self,
        prefix: str = "AGILE",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
    ):
        self.prefix = prefix
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.model = model
        global _counter
        _counter = 0

    # ---- LLM call (optional) ----
    def _try_llm(self, prompt: str) -> Optional[Dict[str, Any]]:
        if not self.api_key:
            return None
        try:
            import requests
            resp = requests.post(
                f"{self.api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 512,
                },
                timeout=10,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            return json.loads(text.strip().strip("`").strip())
        except Exception as e:
            logger.warning("LLM call failed, falling back to rule-based: %s", e)
            return None

    # ---- main entry point ----
    def generate(
        self,
        anomaly_score: float,
        predicted_class: int,
        class_probs: Dict[str, float],
        confidence: float,
        fusion_score: float,
        priority: str,
        flow_context: Optional[Dict[str, Any]] = None,
    ) -> SecurityAlert:
        global _counter
        _counter += 1
        alert_id = f"{self.prefix}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{_counter:05d}"

        threat_label = THREAT_LABELS[predicted_class] if predicted_class < len(THREAT_LABELS) else "Unknown"

        # Try LLM first (step 23-24)
        prompt = build_llm_prompt(
            anomaly_score, threat_label, class_probs, confidence, priority, flow_context,
        )
        llm_result = self._try_llm(prompt)

        if llm_result:
            summary = llm_result.get("summary", "")
            impact  = llm_result.get("impact", "")
            actions = llm_result.get("recommended_actions", [])
            enhanced = True
        else:
            kb = _KB.get(threat_label, _KB["BENIGN"])
            summary = kb["summary"]
            impact  = kb["impact"]
            actions = kb["actions"]
            enhanced = False

        return SecurityAlert(
            alert_id=alert_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            priority=priority,
            threat_type=threat_label,
            threat_class_index=predicted_class,
            class_probabilities=class_probs,
            anomaly_score=anomaly_score,
            confidence_score=confidence,
            fusion_score=fusion_score,
            summary=summary,
            impact=impact,
            recommended_actions=actions,
            evidence={
                "anomaly_above_threshold": bool(anomaly_score > 0.1),
                "top_class": threat_label,
                "top_prob": float(max(class_probs.values())),
            },
            llm_enhanced=enhanced,
        )
