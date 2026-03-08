"""
LLM-Powered Alert Generation v3 — Phase 4.
Extended knowledge base for all 15 CICIDS2017 classes.
All other logic identical to models_v2/llm_intelligence.py.
"""

import json, os, logging
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)

# v3 15-class label list
THREAT_LABELS = [
    "BENIGN",           #  0
    "Bot",              #  1
    "DDoS",             #  2
    "DoS GoldenEye",    #  3
    "DoS Hulk",         #  4
    "DoS Slowhttptest", #  5
    "DoS Slowloris",    #  6
    "FTP-Patator",      #  7
    "Heartbleed",       #  8
    "Infiltration",     #  9
    "PortScan",         # 10
    "SSH-Patator",      # 11
    "Brute Force",      # 12
    "SQL Injection",    # 13
    "XSS",              # 14
]


class PriorityLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"


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

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)


# Full 15-class knowledge base
_KB = {
    "BENIGN": {
        "summary": "Traffic classified as benign — no threat detected.",
        "impact": "None.",
        "actions": ["Continue routine monitoring."],
    },
    "Bot": {
        "summary": "Botnet C2 communication patterns detected — host may be compromised and part of a botnet.",
        "impact": "Compromised hosts used for spam, DDoS amplification, credential theft, or cryptomining.",
        "actions": [
            "Block C2 domains and IPs at DNS and perimeter firewall",
            "Quarantine and re-image infected hosts",
            "Scan the full subnet for lateral spread",
            "Update endpoint protection and run full AV scan",
            "Review outbound traffic logs for data exfiltration indicators",
        ],
    },
    "DDoS": {
        "summary": "Distributed Denial of Service attack detected — high-volume traffic exhausting target resources.",
        "impact": "Service disruption, downtime, degraded performance for legitimate users.",
        "actions": [
            "Activate DDoS mitigation and rate-limiting on the target",
            "Block or null-route source IPs at the firewall",
            "Engage upstream ISP scrubbing if volumetric",
            "Monitor bandwidth utilisation in real time",
            "Preserve flow logs for forensic analysis",
        ],
    },
    "DoS GoldenEye": {
        "summary": "DoS GoldenEye HTTP-layer attack detected — targets HTTP keep-alive to exhaust web server connections.",
        "impact": "Web server unavailability, denial of service for HTTP/HTTPS endpoints.",
        "actions": [
            "Enable connection rate-limiting on the web server",
            "Block source IPs at WAF or reverse proxy",
            "Increase keep-alive timeout limits as a short-term mitigation",
            "Enable SYN cookies on the load balancer",
        ],
    },
    "DoS Hulk": {
        "summary": "DoS Hulk HTTP flood attack detected — generates unique randomised HTTP GET requests to bypass caching.",
        "impact": "Web server CPU exhaustion, service unavailability.",
        "actions": [
            "Enable request-rate limiting and bot detection at the WAF",
            "Block offending IPs or ASNs at the firewall",
            "Enable CAPTCHA challenges for suspicious IP ranges",
            "Scale out web server capacity temporarily",
        ],
    },
    "DoS Slowhttptest": {
        "summary": "Slow HTTP DoS attack (Slowhttptest) detected — sends partial HTTP requests to keep connections open.",
        "impact": "Web server connection pool exhaustion; legitimate users cannot connect.",
        "actions": [
            "Lower connection timeout thresholds on the web server",
            "Block source IPs at the firewall or load balancer",
            "Enable slow-client detection at the reverse proxy (e.g., mod_reqtimeout)",
            "Increase the maximum number of allowed connections as a short-term measure",
        ],
    },
    "DoS Slowloris": {
        "summary": "Slowloris DoS attack detected — holds connections open with partial HTTP headers to exhaust server threads.",
        "impact": "Web server thread pool exhaustion; service degradation or unavailability.",
        "actions": [
            "Enable Slowloris protection in the web server configuration",
            "Set aggressive connection timeout policies",
            "Block source IPs at the perimeter firewall",
            "Consider switching to an event-driven web server (nginx) if using Apache",
        ],
    },
    "FTP-Patator": {
        "summary": "FTP brute-force attack (Patator) detected — automated credential guessing against FTP service.",
        "impact": "Risk of unauthorised FTP access, data theft, or malware upload.",
        "actions": [
            "Block source IP at the firewall immediately",
            "Enforce account lockout after failed login attempts",
            "Disable anonymous FTP access if enabled",
            "Consider replacing FTP with SFTP or FTPS",
            "Audit FTP access logs for any successful logins during the attack window",
        ],
    },
    "Heartbleed": {
        "summary": "Heartbleed (CVE-2014-0160) exploitation attempt detected — attacker reading server memory via malformed TLS heartbeat.",
        "impact": "Leakage of private keys, session tokens, passwords, and sensitive memory contents.",
        "actions": [
            "Patch OpenSSL to version 1.0.1g or later IMMEDIATELY",
            "Revoke and reissue all TLS certificates on affected servers",
            "Force password resets for all users who may have had sessions during the attack",
            "Rotate all session tokens and API keys",
            "Audit server memory dumps for leaked credentials",
        ],
    },
    "Infiltration": {
        "summary": "Network infiltration indicators detected — possible lateral movement, privilege escalation, or C2 beacon.",
        "impact": "Full system compromise, persistent access, data exfiltration, ransomware staging.",
        "actions": [
            "Isolate affected hosts from the network immediately",
            "Reset all credentials for accounts on affected systems",
            "Conduct memory and disk forensics on isolated hosts",
            "Check for persistence mechanisms: scheduled tasks, registry run keys, services",
            "Review DNS and proxy logs for C2 callback domains",
        ],
    },
    "PortScan": {
        "summary": "Systematic port-scanning / network reconnaissance detected — attacker mapping open services.",
        "impact": "Information leakage about open ports and services; likely precursor to targeted exploitation.",
        "actions": [
            "Block scanning source IPs at the firewall",
            "Review exposed services and close unnecessary ports",
            "Update IDS/IPS signatures to catch follow-up exploitation attempts",
            "Monitor the source IP for escalation to active exploitation",
        ],
    },
    "SSH-Patator": {
        "summary": "SSH brute-force attack (Patator) detected — automated credential guessing against SSH service.",
        "impact": "Risk of unauthorised shell access, privilege escalation, and persistent backdoor installation.",
        "actions": [
            "Block source IP at the firewall immediately",
            "Enforce SSH key-based authentication and disable password login",
            "Implement fail2ban or equivalent rate-limiting on SSH",
            "Audit SSH auth logs for any successful logins during the attack window",
            "Consider moving SSH to a non-standard port as an additional deterrent",
        ],
    },
    "Brute Force": {
        "summary": "Web application brute-force attack detected — automated credential guessing against a login endpoint.",
        "impact": "Risk of account takeover, unauthorised data access, and lateral movement via compromised credentials.",
        "actions": [
            "Block attacker IP at the WAF or reverse proxy",
            "Enable account lockout and CAPTCHA on the login form",
            "Force password resets for targeted accounts",
            "Enable multi-factor authentication on affected endpoints",
            "Review application logs for successful logins from the attack source",
        ],
    },
    "SQL Injection": {
        "summary": "SQL injection attack detected — attacker injecting malicious SQL into web application input fields.",
        "impact": "Database data exfiltration, authentication bypass, data modification, or remote code execution.",
        "actions": [
            "Block attacker IP at the WAF immediately",
            "Review and patch the vulnerable query/endpoint using parameterised queries",
            "Audit database access logs for data exfiltration",
            "Rotate database credentials",
            "Run a full application security scan to identify other injection points",
        ],
    },
    "XSS": {
        "summary": "Cross-Site Scripting (XSS) attack detected — attacker injecting malicious scripts into web pages.",
        "impact": "Session hijacking, credential theft, malicious redirects, defacement.",
        "actions": [
            "Block attacker IP at the WAF",
            "Patch the vulnerable input/output endpoint with proper HTML encoding",
            "Implement a Content Security Policy (CSP) header",
            "Invalidate active sessions for users who may have been affected",
            "Conduct a full XSS audit of the application",
        ],
    },
}


def build_llm_prompt(anomaly_score, threat_label, class_probs, confidence, priority, flow_context=None):
    ctx = json.dumps(flow_context, indent=2) if flow_context else "N/A"
    prob_str = ", ".join(f"{k}: {v:.3f}" for k, v in class_probs.items())
    return (
        "You are a cybersecurity analyst AI. Given the following network intrusion "
        "detection output, produce a JSON object with keys: summary, impact, "
        "recommended_actions (list of strings).\n\n"
        f"Anomaly Score: {anomaly_score:.6f}\n"
        f"Threat Classification: {threat_label}\n"
        f"Class Probabilities: {prob_str}\n"
        f"Model Confidence: {confidence:.4f}\n"
        f"Assigned Priority: {priority}\n"
        f"Flow Context:\n{ctx}\n\n"
        "Respond ONLY with a valid JSON object, no markdown fences."
    )


_counter = 0


class AlertGenerator:
    def __init__(self, prefix="AGILE", api_key=None, api_base=None, model="gpt-3.5-turbo"):
        self.prefix   = prefix
        self.api_key  = api_key  or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.model    = model
        global _counter
        _counter = 0

    def _try_llm(self, prompt):
        if not self.api_key:
            return None
        try:
            import requests
            resp = requests.post(
                f"{self.api_base}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": self.model, "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0.3, "max_tokens": 512},
                timeout=10,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            return json.loads(text.strip().strip("`").strip())
        except Exception as e:
            logger.warning("LLM call failed: %s", e)
            return None

    def generate(self, anomaly_score, predicted_class, class_probs, confidence,
                 fusion_score, priority, flow_context=None):
        global _counter
        _counter += 1
        alert_id     = f"{self.prefix}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{_counter:05d}"
        threat_label = THREAT_LABELS[predicted_class] if predicted_class < len(THREAT_LABELS) else "Unknown"

        prompt     = build_llm_prompt(anomaly_score, threat_label, class_probs, confidence, priority, flow_context)
        llm_result = self._try_llm(prompt)

        if llm_result:
            summary, impact, actions, enhanced = (
                llm_result.get("summary", ""),
                llm_result.get("impact", ""),
                llm_result.get("recommended_actions", []),
                True,
            )
        else:
            kb      = _KB.get(threat_label, _KB["BENIGN"])
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
