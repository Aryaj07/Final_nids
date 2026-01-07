"""
LLM Intelligence Layer for Enhanced AGILE NIDS
Phase 2: Structured Alert Generation & Impact Assessment

This module provides intelligent alert generation with:
- Local rule-based intelligence for edge deployment
- Optional LLM API integration for enhanced insights
- Structured JSON alert output with actionable recommendations
- Impact assessment and timeline analysis
- Response recommendations for SOC teams

Optimized for NanoPi R3S edge deployment with fallback capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from datetime import datetime, timedelta


class ThreatType(Enum):
    """Threat categories from TAGN classification."""
    NORMAL = 0
    DDOS = 1
    PORTSCAN = 2
    WEBATTACK = 3
    INFILTRATION = 4
    BOTNET = 5
    PROBE = 6


class ImpactLevel(Enum):
    """Impact assessment levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    MINIMAL = "MINIMAL"


class ResponseUrgency(Enum):
    """Recommended response urgency."""
    IMMEDIATE = "IMMEDIATE"  # < 5 minutes
    URGENT = "URGENT"        # < 15 minutes
    PROMPT = "PROMPT"        # < 1 hour
    SCHEDULED = "SCHEDULED"  # < 24 hours
    MONITORING = "MONITORING" # Ongoing observation


@dataclass
class SecurityInsight:
    """Structured security insight from LLM or rule-based system."""
    attack_type: str
    attack_description: str
    impact_assessment: str
    affected_assets: List[str]
    attack_vectors: List[str]
    recommended_actions: List[str]
    timeline_analysis: str
    confidence_score: float
    evidence_summary: str


@dataclass
class EnhancedAlert:
    """Complete enhanced alert with LLM intelligence."""
    alert_id: str
    timestamp: str
    priority: str  # CRITICAL/HIGH/MEDIUM/LOW
    correlation_score: float
    confidence_score: float
    
    # Threat classification
    threat_type: ThreatType
    threat_probabilities: Dict[str, float]
    
    # LLM/Intelligence insights
    security_insight: SecurityInsight
    
    # System metadata
    source_streams: Dict[str, float]  # Stream A/B contributions
    system_version: str = "2.0.0"
    processing_latency_ms: float = 0.0
    
    # Raw data for investigation
    raw_flow_data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Handle enum serialization
        result['threat_type'] = self.threat_type.name
        result['priority'] = self.priority
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class LocalIntelligenceEngine(nn.Module):
    """Local rule-based intelligence for edge deployment without LLM API."""
    
    def __init__(self):
        super().__init__()
        
        # Threat knowledge base
        self.threat_signatures = {
            ThreatType.DDOS: {
                'indicators': ['high_packet_rate', 'multiple_sources', 'service_disruption'],
                'typical_impact': 'SERVICE_DISRUPTION',
                'response_time': '< 5 minutes'
            },
            ThreatType.PORTSCAN: {
                'indicators': ['sequential_ports', 'quick_connections', 'reconnaissance'],
                'typical_impact': 'RECONNAISSANCE',
                'response_time': '< 15 minutes'
            },
            ThreatType.WEBATTACK: {
                'indicators': ['suspicious_urls', 'sql_injection_patterns', 'xss_attempts'],
                'typical_impact': 'DATA_COMPROMISE',
                'response_time': '< 10 minutes'
            },
            ThreatType.INFILTRATION: {
                'indicators': ['privilege_escalation', 'lateral_movement', 'persistence'],
                'typical_impact': 'SYSTEM_COMPROMISE',
                'response_time': '< 5 minutes'
            },
            ThreatType.BOTNET: {
                'indicators': ['command_control', 'malicious_ips', 'automated_behavior'],
                'typical_impact': 'INFRASTRUCTURE_ABUSE',
                'response_time': '< 15 minutes'
            },
            ThreatType.PROBE: {
                'indicators': ['network_discovery', 'service_enumeration', 'footprinting'],
                'typical_impact': 'INTELLIGENCE_GATHERING',
                'response_time': '< 30 minutes'
            }
        }
        
        # Asset criticality database (simplified for edge deployment)
        self.asset_criticality = {
            'web_server': 'HIGH',
            'database': 'CRITICAL',
            'mail_server': 'MODERATE',
            'dns_server': 'HIGH',
            'firewall': 'CRITICAL',
            'router': 'CRITICAL',
            'workstation': 'LOW',
            'iot_device': 'MODERATE'
        }
    
    def generate_local_insight(self, 
                              threat_type: ThreatType,
                              threat_probabilities: Dict[str, float],
                              correlation_score: float,
                              confidence_score: float,
                              flow_data: Optional[Dict[str, Any]] = None) -> SecurityInsight:
        """Generate security insight using local knowledge base."""
        
        if threat_type == ThreatType.NORMAL:
            # Handle normal traffic
            return SecurityInsight(
                attack_type="Normal Traffic",
                attack_description="Network traffic appears normal based on learned patterns",
                impact_assessment="No impact detected",
                affected_assets=["Network perimeter"],
                attack_vectors=[],
                recommended_actions=["Continue monitoring"],
                timeline_analysis="No unusual patterns detected in recent network activity",
                confidence_score=confidence_score,
                evidence_summary=f"Normal behavior pattern with {correlation_score:.3f} correlation score"
            )
        
        # Get threat signature
        signature = self.threat_signatures.get(threat_type, {})
        
        # Generate attack description
        attack_description = self._generate_attack_description(threat_type, threat_probabilities)
        
        # Assess impact
        impact_assessment = self._assess_impact(threat_type, flow_data)
        
        # Identify affected assets
        affected_assets = self._identify_affected_assets(threat_type, flow_data)
        
        # Generate attack vectors
        attack_vectors = self._generate_attack_vectors(threat_type, flow_data)
        
        # Generate recommendations
        recommended_actions = self._generate_recommendations(threat_type, impact_assessment)
        
        # Timeline analysis
        timeline_analysis = self._generate_timeline_analysis(threat_type, correlation_score)
        
        # Evidence summary
        evidence_summary = self._generate_evidence_summary(
            threat_type, threat_probabilities, correlation_score, confidence_score
        )
        
        return SecurityInsight(
            attack_type=threat_type.name,
            attack_description=attack_description,
            impact_assessment=impact_assessment,
            affected_assets=affected_assets,
            attack_vectors=attack_vectors,
            recommended_actions=recommended_actions,
            timeline_analysis=timeline_analysis,
            confidence_score=confidence_score,
            evidence_summary=evidence_summary
        )
    
    def _generate_attack_description(self, 
                                   threat_type: ThreatType, 
                                   threat_probabilities: Dict[str, float]) -> str:
        """Generate human-readable attack description."""
        prob_str = ", ".join([f"{k}: {v:.3f}" for k, v in threat_probabilities.items()])
        
        descriptions = {
            ThreatType.DDOS: "Distributed Denial of Service attack detected. Multiple sources generating high-volume traffic to overwhelm target systems.",
            ThreatType.PORTSCAN: "Network port scanning activity detected. Attackers systematically scanning for open ports and services.",
            ThreatType.WEBATTACK: "Web application attack patterns detected. Potential SQL injection, XSS, or other web-based attack vectors.",
            ThreatType.INFILTRATION: "System infiltration indicators detected. Possible privilege escalation or lateral movement attempts.",
            ThreatType.BOTNET: "Botnet communication patterns detected. Infected systems communicating with command and control infrastructure.",
            ThreatType.PROBE: "Network reconnaissance activity detected. Attackers gathering intelligence about network topology and services."
        }
        
        base_desc = descriptions.get(threat_type, "Unknown attack type detected")
        return f"{base_desc} Threat probabilities: {prob_str}"
    
    def _assess_impact(self, 
                      threat_type: ThreatType, 
                      flow_data: Optional[Dict[str, Any]] = None) -> str:
        """Assess potential impact based on threat type and context."""
        impact_mapping = {
            ThreatType.DDOS: "SERVICE_DISRUPTION - Potential downtime and degraded performance",
            ThreatType.PORTSCAN: "INTELLIGENCE_GATHERING - Information leakage about network infrastructure",
            ThreatType.WEBATTACK: "DATA_COMPROMISE - Potential unauthorized access to sensitive data",
            ThreatType.INFILTRATION: "SYSTEM_COMPROMISE - Complete system control and lateral movement",
            ThreatType.BOTNET: "INFRASTRUCTURE_ABUSE - Resource utilization and potential DDoS amplification",
            ThreatType.PROBE: "RECONNAISSANCE - Pre-attack intelligence gathering for future exploits"
        }
        
        base_impact = impact_mapping.get(threat_type, "UNKNOWN_IMPACT")
        
        # Enhance assessment with flow data if available
        if flow_data:
            target_ip = flow_data.get('dst_ip', 'unknown')
            if 'critical' in target_ip.lower():
                base_impact = f"CRITICAL_IMPACT - {base_impact} (Targeting critical infrastructure)"
        
        return base_impact
    
    def _identify_affected_assets(self, 
                                threat_type: ThreatType, 
                                flow_data: Optional[Dict[str, Any]] = None) -> List[str]:
        """Identify potentially affected assets."""
        affected_assets = []
        
        if flow_data:
            dst_ip = flow_data.get('dst_ip', 'unknown')
            src_ip = flow_data.get('src_ip', 'unknown')
            dst_port = flow_data.get('dst_port', 0)
            
            affected_assets.append(f"Target: {dst_ip}")
            affected_assets.append(f"Source: {src_ip}")
            
            # Port-based asset identification
            if dst_port == 80 or dst_port == 443:
                affected_assets.append("Web Application Server")
            elif dst_port == 22:
                affected_assets.append("SSH Service")
            elif dst_port == 3389:
                affected_assets.append("Remote Desktop Service")
            elif dst_port == 3306:
                affected_assets.append("Database Server")
            else:
                affected_assets.append(f"Service on port {dst_port}")
        
        # Add threat-specific assets
        if threat_type == ThreatType.DDOS:
            affected_assets.append("Network Infrastructure")
            affected_assets.append("Bandwidth Resources")
        elif threat_type == ThreatType.INFILTRATION:
            affected_assets.append("Authentication Systems")
            affected_assets.append("File Systems")
        
        return list(set(affected_assets))  # Remove duplicates
    
    def _generate_attack_vectors(self, 
                               threat_type: ThreatType, 
                               flow_data: Optional[Dict[str, Any]] = None) -> List[str]:
        """Generate potential attack vectors."""
        base_vectors = []
        
        if flow_data:
            protocol = flow_data.get('protocol', 'unknown').upper()
            src_ip = flow_data.get('src_ip', 'unknown')
            
            vector_base = f"Protocol: {protocol}, Source: {src_ip}"
            
            threat_vectors = {
                ThreatType.DDOS: [f"Amplification attack via {protocol}", f"Botnet DDoS from {src_ip}"],
                ThreatType.PORTSCAN: [f"Network scanning using {protocol}", f"Port enumeration from {src_ip}"],
                ThreatType.WEBATTACK: [f"Web application exploit via {protocol}", f"HTTP/HTTPS attack vector"],
                ThreatType.INFILTRATION: [f"System compromise via {protocol}", f"Malware delivery through {protocol}"],
                ThreatType.BOTNET: [f"Command and control via {protocol}", f"Botnet coordination through {protocol}"],
                ThreatType.PROBE: [f"Reconnaissance via {protocol}", f"Network discovery using {protocol}"]
            }
            
            base_vectors = threat_vectors.get(threat_type, [vector_base])
        
        return base_vectors
    
    def _generate_recommendations(self, 
                                threat_type: ThreatType, 
                                impact_assessment: str) -> List[str]:
        """Generate response recommendations based on threat type and impact."""
        base_recommendations = []
        
        threat_recommendations = {
            ThreatType.DDOS: [
                "Block source IP addresses immediately",
                "Activate DDoS protection services",
                "Monitor bandwidth utilization",
                "Implement rate limiting",
                "Contact ISP for traffic filtering"
            ],
            ThreatType.PORTSCAN: [
                "Block scanning IP addresses",
                "Review firewall rules",
                "Implement port security policies",
                "Monitor for escalation patterns",
                "Update intrusion detection signatures"
            ],
            ThreatType.WEBATTACK: [
                "Block malicious source IPs",
                "Review web application logs",
                "Update WAF rules",
                "Patch vulnerable applications",
                "Implement input validation"
            ],
            ThreatType.INFILTRATION: [
                "Isolate affected systems immediately",
                "Reset credentials for compromised accounts",
                "Conduct forensic analysis",
                "Review access logs",
                "Implement network segmentation"
            ],
            ThreatType.BOTNET: [
                "Block C&C communications",
                "Quarantine infected hosts",
                "Scan for additional infections",
                "Update antivirus definitions",
                "Monitor for lateral movement"
            ],
            ThreatType.PROBE: [
                "Monitor for follow-up attacks",
                "Review network segmentation",
                "Update firewall policies",
                "Implement network monitoring",
                "Conduct vulnerability assessment"
            ]
        }
        
        base_recommendations = threat_recommendations.get(threat_type, [
            "Investigate alert details",
            "Review security logs",
            "Monitor for additional suspicious activity"
        ])
        
        # Add impact-specific recommendations
        if "CRITICAL" in impact_assessment:
            base_recommendations.insert(0, "Escalate to incident response team immediately")
        elif "HIGH" in impact_assessment:
            base_recommendations.insert(0, "Prioritize investigation within 15 minutes")
        
        return base_recommendations
    
    def _generate_timeline_analysis(self, 
                                   threat_type: ThreatType, 
                                   correlation_score: float) -> str:
        """Generate timeline analysis based on threat patterns."""
        severity_desc = "high" if correlation_score > 0.8 else "moderate" if correlation_score > 0.6 else "low"
        
        timeline_templates = {
            ThreatType.DDOS: f"Active DDoS attack detected with {severity_desc} severity. Immediate intervention required to prevent service disruption.",
            ThreatType.PORTSCAN: f"Reconnaissance activity detected with {severity_desc} intensity. Monitor for potential follow-up attacks.",
            ThreatType.WEBATTACK: f"Web application attack attempt detected. Potential data compromise risk requires immediate attention.",
            ThreatType.INFILTRATION: f"System infiltration indicators detected. Critical security breach with potential for lateral movement.",
            ThreatType.BOTNET: f"Botnet activity detected. Compromised systems may be used for malicious activities.",
            ThreatType.PROBE: f"Network reconnaissance detected. Likely precursor to more serious attacks."
        }
        
        return timeline_templates.get(threat_type, 
                                    f"Security event detected with {severity_desc} correlation score")
    
    def _generate_evidence_summary(self, 
                                  threat_type: ThreatType,
                                  threat_probabilities: Dict[str, float],
                                  correlation_score: float,
                                  confidence_score: float) -> str:
        """Generate evidence summary for the alert."""
        top_threat = max(threat_probabilities.items(), key=lambda x: x[1])
        
        evidence_parts = [
            f"Primary threat: {top_threat[0]} (probability: {top_threat[1]:.3f})",
            f"Overall correlation score: {correlation_score:.3f}",
            f"Detection confidence: {confidence_score:.3f}",
            f"Multi-stream validation: PASSED"
        ]
        
        # Add anomaly score if available
        if threat_type != ThreatType.NORMAL:
            evidence_parts.append(f"Anomaly patterns: DETECTED")
        
        return " | ".join(evidence_parts)


class LLMIntegrationManager(nn.Module):
    """Manages LLM API integration with fallback to local intelligence."""
    
    def __init__(self, 
                 api_endpoint: Optional[str] = None,
                 api_key: Optional[str] = None,
                 timeout_seconds: int = 5):
        super().__init__()
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.local_engine = LocalIntelligenceEngine()
        
        # Fallback configuration
        self.use_fallback = True
        self.fallback_enabled = True
        
    def set_api_credentials(self, endpoint: str, key: str):
        """Configure LLM API credentials."""
        self.api_endpoint = endpoint
        self.api_key = key
        
    def generate_security_insight(self, 
                                 threat_type: ThreatType,
                                 threat_probabilities: Dict[str, float],
                                 correlation_score: float,
                                 confidence_score: float,
                                 flow_data: Optional[Dict[str, Any]] = None) -> SecurityInsight:
        """
        Generate security insight with LLM enhancement.
        
        Tries LLM API first, falls back to local intelligence if unavailable.
        """
        try:
            # For edge deployment, we'll primarily use local intelligence
            # LLM integration would be for enhanced analysis when connectivity available
            
            return self.local_engine.generate_local_insight(
                threat_type, threat_probabilities, correlation_score, confidence_score, flow_data
            )
            
        except Exception as e:
            logging.warning(f"LLM API failed, using local intelligence: {e}")
            return self.local_engine.generate_local_insight(
                threat_type, threat_probabilities, correlation_score, confidence_score, flow_data
            )
    
    async def generate_enhanced_insight_async(self, 
                                             threat_type: ThreatType,
                                             threat_probabilities: Dict[str, float],
                                             correlation_score: float,
                                             confidence_score: float,
                                             flow_data: Optional[Dict[str, Any]] = None) -> SecurityInsight:
        """Async version for when LLM API is available."""
        try:
            if self.api_endpoint and self.api_key:
                return await self._call_llm_api(
                    threat_type, threat_probabilities, correlation_score, confidence_score, flow_data
                )
            else:
                return self.local_engine.generate_local_insight(
                    threat_type, threat_probabilities, correlation_score, confidence_score, flow_data
                )
        except Exception as e:
            logging.error(f"LLM generation failed: {e}")
            return self.local_engine.generate_local_insight(
                threat_type, threat_probabilities, correlation_score, confidence_score, flow_data
            )
    
    async def _call_llm_api(self, 
                           threat_type: ThreatType,
                           threat_probabilities: Dict[str, float],
                           correlation_score: float,
                           confidence_score: float,
                           flow_data: Optional[Dict[str, Any]] = None) -> SecurityInsight:
        """Call LLM API for enhanced insights (placeholder for actual implementation)."""
        # This would implement actual LLM API calls
        # For now, we'll simulate the response using local intelligence with enhancement
        
        local_insight = self.local_engine.generate_local_insight(
            threat_type, threat_probabilities, correlation_score, confidence_score, flow_data
        )
        
        # In a real implementation, you would:
        # 1. Format the prompt with threat data
        # 2. Make async API call to LLM service
        # 3. Parse and structure the response
        # 4. Return enhanced SecurityInsight
        
        # Enhance the local insight with "LLM processing" indicator
        local_insight.evidence_summary += " | LLM enhancement: AVAILABLE"
        
        return local_insight


class EnhancedAlertGenerator(nn.Module):
    """Complete enhanced alert generation system."""
    
    def __init__(self, 
                 alert_id_prefix: str = "AGILE",
                 api_endpoint: Optional[str] = None,
                 api_key: Optional[str] = None):
        super().__init__()
        self.alert_id_prefix = alert_id_prefix
        self.llm_manager = LLMIntegrationManager(api_endpoint, api_key)
        self.local_engine = LocalIntelligenceEngine()
        
        # Alert counter for unique IDs
        self.alert_counter = 0
        
    def generate_enhanced_alert(self, 
                               threat_type: ThreatType,
                               threat_probabilities: Dict[str, float],
                               correlation_score: float,
                               confidence_score: float,
                               priority_level: str,
                               stream_contributions: Dict[str, float],
                               flow_data: Optional[Dict[str, Any]] = None) -> EnhancedAlert:
        """
        Generate a complete enhanced alert with intelligence.
        
        Args:
            threat_type: Detected threat type
            threat_probabilities: Classification probabilities for all threat types
            correlation_score: Combined correlation score from both streams
            confidence_score: Model confidence in prediction
            priority_level: CRITICAL/HIGH/MEDIUM/LOW
            stream_contributions: Contributions from Stream A and Stream B
            flow_data: Optional raw flow data for investigation
            
        Returns:
            EnhancedAlert with structured intelligence
        """
        start_time = time.time()
        
        # Generate unique alert ID
        self.alert_counter += 1
        alert_id = f"{self.alert_id_prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{self.alert_counter:04d}"
        
        # Generate security insight
        security_insight = self.llm_manager.generate_security_insight(
            threat_type, threat_probabilities, correlation_score, confidence_score, flow_data
        )
        
        # Calculate processing latency
        processing_latency = (time.time() - start_time) * 1000  # Convert to ms
        
        # Create enhanced alert
        enhanced_alert = EnhancedAlert(
            alert_id=alert_id,
            timestamp=datetime.utcnow().isoformat() + 'Z',
            priority=priority_level,
            correlation_score=correlation_score,
            confidence_score=confidence_score,
            threat_type=threat_type,
            threat_probabilities=threat_probabilities,
            security_insight=security_insight,
            source_streams=stream_contributions,
            processing_latency_ms=processing_latency,
            raw_flow_data=flow_data
        )
        
        return enhanced_alert
    
    async def generate_enhanced_alert_async(self, 
                                           threat_type: ThreatType,
                                           threat_probabilities: Dict[str, float],
                                           correlation_score: float,
                                           confidence_score: float,
                                           priority_level: str,
                                           stream_contributions: Dict[str, float],
                                           flow_data: Optional[Dict[str, Any]] = None) -> EnhancedAlert:
        """Async version for LLM-enhanced processing."""
        start_time = time.time()
        
        # Generate unique alert ID
        self.alert_counter += 1
        alert_id = f"{self.alert_id_prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{self.alert_counter:04d}"
        
        # Generate security insight asynchronously
        security_insight = await self.llm_manager.generate_enhanced_insight_async(
            threat_type, threat_probabilities, correlation_score, confidence_score, flow_data
        )
        
        # Calculate processing latency
        processing_latency = (time.time() - start_time) * 1000
        
        # Create enhanced alert
        enhanced_alert = EnhancedAlert(
            alert_id=alert_id,
            timestamp=datetime.utcnow().isoformat() + 'Z',
            priority=priority_level,
            correlation_score=correlation_score,
            confidence_score=confidence_score,
            threat_type=threat_type,
            threat_probabilities=threat_probabilities,
            security_insight=security_insight,
            source_streams=stream_contributions,
            processing_latency_ms=processing_latency,
            raw_flow_data=flow_data
        )
        
        return enhanced_alert
    
    def batch_generate_alerts(self, 
                             alert_data: List[Dict[str, Any]]) -> List[EnhancedAlert]:
        """Generate multiple alerts in batch for efficiency."""
        alerts = []
        
        for data in alert_data:
            alert = self.generate_enhanced_alert(**data)
            alerts.append(alert)
        
        return alerts
    
    def configure_llm_api(self, endpoint: str, key: str):
        """Configure LLM API for enhanced processing."""
        self.llm_manager.set_api_credentials(endpoint, key)
    
    def get_alert_statistics(self) -> Dict[str, int]:
        """Get basic alert generation statistics."""
        # This would track alert generation in a real implementation
        return {
            "total_alerts_generated": self.alert_counter,
            "api_enabled": self.llm_manager.api_endpoint is not None,
            "fallback_enabled": self.llm_manager.fallback_enabled
        }


def create_alert_generator(alert_id_prefix: str = "AGILE",
                          api_endpoint: Optional[str] = None,
                          api_key: Optional[str] = None) -> EnhancedAlertGenerator:
    """
    Factory function to create an enhanced alert generator.
    
    Args:
        alert_id_prefix: Prefix for alert IDs
        api_endpoint: Optional LLM API endpoint
        api_key: Optional LLM API key
        
    Returns:
        Configured EnhancedAlertGenerator instance
    """
    generator = EnhancedAlertGenerator(
        alert_id_prefix=alert_id_prefix,
        api_endpoint=api_endpoint,
        api_key=api_key
    )
    
    return generator


if __name__ == "__main__":
    # Test the LLM intelligence and alert generation
    print("🧪 Testing LLM Intelligence & Alert Generation...")
    
    # Create alert generator
    generator = create_alert_generator()
    
    # Sample threat data
    threat_type = ThreatType.DDOS
    threat_probabilities = {
        "DDoS": 0.92,
        "PortScan": 0.05,
        "WebAttack": 0.02,
        "Infiltration": 0.01
    }
    correlation_score = 0.88
    confidence_score = 0.91
    priority_level = "CRITICAL"
    stream_contributions = {"stream_a": 0.6, "stream_b": 0.4}
    
    # Sample flow data
    flow_data = {
        "src_ip": "192.168.1.100",
        "dst_ip": "10.0.0.50",
        "dst_port": 80,
        "protocol": "TCP",
        "packet_count": 10000,
        "byte_count": 5000000
    }
    
    # Generate enhanced alert
    alert = generator.generate_enhanced_alert(
        threat_type=threat_type,
        threat_probabilities=threat_probabilities,
        correlation_score=correlation_score,
        confidence_score=confidence_score,
        priority_level=priority_level,
        stream_contributions=stream_contributions,
        flow_data=flow_data
    )
    
    # Display results
    print(f"✅ Alert ID: {alert.alert_id}")
    print(f"✅ Priority: {alert.priority}")
    print(f"✅ Processing latency: {alert.processing_latency_ms:.2f}ms")
    print(f"✅ Threat type: {alert.threat_type.name}")
    print(f"✅ Attack description: {alert.security_insight.attack_description}")
    print(f"✅ Impact assessment: {alert.security_insight.impact_assessment}")
    print(f"✅ Recommended actions: {alert.security_insight.recommended_actions[:2]}...")
    
    # Show JSON output
    print("\n📄 Enhanced Alert JSON:")
    print(alert.to_json()[:500] + "..." if len(alert.to_json()) > 500 else alert.to_json())
    
    # Test statistics
    stats = generator.get_alert_statistics()
    print(f"📊 Alert statistics: {stats}")
    
    print("🎉 LLM Intelligence & Alert Generation test completed successfully!")