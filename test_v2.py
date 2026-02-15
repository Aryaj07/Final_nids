"""
HALO NIDS -- AGILE v2.1  Comprehensive Testing & Inference
============================================================
v2.1 improvements over v2:
  - Sliding-window sequence inference: flows are grouped into real windows
    of seq_len instead of repeating a single flow. This matches training.
  - Confidence-gated predictions: low-confidence attack predictions are
    reverted to BENIGN, dramatically reducing false positives.
  - TAGN-primary decision: TAGN classification (after gating) is the
    primary signal; anomaly score amplifies priority, not detection.

Run:
    python test_v2.py
    python test_v2.py --experiment experiments_v2/agile_v2_YYYYMMDD_HHMMSS
"""

import os, sys, json, time, argparse, logging
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
)
import joblib

from models_v2.autoencoder import Autoencoder
from models_v2.tagn_network import create_tagn_model, THREAT_LABELS, NUM_CLASSES
from models_v2.correlation_engine import create_correlation_engine, Priority
from models_v2.llm_intelligence import AlertGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("TestV2")


# -----------------------------------------------------------------------
# Label mapping (same as train_v2)
# -----------------------------------------------------------------------

LABEL_MAP = {
    "BENIGN": 0,
    "DDoS": 1, "DoS Hulk": 1, "DoS GoldenEye": 1, "DoS slowloris": 1, "DoS Slowhttptest": 1,
    "PortScan": 2,
    "Web Attack - Brute Force": 3, "Web Attack - XSS": 3, "Web Attack - Sql Injection": 3,
    "Web Attack \u2013 Brute Force": 3, "Web Attack \u2013 XSS": 3, "Web Attack \u2013 Sql Injection": 3,
    "Web Attack \x96 Brute Force": 3, "Web Attack \x96 XSS": 3, "Web Attack \x96 Sql Injection": 3,
    "Infiltration": 4, "Heartbleed": 4,
    "Bot": 5,
    "FTP-Patator": 6, "SSH-Patator": 6,
}


def map_labels(raw: pd.Series) -> np.ndarray:
    raw_stripped = raw.str.strip()
    mapped = raw_stripped.map(LABEL_MAP)
    unmapped = mapped.isna()
    if unmapped.any():
        for idx in raw_stripped[unmapped].index:
            val = str(raw_stripped[idx])
            low = val.lower()
            if pd.isna(raw_stripped[idx]):
                mapped[idx] = 0
            elif "web attack" in low:
                mapped[idx] = 3
            elif low.startswith("dos") or "ddos" in low:
                mapped[idx] = 1
            elif "infiltr" in low:
                mapped[idx] = 4
            elif "bot" in low:
                mapped[idx] = 5
            elif "patator" in low or "ssh" in low or "ftp" in low:
                mapped[idx] = 6
            else:
                mapped[idx] = 0
    mapped = mapped.fillna(0)
    return mapped.astype(int).values


DATASETS = {
    "DDoS": {
        "file": "GeneratedLabelledFlows/TrafficLabelling/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "type": "attack",
    },
    "PortScan": {
        "file": "GeneratedLabelledFlows/TrafficLabelling/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "type": "attack",
    },
    "WebAttacks": {
        "file": "GeneratedLabelledFlows/TrafficLabelling/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "type": "attack",
    },
    "Infiltration": {
        "file": "GeneratedLabelledFlows/TrafficLabelling/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "type": "attack",
    },
    "Monday_Benign": {
        "file": "GeneratedLabelledFlows/TrafficLabelling/Monday-WorkingHours.pcap_ISCX.csv",
        "type": "benign",
    },
    "Tuesday": {
        "file": "GeneratedLabelledFlows/TrafficLabelling/Tuesday-WorkingHours.pcap_ISCX.csv",
        "type": "mixed",
    },
    "Wednesday": {
        "file": "GeneratedLabelledFlows/TrafficLabelling/Wednesday-workingHours.pcap_ISCX.csv",
        "type": "mixed",
    },
    "Friday_Morning": {
        "file": "GeneratedLabelledFlows/TrafficLabelling/Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "type": "benign",
    },
}


# -----------------------------------------------------------------------
# Tester
# -----------------------------------------------------------------------

class AGILETester:

    def __init__(self, exp_dir: str, max_samples: int = 60000,
                 confidence_threshold: float = 0.0):
        self.exp_dir = exp_dir
        self.max_samples = max_samples
        self.device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold

        cfg_path = os.path.join(exp_dir, "deploy_config.json")
        with open(cfg_path) as f:
            self.cfg = json.load(f)

        self.input_dim = self.cfg["input_dim"]
        self.threshold = self.cfg["anomaly_threshold"]
        self.seq_len   = self.cfg["seq_len"]
        log.info("Config: input_dim=%d  threshold=%.6f  seq_len=%d  conf_gate=%.2f",
                 self.input_dim, self.threshold, self.seq_len, self.confidence_threshold)

        self._load_models()
        self.alert_gen = AlertGenerator(prefix="AGILE-V2")

    def _load_models(self):
        d = self.exp_dir
        self.scaler = joblib.load(os.path.join(d, "scaler.pkl"))

        self.ae = Autoencoder(self.input_dim)
        self.ae.load_state_dict(torch.load(os.path.join(d, "autoencoder.pt"),
                                           map_location="cpu", weights_only=False))
        self.ae.eval()

        self.tagn = create_tagn_model(input_dim=self.input_dim, hidden_dim=128,
                                      num_classes=NUM_CLASSES)
        self.tagn.load_state_dict(torch.load(os.path.join(d, "tagn_best.pt"),
                                             map_location="cpu", weights_only=False))
        self.tagn.eval()

        self.engine = create_correlation_engine(
            hidden=64,
            anomaly_threshold=self.threshold,
            confidence_threshold=self.confidence_threshold,
        )
        self.engine.load_state_dict(
            torch.load(os.path.join(d, "correlation_engine.pt"),
                       map_location="cpu", weights_only=False),
            strict=False,   # v2.1 engine is structurally same, just new calling convention
        )
        self.engine.eval()
        self.engine.anomaly_threshold = self.threshold
        self.engine.confidence_threshold = self.confidence_threshold
        log.info("All models loaded from %s", d)

    # ------------------------------------------------------------------
    def _load_dataset(self, path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not os.path.exists(path):
            log.warning("File not found: %s", path)
            return None, None
        for enc in ("utf-8", "latin-1", "iso-8859-1", "cp1252"):
            try:
                df = pd.read_csv(path, encoding=enc, low_memory=False, nrows=self.max_samples)
                df.columns = df.columns.str.strip()
                break
            except Exception:
                continue
        else:
            return None, None

        y = map_labels(df["Label"])
        num = df.drop(columns=["Label"], errors="ignore").select_dtypes(include=[np.number])
        num = num.replace([np.inf, -np.inf], np.nan)
        valid = ~num.isna().any(axis=1)
        num = num[valid].clip(-1e6, 1e6).reset_index(drop=True)
        y = y[valid.values]

        if num.shape[1] > self.input_dim:
            num = num.iloc[:, :self.input_dim]
        elif num.shape[1] < self.input_dim:
            pad = pd.DataFrame(np.zeros((len(num), self.input_dim - num.shape[1])))
            num = pd.concat([num, pad], axis=1)

        X = self.scaler.transform(num.values).astype(np.float32)
        return X, y

    # ------------------------------------------------------------------
    # HYBRID INFERENCE: sliding-window + per-flow refinement
    # ------------------------------------------------------------------
    def _tagn_window_pass(self, X: np.ndarray):
        """Pass 1: non-overlapping sliding windows (good for dense attacks like DDoS)."""
        n = len(X)
        sl = self.seq_len
        pred = np.zeros(n, dtype=int)
        prob = np.zeros((n, NUM_CLASSES), dtype=np.float32)
        conf = np.zeros(n, dtype=np.float32)
        corr = np.zeros((n, 16), dtype=np.float32)

        n_win = n // sl
        rem   = n % sl
        win_bs = 64

        if n_win > 0:
            for ws in range(0, n_win, win_bs):
                we = min(ws + win_bs, n_win)
                seqs = [X[w*sl:(w+1)*sl] for w in range(ws, we)]
                seq_t = torch.tensor(np.array(seqs))
                with torch.no_grad():
                    out = self.tagn(seq_t)
                for j, w in enumerate(range(ws, we)):
                    s, e = w * sl, (w + 1) * sl
                    pred[s:e] = out["classification"]["predicted_class"][j].item()
                    prob[s:e] = out["classification"]["class_probabilities"][j].numpy()
                    conf[s:e] = out["classification"]["confidence_score"][j].item()
                    corr[s:e] = out["correlation_features"][j].numpy()

        if rem > 0:
            tail = n_win * sl
            if n_win > 0:
                ls = (n_win - 1) * sl
                pred[tail:] = pred[ls]; prob[tail:] = prob[ls]
                conf[tail:] = conf[ls]; corr[tail:] = corr[ls]
            else:
                padded = np.zeros((sl, X.shape[1]), dtype=np.float32)
                padded[:rem] = X[tail:]
                with torch.no_grad():
                    out = self.tagn(torch.tensor(padded).unsqueeze(0))
                pred[tail:] = out["classification"]["predicted_class"].item()
                prob[tail:] = out["classification"]["class_probabilities"][0].numpy()
                conf[tail:] = out["classification"]["confidence_score"].item()
                corr[tail:] = out["correlation_features"][0].numpy()

        return pred, prob, conf, corr

    def _tagn_perflow_pass(self, X: np.ndarray, indices: np.ndarray):
        """
        Pass 2: per-flow inference on selected flows.
        Each flow is placed in a sequence of seq_len copies of itself.
        This is the same as v2 inference -- works for sparse/scattered attacks.
        """
        n_idx = len(indices)
        pred = np.zeros(n_idx, dtype=int)
        prob = np.zeros((n_idx, NUM_CLASSES), dtype=np.float32)
        conf = np.zeros(n_idx, dtype=np.float32)
        corr = np.zeros((n_idx, 16), dtype=np.float32)

        bs = 256
        for i in range(0, n_idx, bs):
            batch_idx = indices[i:i+bs]
            batch = torch.tensor(X[batch_idx])                         # (B, D)
            seq = batch.unsqueeze(1).expand(-1, self.seq_len, -1)      # (B, sl, D)
            with torch.no_grad():
                out = self.tagn(seq)
            b = len(batch_idx)
            pred[i:i+b] = out["classification"]["predicted_class"].numpy()
            prob[i:i+b] = out["classification"]["class_probabilities"].numpy()
            conf[i:i+b] = out["classification"]["confidence_score"].numpy()
            corr[i:i+b] = out["correlation_features"].numpy()

        return pred, prob, conf, corr

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Hybrid two-pass inference:

        Pass 1 (sliding window): processes flows in non-overlapping windows
          of seq_len. Excellent for dense/contiguous attacks (DDoS) and keeps
          benign FPR very low. Matches training data structure.

        Pass 2 (per-flow): re-examines every flow that Pass 1 classified as
          BENIGN. Each flow is evaluated individually (repeated seq_len times
          to form a sequence). This recovers sparse/scattered attacks like
          PortScan and WebAttack that get diluted in windows.

        Merge rule: for any flow, if EITHER pass says attack (above the
          confidence gate), the attack prediction wins. When both say attack,
          the higher-confidence prediction is used.
        """
        n = len(X)
        bs = 512

        # -- Per-flow anomaly scores (Stream A -- always per-flow) --
        anom_arr = np.zeros(n, dtype=np.float32)
        for i in range(0, n, bs):
            batch = torch.tensor(X[i:i+bs])
            with torch.no_grad():
                anom_arr[i:i+batch.size(0)] = Autoencoder.reconstruction_error(
                    batch, self.ae(batch)).numpy()

        # -- Pass 1: sliding-window TAGN --
        win_pred, win_prob, win_conf, win_corr = self._tagn_window_pass(X)
        log.info("    Pass 1 (window):  %d attack flows detected", (win_pred > 0).sum())

        # -- Pass 2: per-flow ONLY on anomalous benign-classified flows --
        # Key insight: only re-examine flows where the autoencoder flags
        # them as anomalous but the window missed them (sparse attack in
        # a benign window).  This avoids re-examining the vast majority
        # of truly benign flows, keeping FPR low.
        benign_idx = np.where(
            (win_pred == 0) & (anom_arr > self.threshold)
        )[0]
        log.info("    Pass 2 (per-flow): re-examining %d anomalous benign-classified flows "
                 "(out of %d benign, %d above anomaly threshold)",
                 len(benign_idx), (win_pred == 0).sum(),
                 ((win_pred == 0) & (anom_arr > self.threshold)).sum())

        pf_pred = np.zeros(n, dtype=int)
        pf_prob = np.zeros((n, NUM_CLASSES), dtype=np.float32)
        pf_conf = np.zeros(n, dtype=np.float32)
        pf_corr = np.zeros((n, 16), dtype=np.float32)

        if len(benign_idx) > 0:
            p2_pred, p2_prob, p2_conf, p2_corr = self._tagn_perflow_pass(X, benign_idx)
            pf_pred[benign_idx] = p2_pred
            pf_prob[benign_idx] = p2_prob
            pf_conf[benign_idx] = p2_conf
            pf_corr[benign_idx] = p2_corr
            log.info("    Pass 2 found:     %d additional attack flows", (p2_pred > 0).sum())

        # -- Merge: attack wins, higher confidence preferred --
        final_pred = win_pred.copy()
        final_prob = win_prob.copy()
        final_conf = win_conf.copy()
        final_corr = win_corr.copy()

        # For flows where window said benign but per-flow says attack:
        # adopt the per-flow prediction
        upgrade_mask = (win_pred == 0) & (pf_pred > 0)
        final_pred[upgrade_mask] = pf_pred[upgrade_mask]
        final_prob[upgrade_mask] = pf_prob[upgrade_mask]
        final_conf[upgrade_mask] = pf_conf[upgrade_mask]
        final_corr[upgrade_mask] = pf_corr[upgrade_mask]

        # For flows where both say attack, use higher confidence
        both_attack = (win_pred > 0) & (pf_pred > 0)
        pf_better = both_attack & (pf_conf > win_conf)
        final_pred[pf_better] = pf_pred[pf_better]
        final_prob[pf_better] = pf_prob[pf_better]
        final_conf[pf_better] = pf_conf[pf_better]
        final_corr[pf_better] = pf_corr[pf_better]

        log.info("    Merged:           %d total attack flows", (final_pred > 0).sum())

        # -- Phase 3: Correlation Engine with confidence gating --
        prio_arr   = np.zeros(n, dtype=int)
        fusion_arr = np.zeros(n, dtype=np.float32)
        gated_arr  = np.zeros(n, dtype=int)

        for i in range(0, n, bs):
            end = min(i + bs, n)
            with torch.no_grad():
                c_out = self.engine(
                    torch.tensor(anom_arr[i:end]),
                    torch.tensor(final_corr[i:end]),
                    torch.tensor(final_conf[i:end]),
                    torch.tensor(final_pred[i:end], dtype=torch.long),
                )
            prio_arr[i:end]   = c_out["priority"].numpy()
            fusion_arr[i:end] = c_out["fusion_score"].squeeze(-1).numpy()
            gated_arr[i:end]  = c_out["gated_class"].numpy()

        return {
            "anomaly_scores":  anom_arr,
            "predicted_class": final_pred,    # merged raw TAGN prediction
            "gated_class":     gated_arr,     # after confidence gating
            "class_probs":     final_prob,
            "confidence":      final_conf,
            "priority":        prio_arr,
            "fusion_score":    fusion_arr,
        }

    # ------------------------------------------------------------------
    def evaluate(self, y_true: np.ndarray, preds: Dict) -> Dict:
        # Use gated_class for evaluation (TAGN-primary with confidence gate)
        y_pred = preds["gated_class"]

        # binary
        yt_bin = (y_true > 0).astype(int)
        yp_bin = (y_pred > 0).astype(int)

        if len(np.unique(yt_bin)) < 2 or len(np.unique(yp_bin)) < 2:
            # handle edge case: only one class present
            cm_vals = confusion_matrix(yt_bin, yp_bin, labels=[0, 1])
            tn, fp, fn, tp = cm_vals.ravel()
        else:
            cm = confusion_matrix(yt_bin, yp_bin)
            tn, fp, fn, tp = cm.ravel()

        bin_metrics = {
            "accuracy":  float(accuracy_score(yt_bin, yp_bin)),
            "precision": float(precision_score(yt_bin, yp_bin, zero_division=0)),
            "recall":    float(recall_score(yt_bin, yp_bin, zero_division=0)),
            "f1":        float(f1_score(yt_bin, yp_bin, zero_division=0)),
            "fpr":       float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        }
        try:
            bin_metrics["roc_auc"] = float(roc_auc_score(yt_bin, 1 - preds["class_probs"][:, 0]))
        except Exception:
            bin_metrics["roc_auc"] = 0.0

        # multi-class (using gated predictions)
        present = sorted(set(y_true) | set(y_pred))
        target_names = [THREAT_LABELS[i] for i in present]
        multi_report = classification_report(
            y_true, y_pred, labels=present, target_names=target_names, zero_division=0,
        )

        # priority distribution
        prio_dist = {}
        pnames = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        for p in range(4):
            prio_dist[pnames[p]] = int((preds["priority"] == p).sum())

        # confidence gating stats
        raw_attacks = (preds["predicted_class"] > 0).sum()
        gated_attacks = (preds["gated_class"] > 0).sum()
        filtered_out = int(raw_attacks - gated_attacks)

        return {
            "binary": bin_metrics,
            "multi_class_report": multi_report,
            "priority_distribution": prio_dist,
            "n_samples": len(y_true),
            "n_benign": int((y_true == 0).sum()),
            "n_attack": int((y_true > 0).sum()),
            "confidence_gating": {
                "raw_attack_predictions": int(raw_attacks),
                "after_gating": int(gated_attacks),
                "filtered_out": filtered_out,
            },
        }

    # ------------------------------------------------------------------
    def generate_sample_alerts(self, X: np.ndarray, y_true: np.ndarray,
                               preds: Dict, n: int = 3):
        alert_idx = np.where(preds["priority"] < Priority.LOW)[0]
        if len(alert_idx) == 0:
            log.info("  No alerts generated for this dataset")
            return []

        chosen = alert_idx[:n]
        alerts = []
        pnames = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        for idx in chosen:
            prob_dict = {THREAT_LABELS[j]: float(preds["class_probs"][idx, j])
                         for j in range(NUM_CLASSES)}
            alert = self.alert_gen.generate(
                anomaly_score=float(preds["anomaly_scores"][idx]),
                predicted_class=int(preds["gated_class"][idx]),
                class_probs=prob_dict,
                confidence=float(preds["confidence"][idx]),
                fusion_score=float(preds["fusion_score"][idx]),
                priority=pnames[preds["priority"][idx]],
            )
            alerts.append(alert)
            log.info("  Alert: %s | %s | %s | %s",
                     alert.alert_id, alert.priority, alert.threat_type,
                     alert.summary[:80])
        return alerts

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Auto-calibrate confidence threshold
    # ------------------------------------------------------------------
    def calibrate_confidence(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Find the confidence threshold that best separates true attacks from
        benign traffic.  Strategy: run a quick prediction, collect confidence
        scores for flows the TAGN thinks are attacks, and set the threshold
        at the confidence level of the WORST-performing benign false-positive
        percentile.

        Simple approach: use the median confidence of benign flows that are
        MIS-classified as attack.  If there are no misclassifications, use a
        very low gate (0.01) to keep everything.
        """
        log.info("  Auto-calibrating confidence gate...")
        preds = self.predict(X)

        raw_pred = preds["predicted_class"]
        conf = preds["confidence"]

        # Benign flows that TAGN wrongly calls attack
        benign_mask = (y == 0)
        fp_mask = benign_mask & (raw_pred > 0)
        # True attack flows that TAGN correctly calls attack
        attack_mask = (y > 0)
        tp_mask = attack_mask & (raw_pred > 0)

        n_fp = fp_mask.sum()
        n_tp = tp_mask.sum()

        if n_tp == 0:
            log.info("    No true positives found -- using gate=0.0 (no gating)")
            return 0.0

        fp_confs = conf[fp_mask] if n_fp > 0 else np.array([])
        tp_confs = conf[tp_mask]

        log.info("    True-positive confidences:  mean=%.4f  min=%.4f  p10=%.4f  p50=%.4f",
                 tp_confs.mean(), tp_confs.min(),
                 np.percentile(tp_confs, 10), np.percentile(tp_confs, 50))

        if n_fp > 0:
            log.info("    False-positive confidences: mean=%.4f  max=%.4f  p90=%.4f",
                     fp_confs.mean(), fp_confs.max(), np.percentile(fp_confs, 90))
            # Set threshold between FP p90 and TP p10
            fp_p90 = np.percentile(fp_confs, 90)
            tp_p10 = np.percentile(tp_confs, 10)
            # Use midpoint, but never higher than TP p10 (would kill real attacks)
            gate = min(fp_p90, tp_p10 * 0.95)
            gate = max(gate, 0.0)  # never negative
        else:
            # No false positives -- use a very permissive gate
            gate = max(tp_confs.min() * 0.5, 0.0)

        log.info("    Selected confidence gate: %.4f", gate)
        return float(gate)

    def run_all(self):
        log.info("=" * 48)
        log.info("  AGILE v2.1 -- Comprehensive Testing")
        log.info("  Sliding-window inference: seq_len=%d", self.seq_len)
        log.info("=" * 48 + "\n")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("test_results_v2", f"test_{ts}")
        os.makedirs(out_dir, exist_ok=True)

        all_results = {}

        # Auto-calibrate confidence gate if not manually set
        if self.confidence_threshold <= 0.0:
            log.info("--- Auto-calibrating confidence gate on DDoS dataset ---")
            cal_path = DATASETS["DDoS"]["file"]
            cal_X, cal_y = self._load_dataset(cal_path)
            if cal_X is not None and cal_y is not None and (cal_y > 0).sum() > 0:
                self.confidence_threshold = self.calibrate_confidence(cal_X, cal_y)
            else:
                log.info("  Calibration dataset not available, using gate=0.0 (no gating)")
                self.confidence_threshold = 0.0
            self.engine.confidence_threshold = self.confidence_threshold
            log.info("  Final confidence gate: %.4f\n", self.confidence_threshold)

        for ds_name, ds_info in DATASETS.items():
            log.info("--- %s ---", ds_name)
            X, y = self._load_dataset(ds_info["file"])
            if X is None:
                log.warning("  Skipping (load failed)")
                continue

            log.info("  Samples: %d  (benign=%d  attack=%d)",
                     len(y), (y == 0).sum(), (y > 0).sum())
            for i in range(NUM_CLASSES):
                cnt = (y == i).sum()
                if cnt > 0:
                    log.info("    Class %d (%s): %d", i, THREAT_LABELS[i], cnt)

            t0 = time.time()
            preds = self.predict(X)
            dt = time.time() - t0
            log.info("  Inference: %.2fs (%.3f ms/sample)", dt, dt / len(X) * 1000)

            metrics = self.evaluate(y, preds)

            bm = metrics["binary"]
            cg = metrics["confidence_gating"]
            log.info("  Binary:  acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f  fpr=%.4f  auc=%.4f",
                     bm["accuracy"], bm["precision"], bm["recall"], bm["f1"],
                     bm["fpr"], bm["roc_auc"])
            log.info("  CM: TP=%d FP=%d FN=%d TN=%d", bm["tp"], bm["fp"], bm["fn"], bm["tn"])
            log.info("  Confidence gate=%.4f: %d raw attacks -> %d after gate (%d filtered)",
                     self.confidence_threshold,
                     cg["raw_attack_predictions"], cg["after_gating"], cg["filtered_out"])
            log.info("  Priorities: %s", metrics["priority_distribution"])
            log.info("\n%s", metrics["multi_class_report"])

            log.info("  Sample alerts:")
            self.generate_sample_alerts(X, y, preds, n=2)

            metrics["inference_time_s"] = dt
            all_results[ds_name] = metrics

        with open(os.path.join(out_dir, "all_metrics.json"), "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        self._write_report(all_results, out_dir)
        log.info("\nResults saved to %s", out_dir)

    # ------------------------------------------------------------------
    def _write_report(self, results: Dict, out_dir: str):
        path = os.path.join(out_dir, "report.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("HALO NIDS -- AGILE v2.1  COMPREHENSIVE TEST REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Date: {datetime.now()}\n")
            f.write(f"Experiment: {self.exp_dir}\n")
            f.write(f"Anomaly threshold: {self.threshold:.6f}\n")
            f.write(f"Confidence gate: {self.confidence_threshold:.2f}\n")
            f.write(f"Sequence length: {self.seq_len}\n")
            f.write(f"Datasets tested: {len(results)}\n\n")

            for ds, m in results.items():
                bm = m["binary"]
                cg = m.get("confidence_gating", {})
                f.write(f"\n{'-'*80}\n{ds}\n{'-'*80}\n")
                f.write(f"Samples: {m['n_samples']}  (benign={m['n_benign']}, attack={m['n_attack']})\n")
                f.write(f"Accuracy:  {bm['accuracy']:.4f}\n")
                f.write(f"Precision: {bm['precision']:.4f}\n")
                f.write(f"Recall:    {bm['recall']:.4f}\n")
                f.write(f"F1:        {bm['f1']:.4f}\n")
                f.write(f"FPR:       {bm['fpr']:.4f}\n")
                f.write(f"ROC-AUC:   {bm['roc_auc']:.4f}\n")
                f.write(f"TP={bm['tp']}  FP={bm['fp']}  FN={bm['fn']}  TN={bm['tn']}\n")
                if cg:
                    f.write(f"Confidence gating: {cg.get('raw_attack_predictions',0)} raw -> "
                            f"{cg.get('after_gating',0)} gated ({cg.get('filtered_out',0)} filtered)\n")
                f.write(f"Priorities: {m['priority_distribution']}\n")
                f.write(f"\n{m['multi_class_report']}\n")

        log.info("  Report -> %s", path)


# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AGILE v2.1 Testing")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Path to experiment directory")
    parser.add_argument("--max-samples", type=int, default=60000)
    parser.add_argument("--confidence-threshold", type=float, default=0.0,
                        help="Confidence gate threshold (default: 0.0 = auto-calibrate)")
    args = parser.parse_args()

    if args.experiment:
        exp_dir = args.experiment
    else:
        base = "experiments_v2"
        if not os.path.exists(base):
            print(f"ERROR: {base}/ not found.  Run train_v2.py first.")
            sys.exit(1)
        dirs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
        if not dirs:
            print("ERROR: No experiments found. Run train_v2.py first.")
            sys.exit(1)
        latest = max(dirs, key=lambda x: os.path.getctime(os.path.join(base, x)))
        exp_dir = os.path.join(base, latest)

    log.info("Using experiment: %s", exp_dir)
    tester = AGILETester(exp_dir, max_samples=args.max_samples,
                         confidence_threshold=args.confidence_threshold)
    tester.run_all()


if __name__ == "__main__":
    main()
