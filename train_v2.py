"""
HALO NIDS — AGILE v2  Complete Training Pipeline
=================================================
Trains the full system described in Algorithm 1:

  Phase 1 — Feature extraction & preprocessing  (CICIDS2017 CSV → scaled tensors)
  Phase 2 — Stream A: Autoencoder (unsupervised, benign-only)
            Stream B: Full TAGN with graph attention (supervised, 7-class)
  Phase 3 — Correlation engine (learned fusion + rule-based priority)
  Phase 4 — Alert generator is rule-based / LLM — no training needed
  Phase 5 — End-to-end validation with REAL computed metrics

Run:
    python train_v2.py
"""

import os, sys, gc, json, time, logging, warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report,
)
import joblib

# ── our v2 modules ──
from models_v2.autoencoder import Autoencoder
from models_v2.tagn_network import TAGNNetwork, create_tagn_model, THREAT_LABELS, NUM_CLASSES
from models_v2.correlation_engine import create_correlation_engine

warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────────────────────
# Label mapping for CICIDS2017
# ──────────────────────────────────────────────────────────────────────────────

# Map the raw string labels in the CSVs to our 7-class indices
LABEL_MAP = {
    "BENIGN":                               0,
    # DDoS  (class 1)
    "DDoS":                                 1,
    "DoS Hulk":                             1,
    "DoS GoldenEye":                        1,
    "DoS slowloris":                        1,
    "DoS Slowhttptest":                     1,
    # PortScan  (class 2)
    "PortScan":                             2,
    # Web Attacks  (class 3) — handle en-dash, hyphen, and latin-1 dash
    "Web Attack - Brute Force":             3,
    "Web Attack - XSS":                     3,
    "Web Attack - Sql Injection":           3,
    "Web Attack \u2013 Brute Force":        3,   # en-dash
    "Web Attack \u2013 XSS":                3,
    "Web Attack \u2013 Sql Injection":      3,
    "Web Attack \x96 Brute Force":          3,   # latin-1 dash
    "Web Attack \x96 XSS":                  3,
    "Web Attack \x96 Sql Injection":        3,
    # Infiltration  (class 4)
    "Infiltration":                         4,
    "Heartbleed":                           4,
    # Botnet  (class 5)
    "Bot":                                  5,
    # Probe / brute-force  (class 6)
    "FTP-Patator":                          6,
    "SSH-Patator":                          6,
}


def map_labels(raw_labels: pd.Series) -> np.ndarray:
    """Convert raw CICIDS2017 label strings to 0..6 integer labels."""
    raw = raw_labels.str.strip()
    mapped = raw.map(LABEL_MAP)

    # Fuzzy fallback for labels that differ only by dash encoding
    unmapped_mask = mapped.isna()
    if unmapped_mask.any():
        for idx in raw[unmapped_mask].index:
            val = str(raw[idx])
            low = val.lower()
            if pd.isna(raw[idx]):
                mapped[idx] = 0          # NaN -> BENIGN
            elif "web attack" in low:
                mapped[idx] = 3          # any Web Attack variant
            elif low.startswith("dos") or "ddos" in low:
                mapped[idx] = 1          # any DoS variant
            elif "infiltr" in low:
                mapped[idx] = 4
            elif "bot" in low:
                mapped[idx] = 5
            elif "patator" in low or "ssh" in low or "ftp" in low:
                mapped[idx] = 6
            else:
                mapped[idx] = 0          # truly unknown -> BENIGN

    still_na = mapped.isna().sum()
    if still_na > 0:
        logging.warning("Unmapped labels (%d rows) treated as BENIGN", still_na)
        mapped = mapped.fillna(0)
    return mapped.astype(int).values


# ──────────────────────────────────────────────────────────────────────────────
# CSV files
# ──────────────────────────────────────────────────────────────────────────────

TRAIN_FILES = [
    "GeneratedLabelledFlows/TrafficLabelling/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "GeneratedLabelledFlows/TrafficLabelling/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "GeneratedLabelledFlows/TrafficLabelling/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "GeneratedLabelledFlows/TrafficLabelling/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "GeneratedLabelledFlows/TrafficLabelling/Monday-WorkingHours.pcap_ISCX.csv",
    "GeneratedLabelledFlows/TrafficLabelling/Tuesday-WorkingHours.pcap_ISCX.csv",
    "GeneratedLabelledFlows/TrafficLabelling/Wednesday-workingHours.pcap_ISCX.csv",
]

TEST_FILES = [
    "GeneratedLabelledFlows/TrafficLabelling/Friday-WorkingHours-Morning.pcap_ISCX.csv",
]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def read_csv(path: str, max_rows=None) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1", "iso-8859-1", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False, nrows=max_rows)
            df.columns = df.columns.str.strip()
            return df
        except Exception:
            continue
    raise IOError(f"Cannot read {path}")


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    num = num.replace([np.inf, -np.inf], np.nan).dropna()
    num = num.clip(-1e6, 1e6)
    return num


def detect_device():
    """Return best available torch device."""
    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device(), "DirectML"
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda"), f"CUDA ({torch.cuda.get_device_name(0)})"
    return torch.device("cpu"), "CPU"


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class AGILETrainer:

    def __init__(self, experiment_name: str = "agile_v2"):
        self.t0 = time.time()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join("experiments_v2", f"{experiment_name}_{ts}")
        os.makedirs(self.exp_dir, exist_ok=True)

        # logging — use utf-8 for both file and console to avoid cp1252 errors on Windows
        file_handler = logging.FileHandler(os.path.join(self.exp_dir, "training.log"), encoding="utf-8")
        stream_handler = logging.StreamHandler()
        stream_handler.setStream(open(sys.stdout.fileno(), mode='w', encoding='utf-8', errors='replace', closefd=False))
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)-8s  %(message)s",
            handlers=[file_handler, stream_handler],
        )
        self.log = logging.getLogger("AGILEv2")

        # device — LSTM not supported on DirectML, so TAGN always on CPU
        self.gpu, self.gpu_name = detect_device()
        self.cpu = torch.device("cpu")
        self.log.info("GPU device: %s  |  TAGN will use CPU (LSTM constraint)", self.gpu_name)

        # config
        self.cfg = {
            "ae_epochs": 20, "ae_lr": 1e-3, "ae_bs": 256,
            "tagn_epochs": 60, "tagn_lr": 5e-4, "tagn_bs": 128,
            "tagn_seq_len": 25, "tagn_patience": 10,
            "corr_epochs": 15, "corr_lr": 1e-4, "corr_bs": 64,
        }
        self.input_dim: int = 0
        self.scaler: Optional[StandardScaler] = None

    # ------------------------------------------------------------------
    # Phase 1: Data loading & preprocessing
    # ------------------------------------------------------------------
    def load_data(self) -> Dict:
        self.log.info("--- Phase 1: Loading & preprocessing data ---")

        frames = []
        for f in TRAIN_FILES:
            if os.path.exists(f):
                df = read_csv(f)
                self.log.info("  %s  ->  %d rows", os.path.basename(f), len(df))
                frames.append(df)
            else:
                self.log.warning("  MISSING: %s", f)
        if not frames:
            raise FileNotFoundError("No training CSVs found")

        combined = pd.concat(frames, ignore_index=True)
        self.log.info("Combined training data: %d rows", len(combined))

        # extract labels
        raw_labels = combined["Label"]
        labels = map_labels(raw_labels)

        # show class distribution
        for i, name in enumerate(THREAT_LABELS):
            cnt = (labels == i).sum()
            if cnt > 0:
                self.log.info("  Class %d (%s): %d samples", i, name, cnt)

        # clean numeric
        numeric = clean_numeric(combined.drop(columns=["Label"], errors="ignore"))
        # align labels to surviving rows
        valid_idx = numeric.index
        labels = labels[valid_idx]
        numeric = numeric.reset_index(drop=True)

        self.input_dim = numeric.shape[1]
        self.log.info("Input dimension: %d", self.input_dim)

        # fit scaler on BENIGN only
        benign_mask = labels == 0
        scaler = StandardScaler()
        scaler.fit(numeric.values[benign_mask])
        self.scaler = scaler
        joblib.dump(scaler, os.path.join(self.exp_dir, "scaler.pkl"))

        X_all = scaler.transform(numeric.values).astype(np.float32)

        # load test data
        test_frames = []
        for f in TEST_FILES:
            if os.path.exists(f):
                test_frames.append(read_csv(f))
        test_df = pd.concat(test_frames, ignore_index=True) if test_frames else None

        self.log.info("Phase 1 complete.\n")
        return {
            "X_all": X_all,
            "labels": labels,
            "test_df": test_df,
        }

    # ------------------------------------------------------------------
    # Phase 2A: Train Autoencoder
    # ------------------------------------------------------------------
    def train_autoencoder(self, X_all: np.ndarray, labels: np.ndarray) -> Autoencoder:
        self.log.info("--- Phase 2A: Training Autoencoder (Stream A) ---")
        X_benign = X_all[labels == 0]
        self.log.info("  Training on %d benign samples", len(X_benign))

        device = self.gpu  # autoencoder is MLP — works on DirectML
        # fallback to CPU if device fails
        try:
            torch.zeros(1, device=device)
        except Exception:
            device = self.cpu

        ae = Autoencoder(self.input_dim).to(device)
        opt = optim.Adam(ae.parameters(), lr=self.cfg["ae_lr"], weight_decay=1e-5)
        crit = nn.MSELoss()
        loader = DataLoader(
            TensorDataset(torch.tensor(X_benign)),
            batch_size=self.cfg["ae_bs"], shuffle=True,
        )

        ae.train()
        for ep in range(1, self.cfg["ae_epochs"] + 1):
            ep_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(device)
                opt.zero_grad()
                loss = crit(ae(batch), batch)
                loss.backward()
                opt.step()
                ep_loss += loss.item()
            if ep % 5 == 0 or ep == 1:
                self.log.info("  Epoch %2d/%d  loss=%.6f", ep, self.cfg["ae_epochs"], ep_loss / len(loader))

        ae = ae.cpu()
        path = os.path.join(self.exp_dir, "autoencoder.pt")
        torch.save(ae.state_dict(), path)
        self.log.info("  Saved -> %s\n", path)
        return ae

    # ------------------------------------------------------------------
    # Phase 2B: Train TAGN (multi-class, with graph attention)
    # ------------------------------------------------------------------
    def _make_sequences(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        """
        Create non-overlapping sequences.  Each sequence gets the label of
        the *majority* class within it (so we don't mix benign/attack labels).
        """
        seqs, labs = [], []
        for start in range(0, len(X) - seq_len + 1, seq_len):
            chunk_x = X[start : start + seq_len]
            chunk_y = y[start : start + seq_len]
            # majority vote label
            counts = np.bincount(chunk_y, minlength=NUM_CLASSES)
            lab = int(counts.argmax())
            seqs.append(chunk_x)
            labs.append(lab)
        return np.array(seqs, dtype=np.float32), np.array(labs, dtype=np.int64)

    def train_tagn(self, X_all: np.ndarray, labels: np.ndarray) -> TAGNNetwork:
        self.log.info("--- Phase 2B: Training TAGN (Stream B -- 7-class) ---")
        seq_len = self.cfg["tagn_seq_len"]

        # sort data by class so sequences are coherent
        order = np.argsort(labels, kind="stable")
        X_sorted = X_all[order]
        y_sorted = labels[order]

        X_seq, y_seq = self._make_sequences(X_sorted, y_sorted, seq_len)
        self.log.info("  Created %d sequences (len=%d)", len(X_seq), seq_len)
        for i, name in enumerate(THREAT_LABELS):
            cnt = (y_seq == i).sum()
            if cnt > 0:
                self.log.info("    Class %d (%s): %d seqs", i, name, cnt)

        # stratified split
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_seq, y_seq, test_size=0.2, stratify=y_seq, random_state=42,
        )
        self.log.info("  Train: %d  Val: %d", len(X_tr), len(X_va))

        # class weights for imbalanced data
        counts = np.bincount(y_tr, minlength=NUM_CLASSES).astype(np.float32)
        counts = np.maximum(counts, 1.0)
        weights = 1.0 / counts
        weights = weights / weights.sum() * NUM_CLASSES
        class_weights = torch.tensor(weights, dtype=torch.float32)
        self.log.info("  Class weights: %s", [f"{w:.3f}" for w in weights])

        tr_loader = DataLoader(
            TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
            batch_size=self.cfg["tagn_bs"], shuffle=True,
        )
        va_loader = DataLoader(
            TensorDataset(torch.tensor(X_va), torch.tensor(y_va)),
            batch_size=self.cfg["tagn_bs"], shuffle=False,
        )

        device = self.cpu   # LSTM → CPU
        model = create_tagn_model(
            input_dim=self.input_dim, hidden_dim=128, n_heads=4, num_classes=NUM_CLASSES,
        ).to(device)

        opt = optim.AdamW(model.parameters(), lr=self.cfg["tagn_lr"], weight_decay=1e-3)
        crit = nn.CrossEntropyLoss(weight=class_weights.to(device))
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

        best_val_acc = 0.0
        patience_ctr = 0
        best_path = os.path.join(self.exp_dir, "tagn_best.pt")

        for ep in range(1, self.cfg["tagn_epochs"] + 1):
            # train
            model.train()
            tr_loss, tr_correct, tr_total = 0.0, 0, 0
            for bx, by in tr_loader:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                out = model(bx)
                logits = out["classification"]["logits"]
                loss = crit(logits, by)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tr_loss += loss.item()
                tr_correct += (logits.argmax(1) == by).sum().item()
                tr_total += by.size(0)

            # validate
            model.eval()
            va_loss, va_correct, va_total = 0.0, 0, 0
            with torch.no_grad():
                for bx, by in va_loader:
                    bx, by = bx.to(device), by.to(device)
                    out = model(bx)
                    logits = out["classification"]["logits"]
                    va_loss += crit(logits, by).item()
                    va_correct += (logits.argmax(1) == by).sum().item()
                    va_total += by.size(0)

            tr_acc = 100 * tr_correct / max(tr_total, 1)
            va_acc = 100 * va_correct / max(va_total, 1)
            sched.step(va_loss / max(len(va_loader), 1))

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                patience_ctr = 0
                torch.save(model.state_dict(), best_path)
            else:
                patience_ctr += 1

            if ep % 5 == 0 or ep == 1 or patience_ctr == 0:
                self.log.info(
                    "  Epoch %2d/%d  tr_loss=%.4f  tr_acc=%.2f%%  va_acc=%.2f%%  best=%.2f%%",
                    ep, self.cfg["tagn_epochs"], tr_loss / len(tr_loader), tr_acc, va_acc, best_val_acc,
                )

            if patience_ctr >= self.cfg["tagn_patience"]:
                self.log.info("  Early stopping at epoch %d", ep)
                break

        # reload best
        model.load_state_dict(torch.load(best_path, weights_only=False))
        self.log.info("  Best validation accuracy: %.2f%%", best_val_acc)
        self.log.info("  Saved -> %s\n", best_path)
        return model

    # ------------------------------------------------------------------
    # Phase 3: Train Correlation Engine
    # ------------------------------------------------------------------
    def train_correlation(
        self, ae: Autoencoder, tagn: TAGNNetwork,
        X_all: np.ndarray, labels: np.ndarray,
    ):
        self.log.info("--- Phase 3: Training Correlation Engine ---")
        ae.eval()
        tagn.eval()

        seq_len = self.cfg["tagn_seq_len"]
        order = np.argsort(labels, kind="stable")
        X_sorted = X_all[order]
        y_sorted = labels[order]

        X_seq, y_seq = self._make_sequences(X_sorted, y_sorted, seq_len)
        # binary target for fusion score: 1 if attack, 0 if benign
        y_binary = (y_seq > 0).astype(np.float32)

        self.log.info("  Sequences: %d  (benign: %d  attack: %d)",
                      len(y_seq), (y_binary == 0).sum(), (y_binary == 1).sum())

        engine = create_correlation_engine(hidden=64)
        opt = optim.AdamW(engine.parameters(), lr=self.cfg["corr_lr"])
        bce = nn.BCELoss()

        loader = DataLoader(
            TensorDataset(
                torch.tensor(X_seq),
                torch.tensor(y_seq),
                torch.tensor(y_binary).unsqueeze(1),
            ),
            batch_size=self.cfg["corr_bs"], shuffle=True,
        )

        for ep in range(1, self.cfg["corr_epochs"] + 1):
            total_loss = 0.0
            for seq_batch, cls_batch, bin_batch in loader:
                with torch.no_grad():
                    # stream A
                    ae_in = seq_batch.mean(dim=1)
                    recon = ae(ae_in)
                    anom = Autoencoder.reconstruction_error(ae_in, recon).unsqueeze(1)  # (B,1)
                    # stream B
                    tagn_out = tagn(seq_batch)
                    corr_feat = tagn_out["correlation_features"]          # (B,16)
                    conf = tagn_out["classification"]["confidence_score"].unsqueeze(1)
                    pred_cls = tagn_out["classification"]["predicted_class"]

                opt.zero_grad()
                results = engine(anom, corr_feat, conf, pred_cls)
                loss = bce(results["fusion_score"], bin_batch)
                loss.backward()
                opt.step()
                total_loss += loss.item()

            if ep % 5 == 0 or ep == 1:
                self.log.info("  Epoch %2d/%d  loss=%.4f", ep, self.cfg["corr_epochs"],
                              total_loss / len(loader))

        path = os.path.join(self.exp_dir, "correlation_engine.pt")
        torch.save(engine.state_dict(), path)
        self.log.info("  Saved -> %s\n", path)
        return engine

    # ------------------------------------------------------------------
    # Threshold calibration
    # ------------------------------------------------------------------
    def calibrate_threshold(
        self, ae: Autoencoder, tagn: TAGNNetwork,
        X_all: np.ndarray, labels: np.ndarray,
    ) -> float:
        self.log.info("--- Calibrating anomaly threshold ---")
        ae.eval(); tagn.eval()

        # use benign data
        X_benign = X_all[labels == 0]
        # sample at most 50k for speed
        if len(X_benign) > 50000:
            idx = np.random.choice(len(X_benign), 50000, replace=False)
            X_benign = X_benign[idx]

        scores = []
        bs = 512
        for i in range(0, len(X_benign), bs):
            batch = torch.tensor(X_benign[i:i+bs])
            with torch.no_grad():
                recon = ae(batch)
                s = Autoencoder.reconstruction_error(batch, recon).numpy()
                scores.append(s)
        scores = np.concatenate(scores)

        p95 = float(np.percentile(scores, 95))
        p99 = float(np.percentile(scores, 99))
        self.log.info("  Benign anomaly-score  mean=%.6f  p95=%.6f  p99=%.6f", scores.mean(), p95, p99)
        threshold = p99          # use 99th percentile for lower FPR
        self.log.info("  Selected threshold = %.6f  (99th percentile)\n", threshold)
        return threshold

    # ------------------------------------------------------------------
    # Phase 5: Real validation
    # ------------------------------------------------------------------
    def validate(
        self, ae: Autoencoder, tagn: TAGNNetwork, engine,
        test_df: Optional[pd.DataFrame], threshold: float,
    ) -> Dict:
        self.log.info("--- Phase 5: Validation on test data (sliding-window + conf-gate) ---")

        if test_df is None or test_df.empty:
            self.log.warning("  No test data -- skipping validation")
            return {}

        raw_labels = test_df["Label"]
        y_true = map_labels(raw_labels)
        numeric = clean_numeric(test_df.drop(columns=["Label"], errors="ignore"))
        valid_idx = numeric.index
        y_true = y_true[valid_idx]
        numeric = numeric.reset_index(drop=True)

        if numeric.shape[1] > self.input_dim:
            numeric = numeric.iloc[:, :self.input_dim]
        elif numeric.shape[1] < self.input_dim:
            pad = pd.DataFrame(np.zeros((len(numeric), self.input_dim - numeric.shape[1])))
            numeric = pd.concat([numeric, pad], axis=1)

        X_test = self.scaler.transform(numeric.values).astype(np.float32)
        n = len(X_test)
        self.log.info("  Test samples: %d", n)

        ae.eval(); tagn.eval(); engine.eval()
        engine.anomaly_threshold = threshold

        seq_len = self.cfg["tagn_seq_len"]
        bs = 512

        # Per-flow anomaly scores
        y_anomaly = np.zeros(n, dtype=np.float32)
        for i in range(0, n, bs):
            batch = torch.tensor(X_test[i:i+bs])
            with torch.no_grad():
                y_anomaly[i:i+batch.size(0)] = Autoencoder.reconstruction_error(batch, ae(batch)).numpy()

        # Sliding-window TAGN
        y_pred_cls = np.zeros(n, dtype=int)
        y_probs = np.zeros((n, NUM_CLASSES), dtype=np.float32)
        y_conf = np.zeros(n, dtype=np.float32)
        y_corr = np.zeros((n, 16), dtype=np.float32)
        n_win = n // seq_len
        rem = n % seq_len
        if n_win > 0:
            for ws in range(0, n_win, 64):
                we = min(ws + 64, n_win)
                seqs = [X_test[w*seq_len:(w+1)*seq_len] for w in range(ws, we)]
                with torch.no_grad():
                    out = tagn(torch.tensor(np.array(seqs)))
                for j, w in enumerate(range(ws, we)):
                    s, e = w * seq_len, (w+1) * seq_len
                    y_pred_cls[s:e] = out["classification"]["predicted_class"][j].item()
                    y_probs[s:e] = out["classification"]["class_probabilities"][j].numpy()
                    y_conf[s:e] = out["classification"]["confidence_score"][j].item()
                    y_corr[s:e] = out["correlation_features"][j].numpy()
        if rem > 0:
            tail = n_win * seq_len
            if n_win > 0:
                ls = (n_win - 1) * seq_len
                y_pred_cls[tail:] = y_pred_cls[ls]; y_probs[tail:] = y_probs[ls]
                y_conf[tail:] = y_conf[ls]; y_corr[tail:] = y_corr[ls]
            else:
                padded = np.zeros((seq_len, X_test.shape[1]), dtype=np.float32)
                padded[:rem] = X_test[tail:]
                with torch.no_grad():
                    out = tagn(torch.tensor(padded).unsqueeze(0))
                y_pred_cls[tail:] = out["classification"]["predicted_class"].item()
                y_probs[tail:] = out["classification"]["class_probabilities"][0].numpy()
                y_conf[tail:] = out["classification"]["confidence_score"].item()
                y_corr[tail:] = out["correlation_features"][0].numpy()

        # Correlation engine with confidence gating
        priorities = np.zeros(n, dtype=int)
        gated_cls = np.zeros(n, dtype=int)
        for i in range(0, n, bs):
            end = min(i + bs, n)
            with torch.no_grad():
                c_out = engine(torch.tensor(y_anomaly[i:end]), torch.tensor(y_corr[i:end]),
                               torch.tensor(y_conf[i:end]), torch.tensor(y_pred_cls[i:end], dtype=torch.long))
            priorities[i:end] = c_out["priority"].numpy()
            gated_cls[i:end] = c_out["gated_class"].numpy()

        # Binary metrics using gated predictions
        y_true_bin = (y_true > 0).astype(int)
        y_pred_bin = (gated_cls > 0).astype(int)

        acc  = accuracy_score(y_true_bin, y_pred_bin)
        prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        rec  = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        f1   = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        cm   = confusion_matrix(y_true_bin, y_pred_bin)
        tn, fp, fn, tp = cm.ravel()
        fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        try:
            auc = roc_auc_score(y_true_bin, 1 - y_probs[:, 0])  # P(attack)
        except Exception:
            auc = 0.0

        self.log.info("  Binary metrics (benign vs attack):")
        self.log.info("    Accuracy : %.4f", acc)
        self.log.info("    Precision: %.4f", prec)
        self.log.info("    Recall   : %.4f", rec)
        self.log.info("    F1       : %.4f", f1)
        self.log.info("    FPR      : %.4f", fpr)
        self.log.info("    ROC-AUC  : %.4f", auc)
        self.log.info("    TP=%d  FP=%d  FN=%d  TN=%d", tp, fp, fn, tn)

        # --- multi-class report (using gated predictions) ---
        present = sorted(set(y_true) | set(gated_cls))
        target_names = [THREAT_LABELS[i] for i in present]
        cls_report = classification_report(
            y_true, gated_cls, labels=present, target_names=target_names, zero_division=0,
        )
        self.log.info("\n  Multi-class classification report:\n%s", cls_report)

        # priority distribution
        for p in range(4):
            pnames = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
            cnt = (priorities == p).sum()
            if cnt > 0:
                self.log.info("  Priority %s: %d alerts", pnames[p], cnt)

        metrics = {
            "binary": {
                "accuracy": float(acc), "precision": float(prec),
                "recall": float(rec), "f1": float(f1), "fpr": float(fpr),
                "roc_auc": float(auc),
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            },
            "multi_class_report": cls_report,
            "anomaly_threshold": threshold,
            "test_samples": len(X_test),
        }

        # save
        with open(os.path.join(self.exp_dir, "validation_results.json"), "w") as f:
            # cls_report is a string, fine for JSON
            json.dump(metrics, f, indent=2)
        self.log.info("  Validation results saved.\n")
        return metrics

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def run(self):
        self.log.info("=" * 48)
        self.log.info("  HALO NIDS -- AGILE v2  Training Pipeline")
        self.log.info("=" * 48 + "\n")

        data = self.load_data()
        X_all, labels = data["X_all"], data["labels"]

        ae   = self.train_autoencoder(X_all, labels)
        tagn = self.train_tagn(X_all, labels)
        eng  = self.train_correlation(ae, tagn, X_all, labels)

        threshold = self.calibrate_threshold(ae, tagn, X_all, labels)

        # save deployment config
        deploy = {
            "input_dim": self.input_dim,
            "num_classes": NUM_CLASSES,
            "anomaly_threshold": threshold,
            "seq_len": self.cfg["tagn_seq_len"],
            "model_files": {
                "autoencoder": "autoencoder.pt",
                "tagn": "tagn_best.pt",
                "correlation": "correlation_engine.pt",
                "scaler": "scaler.pkl",
            },
        }
        with open(os.path.join(self.exp_dir, "deploy_config.json"), "w") as f:
            json.dump(deploy, f, indent=2)

        # validate
        metrics = self.validate(ae, tagn, eng, data.get("test_df"), threshold)

        elapsed = (time.time() - self.t0) / 60
        self.log.info("--- Training complete in %.1f minutes ---", elapsed)
        self.log.info("Experiment directory: %s", self.exp_dir)

        # training report
        report = {
            "experiment": self.exp_dir,
            "duration_min": round(elapsed, 2),
            "input_dim": self.input_dim,
            "config": self.cfg,
            "anomaly_threshold": threshold,
            "validation": metrics,
        }
        with open(os.path.join(self.exp_dir, "training_report.json"), "w") as f:
            json.dump(report, f, indent=2, default=str)

        return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    trainer = AGILETrainer()
    trainer.run()


if __name__ == "__main__":
    main()
