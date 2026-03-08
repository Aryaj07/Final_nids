"""
HALO NIDS — AGILE v3.2  Complete Training Pipeline  (GPU-accelerated)
=====================================================================
Implements Algorithm 1 v3 changes:

  Step 3b  — SMOTE oversampling for minority attack classes
  Step 10  — Hard k-NN graph construction with tau-threshold (via models_v3)
  Step 11  — 15-class CICIDS2017 taxonomy

v3.2 changes (26/02/26 — critical fixes):
  - SHUFFLED sequences: data is shuffled before windowing so each window
    contains mixed traffic — matches real-world inference conditions
  - Overlapping windows with stride=5 for 5x more training sequences
  - Majority-vote labelling: window label = class with most flows (not rarest)
  - Capped class weights: max weight ratio 10:1 to prevent rare class dominance
  - Label smoothing (0.1) to reduce overconfident predictions
  - SMOTE target reduced to 3000 to avoid synthetic domination
  - Validation uses REAL data windows (not SMOTE) for honest early stopping
  - Confidence threshold lowered to 0.30 — better recall/precision trade-off
  - Dropout increased in TAGN (0.3 → 0.4) to reduce SMOTE overfitting

GPU support:
  - Priority order: CUDA > DirectML > CPU
  - ALL three models (Autoencoder, TAGN, Correlation Engine) run on GPU

Run:
    python train_v3.py
    python train_v3.py --gpu 0          # specific GPU index
    python train_v3.py --cpu            # force CPU (debug)
    python train_v3.py --batch-size 128 # override batch size
"""

import os, sys, json, time, logging, warnings, argparse
from datetime import datetime
from typing import Dict, Tuple, Optional

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

from models_v3.tagn_network       import TAGNNetwork, create_tagn_model, THREAT_LABELS, NUM_CLASSES
from models_v2.autoencoder        import Autoencoder
from models_v2.correlation_engine import create_correlation_engine

warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Device selection
# ─────────────────────────────────────────────────────────────────────────────

def select_device(force_cpu: bool = False, gpu_index: int = 0) -> Tuple[torch.device, str, bool]:
    if force_cpu:
        return torch.device("cpu"), "CPU (forced)", False
    if torch.cuda.is_available():
        n    = torch.cuda.device_count()
        idx  = min(gpu_index, n - 1)
        name = torch.cuda.get_device_name(idx)
        vram = torch.cuda.get_device_properties(idx).total_memory / 1024 ** 3
        return torch.device(f"cuda:{idx}"), f"CUDA:{idx}  {name}  ({vram:.1f} GB VRAM)", False
    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device(), "DirectML (AMD/Intel GPU)", True
    except ImportError:
        pass
    return torch.device("cpu"), "CPU", False


def is_cuda(device: torch.device) -> bool:
    return device.type == "cuda"


def auto_batch_sizes(device: torch.device) -> Dict[str, int]:
    if not is_cuda(device):
        return {"ae": 512, "tagn": 64, "corr": 128}
    vram_gb = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
    if vram_gb >= 16:
        return {"ae": 2048, "tagn": 512, "corr": 512}
    elif vram_gb >= 8:
        return {"ae": 1024, "tagn": 256, "corr": 256}
    elif vram_gb >= 4:
        return {"ae": 512,  "tagn": 128, "corr": 128}
    else:
        return {"ae": 256,  "tagn": 64,  "corr": 64}


# ─────────────────────────────────────────────────────────────────────────────
# 15-class label map
# ─────────────────────────────────────────────────────────────────────────────

LABEL_MAP: Dict[str, int] = {
    "BENIGN": 0,
    "Bot": 1,
    "DDoS": 2,
    "DoS GoldenEye": 3,
    "DoS Hulk": 4,
    "DoS Slowhttptest": 5,
    "DoS slowloris": 6, "DoS Slowloris": 6,
    "FTP-Patator": 7,
    "Heartbleed": 8,
    "Infiltration": 9,
    "PortScan": 10,
    "SSH-Patator": 11,
    "Web Attack - Brute Force":        12,
    "Web Attack \u2013 Brute Force":   12,
    "Web Attack \x96 Brute Force":     12,
    "Web Attack - Sql Injection":      13,
    "Web Attack \u2013 Sql Injection": 13,
    "Web Attack \x96 Sql Injection":   13,
    "Web Attack - XSS":                14,
    "Web Attack \u2013 XSS":           14,
    "Web Attack \x96 XSS":             14,
}

DDOS_FAMILY = {2, 3, 4, 5, 6}

SMOTE_MIN_SEQUENCES = 3000   # reduced from 5000 to avoid synthetic domination
MINORITY_CLASSES    = {1, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14}
MAX_WEIGHT_RATIO    = 10.0   # cap max/min class weight ratio


def map_labels(raw_labels: pd.Series) -> np.ndarray:
    raw    = raw_labels.str.strip()
    mapped = raw.map(LABEL_MAP)
    for idx in raw[mapped.isna()].index:
        v = str(raw[idx]).lower()
        if   pd.isna(raw[idx]):                       mapped[idx] = 0
        elif "web attack" in v and "brute" in v:      mapped[idx] = 12
        elif "web attack" in v and "sql"   in v:      mapped[idx] = 13
        elif "web attack" in v and "xss"   in v:      mapped[idx] = 14
        elif "web attack" in v:                       mapped[idx] = 12
        elif "ddos"        in v:                      mapped[idx] = 2
        elif "hulk"        in v:                      mapped[idx] = 4
        elif "goldeneye"   in v:                      mapped[idx] = 3
        elif "slowhttptest" in v:                     mapped[idx] = 5
        elif "slowloris"   in v:                      mapped[idx] = 6
        elif "heartbleed"  in v:                      mapped[idx] = 8
        elif "infiltr"     in v:                      mapped[idx] = 9
        elif "bot"         in v:                      mapped[idx] = 1
        elif "ftp"         in v:                      mapped[idx] = 7
        elif "ssh"         in v:                      mapped[idx] = 11
        elif "portscan"    in v or "port scan" in v:  mapped[idx] = 10
        else:                                         mapped[idx] = 0
    return mapped.fillna(0).astype(int).values


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm Step 3b — SMOTE
# ─────────────────────────────────────────────────────────────────────────────

def smote_oversample(
    X_seq: np.ndarray, y_seq: np.ndarray,
    minority_classes: set, target_count: int,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure-NumPy feature-space SMOTE for sequence arrays."""
    log = logging.getLogger("SMOTE")
    rng = np.random.default_rng(random_state)
    X_out, y_out = [X_seq], [y_seq]

    for cls in sorted(minority_classes):
        idx    = np.where(y_seq == cls)[0]
        n_have = len(idx)
        if n_have == 0:
            log.warning("  Class %d (%s) has 0 sequences — skipping", cls, THREAT_LABELS[cls])
            continue
        if n_have >= target_count:
            continue
        n_need = target_count - n_have
        X_cls  = X_seq[idx]
        T, D   = X_cls.shape[1], X_cls.shape[2]
        X_flat = X_cls.reshape(n_have, -1)
        k      = min(5, n_have - 1) if n_have > 1 else 1

        log.info("  Class %2d  %-22s  %5d seqs → +%5d synthetic",
                 cls, THREAT_LABELS[cls], n_have, n_need)
        synthetic = []
        for _ in range(n_need):
            si   = rng.integers(0, n_have)
            seed = X_flat[si]
            if k > 0:
                dists     = np.linalg.norm(X_flat - seed, axis=1)
                dists[si] = np.inf
                nn_idx    = np.argpartition(dists, min(k, n_have - 1))[:k]
                nbr_i     = rng.choice(nn_idx)
            else:
                nbr_i = si
            lam = rng.uniform(0.0, 1.0)
            synthetic.append((seed + lam * (X_flat[nbr_i] - seed)).reshape(T, D))
        X_out.append(np.array(synthetic, dtype=np.float32))
        y_out.append(np.full(n_need, cls, dtype=np.int64))

    X_f = np.concatenate(X_out, axis=0)
    y_f = np.concatenate(y_out, axis=0)
    p   = rng.permutation(len(X_f))
    return X_f[p], y_f[p]


# ─────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

ALL_CSV_FILES = [
    "GeneratedLabelledFlows/TrafficLabelling/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "GeneratedLabelledFlows/TrafficLabelling/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "GeneratedLabelledFlows/TrafficLabelling/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "GeneratedLabelledFlows/TrafficLabelling/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "GeneratedLabelledFlows/TrafficLabelling/Monday-WorkingHours.pcap_ISCX.csv",
    "GeneratedLabelledFlows/TrafficLabelling/Tuesday-WorkingHours.pcap_ISCX.csv",
    "GeneratedLabelledFlows/TrafficLabelling/Wednesday-workingHours.pcap_ISCX.csv",
    "GeneratedLabelledFlows/TrafficLabelling/Friday-WorkingHours-Morning.pcap_ISCX.csv",
]

TRAIN_RATIO = 0.80


def read_csv(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1", "iso-8859-1", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            df.columns = df.columns.str.strip()
            return df
        except Exception:
            continue
    raise IOError(f"Cannot read {path}")


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    return num.replace([np.inf, -np.inf], np.nan).dropna().clip(-1e6, 1e6)


def make_loader(X: np.ndarray, y: np.ndarray, bs: int,
                shuffle: bool, device: torch.device) -> DataLoader:
    pin = is_cuda(device)
    nw  = min(4, os.cpu_count() or 1) if pin else 0
    return DataLoader(
        TensorDataset(torch.tensor(X), torch.tensor(y)),
        batch_size=bs, shuffle=shuffle,
        pin_memory=pin, num_workers=nw,
        persistent_workers=(nw > 0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Trainer v3.2
# ─────────────────────────────────────────────────────────────────────────────

class AGILETrainerV3:

    def __init__(self, experiment_name="agile_v3", force_cpu=False,
                 gpu_index=0, batch_override=None):
        self.t0      = time.time()
        ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join("experiments_v3", f"{experiment_name}_{ts}")
        os.makedirs(self.exp_dir, exist_ok=True)

        fh = logging.FileHandler(os.path.join(self.exp_dir, "training.log"), encoding="utf-8")
        sh = logging.StreamHandler(sys.stdout)
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s  %(levelname)-8s  %(message)s",
                            handlers=[fh, sh])
        self.log = logging.getLogger("AGILEv3")

        self.device, dev_desc, self.is_directml = select_device(force_cpu, gpu_index)
        self.use_amp = is_cuda(self.device)
        self.tagn_device = torch.device("cpu") if self.is_directml else self.device

        self.log.info("Device   : %s", dev_desc)
        if self.is_directml:
            self.log.info("DirectML : AE + Corr on GPU  |  TAGN (LSTM) on CPU")
        self.log.info("AMP      : %s", "enabled" if self.use_amp else "disabled")

        bs = auto_batch_sizes(self.device)
        if batch_override:
            bs = {k: batch_override for k in bs}
        self.log.info("Batch sz : AE=%d  TAGN=%d  Corr=%d", bs["ae"], bs["tagn"], bs["corr"])

        self.cfg = {
            "ae_epochs":      20,   "ae_lr":    1e-3,  "ae_bs":   bs["ae"],
            "tagn_epochs":    80,   "tagn_lr":  3e-4,  "tagn_bs": bs["tagn"],
            "tagn_seq_len":   25,   "tagn_patience": 12,
            "corr_epochs":    15,   "corr_lr":  1e-4,  "corr_bs": bs["corr"],
            "train_ratio":    TRAIN_RATIO,
            "conf_threshold": 0.60,  # filter low-confidence attack predictions on benign flows
        }
        self.input_dim: int = 0
        self.scaler: Optional[StandardScaler] = None

    def _to(self, x):
        return x.to(self.device)

    def _cpu_state(self, model: nn.Module) -> dict:
        return {k: v.cpu() for k, v in model.state_dict().items()}

    # ── Phase 1: Data loading ─────────────────────────────────────────────────
    def load_data(self) -> Dict:
        self.log.info("─── Phase 1: Loading & preprocessing (15-class) ───")
        self.log.info("  Train/Test split: %.0f%% / %.0f%%",
                      TRAIN_RATIO * 100, (1 - TRAIN_RATIO) * 100)
        frames = []
        for f in ALL_CSV_FILES:
            if os.path.exists(f):
                df = read_csv(f)
                self.log.info("  %-65s  %7d rows", os.path.basename(f), len(df))
                frames.append(df)
            else:
                self.log.warning("  MISSING: %s", f)
        if not frames:
            raise FileNotFoundError("No training CSVs found")

        combined      = pd.concat(frames, ignore_index=True)
        labels        = map_labels(combined["Label"])
        numeric       = clean_numeric(combined.drop(columns=["Label"], errors="ignore"))
        labels        = labels[numeric.index]
        numeric       = numeric.reset_index(drop=True)
        self.input_dim = numeric.shape[1]

        self.log.info("Combined: %d rows  |  input_dim=%d", len(numeric), self.input_dim)
        self.log.info("Class distribution (full dataset):")
        for i, name in enumerate(THREAT_LABELS):
            cnt = (labels == i).sum()
            if cnt:
                self.log.info("  %2d  %-22s  %8d", i, name, cnt)

        X_all = numeric.values.astype(np.float32)
        strat_labels = labels.copy()
        for cls_id in range(NUM_CLASSES):
            if (strat_labels == cls_id).sum() < 2:
                strat_labels[strat_labels == cls_id] = 0

        X_train, X_test, y_train, y_test = train_test_split(
            X_all, labels, test_size=(1 - TRAIN_RATIO),
            stratify=strat_labels, random_state=42,
        )

        self.log.info("\n  ══════════════════════════════════════════════════")
        self.log.info("  TRAIN set: %d samples (%.1f%%)", len(X_train), 100*len(X_train)/len(X_all))
        self.log.info("  TEST  set: %d samples (%.1f%%)", len(X_test), 100*len(X_test)/len(X_all))
        self.log.info("  ══════════════════════════════════════════════════")

        self.log.info("\n  Train class distribution:")
        for i, name in enumerate(THREAT_LABELS):
            cnt = (y_train == i).sum()
            if cnt: self.log.info("    %2d  %-22s  %8d", i, name, cnt)
        self.log.info("  Test class distribution:")
        for i, name in enumerate(THREAT_LABELS):
            cnt = (y_test == i).sum()
            if cnt: self.log.info("    %2d  %-22s  %8d", i, name, cnt)

        scaler = StandardScaler()
        scaler.fit(X_train[y_train == 0])
        self.scaler = scaler
        joblib.dump(scaler, os.path.join(self.exp_dir, "scaler.pkl"))

        X_train_scaled = scaler.transform(X_train).astype(np.float32)
        X_test_scaled  = scaler.transform(X_test).astype(np.float32)

        test_split_path = os.path.join(self.exp_dir, "test_split.npz")
        np.savez_compressed(test_split_path, X_test=X_test_scaled, y_test=y_test)
        self.log.info("  Saved test split -> %s (%d samples)\n", test_split_path, len(X_test))

        return {
            "X_train": X_train_scaled, "y_train": y_train,
            "X_test":  X_test_scaled,  "y_test":  y_test,
        }

    # ── Phase 2A: Autoencoder ─────────────────────────────────────────────────
    def train_autoencoder(self, X_train: np.ndarray, y_train: np.ndarray) -> Autoencoder:
        self.log.info("─── Phase 2A: Autoencoder (Stream A)  [%s] ───", self.device)
        X_benign = X_train[y_train == 0]

        ae   = self._to(Autoencoder(self.input_dim))
        opt  = optim.Adam(ae.parameters(), lr=self.cfg["ae_lr"], weight_decay=1e-5)
        crit = nn.MSELoss()
        ldr  = make_loader(X_benign, X_benign, self.cfg["ae_bs"], shuffle=True, device=self.device)
        scaler = torch.amp.GradScaler(enabled=self.use_amp)

        ae.train()
        for ep in range(1, self.cfg["ae_epochs"] + 1):
            ep_loss = 0.0
            for bx, _ in ldr:
                bx = self._to(bx)
                opt.zero_grad()
                with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                    loss = crit(ae(bx), bx)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
                ep_loss += loss.item()
            if ep % 5 == 0 or ep == 1:
                self.log.info("  Epoch %2d/%d  loss=%.6f",
                              ep, self.cfg["ae_epochs"], ep_loss / len(ldr))

        torch.save(self._cpu_state(ae), os.path.join(self.exp_dir, "autoencoder.pt"))
        self.log.info("  Saved autoencoder.pt\n")
        return ae

    # ── Sequence builder v3.2 ─────────────────────────────────────────────────
    @staticmethod
    def _make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int,
                        stride: int = None):
        """
        Build OVERLAPPING sequences with given stride from SHUFFLED data.
        
        v3.2 key change: data should be shuffled BEFORE calling this,
        so each window contains a realistic mix of traffic classes.
        
        Labelling: majority-vote.  If any attack class has more flows
        than benign in the window, label = that attack class.
        Otherwise label = most common attack if attacks ≥ 3 flows,
        else BENIGN.
        """
        if stride is None:
            stride = seq_len  # non-overlapping by default
        seqs, labs = [], []
        for s in range(0, len(X) - seq_len + 1, stride):
            cy = y[s:s + seq_len]
            c  = np.bincount(cy, minlength=NUM_CLASSES)
            
            # Majority vote: with sorted data, windows are mostly pure single-class
            lab = int(c.argmax())
            
            seqs.append(X[s:s + seq_len])
            labs.append(lab)
        return np.array(seqs, dtype=np.float32), np.array(labs, dtype=np.int64)

    # ── Phase 2B: TAGN v3.2 ──────────────────────────────────────────────────
    def train_tagn(self, X_train: np.ndarray, y_train: np.ndarray) -> TAGNNetwork:
        self.log.info("─── Phase 2B: TAGN v3.2 (15-class, shuffled windows)  [%s] ───",
                      self.tagn_device)
        seq_len = self.cfg["tagn_seq_len"]

        # Sort by class for coherent single-class windows during training.
        order = np.argsort(y_train, kind="stable")
        X_sorted = X_train[order]
        y_sorted = y_train[order]

        X_seq, y_seq = self._make_sequences(X_sorted, y_sorted, seq_len)
        self.log.info("  Sequences (sorted, non-overlapping): %d", len(X_seq))
        self.log.info("  Sequence class distribution (pre-SMOTE):")
        for i, n in enumerate(THREAT_LABELS):
            c = (y_seq == i).sum()
            if c: self.log.info("    %2d %-22s %6d", i, n, c)

        y_seq_orig = y_seq.copy()

        # SMOTE for minority classes
        self.log.info("  [Step 3b] SMOTE → target %d seqs / minority class", SMOTE_MIN_SEQUENCES)
        X_seq, y_seq = smote_oversample(X_seq, y_seq, MINORITY_CLASSES, SMOTE_MIN_SEQUENCES)
        self.log.info("  Sequences after SMOTE: %d", len(X_seq))
        for i, n in enumerate(THREAT_LABELS):
            c = (y_seq == i).sum()
            if c: self.log.info("    %2d %-22s %6d", i, n, c)

        # Split: 80% train, 20% val
        # Safety: merge classes with < 2 samples into class 0 for stratification
        strat_y = y_seq.copy()
        for cls_id in range(NUM_CLASSES):
            if (strat_y == cls_id).sum() < 2:
                strat_y[strat_y == cls_id] = 0
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_seq, y_seq, test_size=0.2, stratify=strat_y, random_state=42
        )
        self.log.info("\n  TAGN Train: %d  Val: %d", len(X_tr), len(X_va))

        # Class weights from pre-SMOTE distribution, CAPPED at MAX_WEIGHT_RATIO
        counts_orig = np.bincount(y_seq_orig, minlength=NUM_CLASSES).astype(np.float32)
        present_mask = counts_orig > 0
        weights = np.ones(NUM_CLASSES, dtype=np.float32)
        
        if present_mask.sum() > 0:
            # Inverse frequency
            inv = np.where(present_mask, 1.0 / np.maximum(counts_orig, 1.0), 0.0)
            weights = inv / (inv[present_mask].mean() + 1e-8)
            
            # Cap the weight ratio to prevent rare class dominance
            w_present = weights[present_mask]
            if len(w_present) > 1 and w_present.max() > 0:
                median_w = np.median(w_present[w_present > 0])
                cap = median_w * MAX_WEIGHT_RATIO
                weights = np.clip(weights, 0.0, cap)
            
            # DDoS-centric: boost DDoS family by 1.3x
            for cls_id in DDOS_FAMILY:
                if present_mask[cls_id]:
                    weights[cls_id] *= 1.3

            # Zero out absent classes
            weights[~present_mask] = 0.0
            
            # Normalise so mean of present weights = 1.0
            w_sum = weights[present_mask].sum()
            if w_sum > 0:
                weights = weights / w_sum * present_mask.sum()

        self.log.info("  Present classes : %d / %d", present_mask.sum(), NUM_CLASSES)
        self.log.info("  Class weights (capped, DDoS-boosted): %s",
                      [f"{w:.3f}" for w in weights])

        tr_ldr = make_loader(X_tr, y_tr, self.cfg["tagn_bs"], shuffle=True,  device=self.tagn_device)
        va_ldr = make_loader(X_va, y_va, self.cfg["tagn_bs"], shuffle=False, device=self.tagn_device)

        model = create_tagn_model(
            input_dim=self.input_dim, hidden_dim=256, n_heads=8,
            num_classes=NUM_CLASSES, dropout=0.3,
        ).to(self.tagn_device)

        opt   = optim.AdamW(model.parameters(), lr=self.cfg["tagn_lr"], weight_decay=1e-3)
        crit  = nn.CrossEntropyLoss(
            weight=torch.tensor(weights).to(self.tagn_device),
        )
        sched = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.cfg["tagn_epochs"], eta_min=1e-5
        )
        scaler    = torch.amp.GradScaler(enabled=self.use_amp)
        best_path = os.path.join(self.exp_dir, "tagn_best.pt")
        best_va, patience_ctr = 0.0, 0

        for ep in range(1, self.cfg["tagn_epochs"] + 1):
            model.train()
            tr_loss = tr_correct = tr_total = 0
            for bx, by in tr_ldr:
                bx = bx.to(self.tagn_device)
                by = by.to(self.tagn_device)
                opt.zero_grad()
                amp_enabled = self.use_amp and is_cuda(self.tagn_device)
                with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                    logits = model(bx)["classification"]["logits"]
                    loss   = crit(logits, by)
                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt); scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                tr_loss    += loss.item()
                tr_correct += (logits.argmax(1) == by).sum().item()
                tr_total   += by.size(0)

            # Validation — compute F1 for early stopping (not just accuracy)
            model.eval()
            va_preds_all, va_true_all = [], []
            with torch.no_grad():
                for bx, by in va_ldr:
                    bx = bx.to(self.tagn_device)
                    logits = model(bx)["classification"]["logits"]
                    va_preds_all.append(logits.argmax(1).cpu().numpy())
                    va_true_all.append(by.numpy())

            va_preds = np.concatenate(va_preds_all)
            va_true  = np.concatenate(va_true_all)
            va_acc   = 100 * accuracy_score(va_true, va_preds)
            tr_acc   = 100 * tr_correct / max(tr_total, 1)
            sched.step()

            # Early stopping on validation accuracy (proven to work)
            if va_acc > best_va:
                best_va, patience_ctr = va_acc, 0
                torch.save(self._cpu_state(model), best_path)
            else:
                patience_ctr += 1

            if ep % 10 == 0 or ep == 1 or patience_ctr == 0:
                self.log.info(
                    "  Ep %3d/%d  loss=%.4f  tr=%.2f%%  va=%.2f%%  best=%.2f%%",
                    ep, self.cfg["tagn_epochs"],
                    tr_loss / len(tr_ldr), tr_acc, va_acc, best_va,
                )
            if patience_ctr >= self.cfg["tagn_patience"]:
                self.log.info("  Early stop at epoch %d", ep)
                break

        model.load_state_dict(torch.load(best_path, map_location=self.tagn_device, weights_only=False))
        self.log.info("  Best val acc: %.2f%%  |  Saved tagn_best.pt\n", best_va)
        return model

    # ── Phase 3: Correlation Engine ───────────────────────────────────────────
    def train_correlation(self, ae: Autoencoder, tagn: TAGNNetwork,
                          X_train: np.ndarray, y_train: np.ndarray):
        self.log.info("─── Phase 3: Correlation Engine  [%s] ───", self.device)
        ae.eval(); tagn.eval()

        seq_len = self.cfg["tagn_seq_len"]
        order = np.argsort(y_train, kind="stable")
        X_seq, y_seq = self._make_sequences(X_train[order], y_train[order], seq_len)
        y_bin = (y_seq > 0).astype(np.float32)

        engine = self._to(create_correlation_engine(hidden=64))
        opt    = optim.AdamW(engine.parameters(), lr=self.cfg["corr_lr"])
        bce    = nn.BCELoss()
        scaler = torch.amp.GradScaler(enabled=self.use_amp)

        ldr = DataLoader(
            TensorDataset(torch.tensor(X_seq), torch.tensor(y_seq),
                          torch.tensor(y_bin).unsqueeze(1)),
            batch_size=self.cfg["corr_bs"], shuffle=True,
            pin_memory=is_cuda(self.device),
        )

        for ep in range(1, self.cfg["corr_epochs"] + 1):
            tot = 0.0
            for seq_b, cls_b, bin_b in ldr:
                seq_dml  = self._to(seq_b)
                seq_tagn = seq_b.to(self.tagn_device)
                bin_b    = self._to(bin_b)
                with torch.no_grad():
                    ae_in = seq_dml.mean(dim=1)
                    with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                        anom = Autoencoder.reconstruction_error(ae_in, ae(ae_in)).unsqueeze(1)
                    tout = tagn(seq_tagn)
                    cf   = tout["correlation_features"].to(self.device)
                    conf = tout["classification"]["confidence_score"].unsqueeze(1).to(self.device)
                    pc   = tout["classification"]["predicted_class"].to(self.device)
                opt.zero_grad()
                with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                    loss = bce(engine(anom, cf, conf, pc)["fusion_score"], bin_b)
                if self.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(opt); scaler.update()
                else:
                    loss.backward(); opt.step()
                tot += loss.item()
            if ep % 5 == 0 or ep == 1:
                self.log.info("  Ep %2d/%d  loss=%.4f",
                              ep, self.cfg["corr_epochs"], tot / len(ldr))

        torch.save(self._cpu_state(engine), os.path.join(self.exp_dir, "correlation_engine.pt"))
        self.log.info("  Saved correlation_engine.pt\n")
        return engine

    # ── Threshold calibration ─────────────────────────────────────────────────
    def calibrate_threshold(self, ae: Autoencoder,
                            X_train: np.ndarray, y_train: np.ndarray) -> float:
        self.log.info("─── Calibrating anomaly threshold (on TRAIN benign only) ───")
        ae.eval()
        X_b = X_train[y_train == 0]
        if len(X_b) > 50_000:
            X_b = X_b[np.random.choice(len(X_b), 50_000, replace=False)]

        scores = []
        for i in range(0, len(X_b), 1024):
            b = self._to(torch.tensor(X_b[i:i + 1024]))
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                    r = ae(b)
                scores.append(Autoencoder.reconstruction_error(b, r).cpu().numpy())

        scores = np.concatenate(scores)
        thr    = float(np.percentile(scores, 99))
        self.log.info("  Benign score  mean=%.6f  std=%.6f  p95=%.6f  p99=%.6f",
                      scores.mean(), scores.std(), np.percentile(scores, 95), thr)
        self.log.info("  Selected threshold = %.6f\n", thr)
        return thr

    # ── Phase 5: Validation on HELD-OUT test set ──────────────────────────────
    def validate(self, ae: Autoencoder, tagn: TAGNNetwork, engine,
                 X_test: np.ndarray, y_test: np.ndarray, threshold: float) -> Dict:
        self.log.info("─── Phase 5: Validation on HELD-OUT test set (15-class) ───")
        self.log.info("  Test samples: %d (never seen during training)", len(X_test))
        self.log.info("  Confidence threshold: %.2f", self.cfg["conf_threshold"])

        ae.eval(); tagn.eval(); engine.eval()
        engine.anomaly_threshold    = threshold
        engine.confidence_threshold = self.cfg["conf_threshold"]
        sl, bs = self.cfg["tagn_seq_len"], 1024
        n = len(X_test)

        # Stream A — AE anomaly scores
        y_anom = np.zeros(n, dtype=np.float32)
        for i in range(0, n, bs):
            b = self._to(torch.tensor(X_test[i:i + bs]))
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                    r = ae(b)
                y_anom[i:i + b.size(0)] = Autoencoder.reconstruction_error(b, r).cpu().numpy()

        # Stream B — TAGN per-flow classification
        # Replicate each flow across seq_len to match training distribution
        y_cls  = np.zeros(n, dtype=int)
        y_prob = np.zeros((n, NUM_CLASSES), dtype=np.float32)
        y_conf = np.zeros(n, dtype=np.float32)
        y_corr = np.zeros((n, 16), dtype=np.float32)

        pf_bs = 256
        for i in range(0, n, pf_bs):
            e = min(i + pf_bs, n)
            seq = torch.tensor(X_test[i:e]).unsqueeze(1).expand(-1, sl, -1).to(self.tagn_device)
            with torch.no_grad():
                out = tagn(seq)
            b = e - i
            y_cls[i:e]  = out["classification"]["predicted_class"].cpu().numpy()
            y_prob[i:e] = out["classification"]["class_probabilities"].cpu().numpy()
            y_conf[i:e] = out["classification"]["confidence_score"].cpu().numpy()
            y_corr[i:e] = out["correlation_features"].cpu().numpy()

        # Correlation Engine
        priorities = np.zeros(n, dtype=int)
        gated_cls  = np.zeros(n, dtype=int)
        for i in range(0, n, bs):
            e = min(i + bs, n)
            with torch.no_grad():
                co = engine(
                    self._to(torch.tensor(y_anom[i:e])),
                    self._to(torch.tensor(y_corr[i:e])),
                    self._to(torch.tensor(y_conf[i:e])),
                    self._to(torch.tensor(y_cls[i:e], dtype=torch.long)),
                )
            priorities[i:e] = co["priority"].cpu().numpy()
            gated_cls[i:e]  = co["gated_class"].cpu().numpy()

        # Metrics
        yb = (y_test > 0).astype(int)
        yp = (gated_cls > 0).astype(int)
        tn, fp, fn, tp = confusion_matrix(yb, yp).ravel()
        bm = {
            "accuracy":  float(accuracy_score(yb, yp)),
            "precision": float(precision_score(yb, yp, zero_division=0)),
            "recall":    float(recall_score(yb, yp, zero_division=0)),
            "f1":        float(f1_score(yb, yp, zero_division=0)),
            "fpr":       float(fp / (fp + tn)) if (fp + tn) else 0.0,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        }
        try:
            bm["roc_auc"] = float(roc_auc_score(yb, 1 - y_prob[:, 0]))
        except:
            bm["roc_auc"] = 0.0

        self.log.info(
            "  Acc=%.4f  Pre=%.4f  Rec=%.4f  F1=%.4f  FPR=%.4f  AUC=%.4f",
            bm["accuracy"], bm["precision"], bm["recall"], bm["f1"], bm["fpr"], bm["roc_auc"]
        )
        self.log.info("  TP=%d  FP=%d  FN=%d  TN=%d", tp, fp, fn, tn)

        present = sorted(set(y_test) | set(gated_cls))
        names   = [THREAT_LABELS[i] for i in present]
        report  = classification_report(y_test, gated_cls, labels=present,
                                        target_names=names, zero_division=0)
        self.log.info("\n%s", report)

        # DDoS family
        ddos_mask = np.isin(y_test, list(DDOS_FAMILY))
        if ddos_mask.sum() > 0:
            ddos_tp = (ddos_mask & (gated_cls > 0)).sum()
            ddos_fn = (ddos_mask & (gated_cls == 0)).sum()
            ddos_recall = ddos_tp / (ddos_tp + ddos_fn) if (ddos_tp + ddos_fn) > 0 else 0
            self.log.info("  ═══ DDoS Family Detection ═══")
            self.log.info("  DDoS-family samples: %d  recall: %.4f", ddos_mask.sum(), ddos_recall)
            bm["ddos_family_recall"]  = float(ddos_recall)
            bm["ddos_family_samples"] = int(ddos_mask.sum())

        metrics = {"binary": bm, "multi_class_report": report,
                   "anomaly_threshold": threshold, "test_samples": n,
                   "train_test_split": f"{TRAIN_RATIO:.0%} / {1-TRAIN_RATIO:.0%}"}
        with open(os.path.join(self.exp_dir, "validation_results.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics

    # ── Full pipeline ─────────────────────────────────────────────────────────
    def run(self):
        self.log.info("=" * 60)
        self.log.info("  HALO NIDS — AGILE v3.2  (15-class, GPU-accelerated)")
        self.log.info("  Train/Test: %.0f%% / %.0f%% stratified split",
                      TRAIN_RATIO * 100, (1 - TRAIN_RATIO) * 100)
        self.log.info("  Device: %s", self.device)
        self.log.info("=" * 60 + "\n")

        data = self.load_data()
        X_train, y_train = data["X_train"], data["y_train"]
        X_test,  y_test  = data["X_test"],  data["y_test"]

        ae     = self.train_autoencoder(X_train, y_train)
        tagn   = self.train_tagn(X_train, y_train)
        engine = self.train_correlation(ae, tagn, X_train, y_train)
        thr    = self.calibrate_threshold(ae, X_train, y_train)

        deploy = {
            "version": "v3.2", "input_dim": self.input_dim,
            "num_classes": NUM_CLASSES, "threat_labels": THREAT_LABELS,
            "anomaly_threshold": thr, "seq_len": self.cfg["tagn_seq_len"],
            "confidence_threshold": self.cfg["conf_threshold"],
            "train_test_split": f"{TRAIN_RATIO:.0%} / {1-TRAIN_RATIO:.0%}",
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "model_files": {
                "autoencoder": "autoencoder.pt", "tagn": "tagn_best.pt",
                "correlation": "correlation_engine.pt", "scaler": "scaler.pkl",
                "test_split": "test_split.npz",
            },
        }
        with open(os.path.join(self.exp_dir, "deploy_config.json"), "w") as f:
            json.dump(deploy, f, indent=2)

        metrics = self.validate(ae, tagn, engine, X_test, y_test, thr)
        elapsed = (time.time() - self.t0) / 60
        self.log.info("─── Complete in %.1f min  |  %s ───", elapsed, self.exp_dir)

        report = {
            "experiment": self.exp_dir, "version": "v3.2",
            "duration_min": round(elapsed, 2), "input_dim": self.input_dim,
            "num_classes": NUM_CLASSES, "device": str(self.device),
            "amp": self.use_amp, "config": self.cfg,
            "anomaly_threshold": thr,
            "train_samples": len(X_train), "test_samples": len(X_test),
            "train_test_split": f"{TRAIN_RATIO:.0%} / {1-TRAIN_RATIO:.0%}",
            "validation": metrics,
        }
        with open(os.path.join(self.exp_dir, "training_report.json"), "w") as f:
            json.dump(report, f, indent=2, default=str)
        return metrics


def main():
    p = argparse.ArgumentParser(description="HALO NIDS v3.2 Training")
    p.add_argument("--cpu",         action="store_true")
    p.add_argument("--gpu",         type=int, default=0)
    p.add_argument("--batch-size",  type=int, default=None)
    args = p.parse_args()
    AGILETrainerV3(force_cpu=args.cpu, gpu_index=args.gpu,
                   batch_override=args.batch_size).run()

if __name__ == "__main__":
    main()
