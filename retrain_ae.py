"""
Retrain ONLY the autoencoder (Stream A) with improved v3 architecture.
Keeps the existing TAGN and Correlation Engine from the specified experiment.

Usage:
    python retrain_ae.py --experiment experiments_v3/agile_v3_20260227_140503
    python retrain_ae.py --experiment experiments_v3/agile_v3_20260227_140503 --epochs 30
"""

import os, sys, json, time, argparse, logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib

# Use the improved v3 autoencoder
from models_v3.autoencoder_v3 import Autoencoder

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("RetrainAE")

# Same CSV list and helpers as train_v3.py
import pandas as pd

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

from train_v3 import read_csv, clean_numeric, map_labels, TRAIN_RATIO
from sklearn.model_selection import train_test_split
from models_v3.tagn_network import NUM_CLASSES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", type=str, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=512)
    args = p.parse_args()

    exp_dir = args.experiment
    log.info("Experiment: %s", exp_dir)

    # Load config
    with open(os.path.join(exp_dir, "deploy_config.json")) as f:
        cfg = json.load(f)
    input_dim = cfg["input_dim"]

    # Load the existing scaler
    scaler = joblib.load(os.path.join(exp_dir, "scaler.pkl"))

    # Reload and split data exactly as train_v3.py did
    log.info("Loading data...")
    frames = []
    for f in ALL_CSV_FILES:
        if os.path.exists(f):
            frames.append(read_csv(f))
    combined = pd.concat(frames, ignore_index=True)
    labels = map_labels(combined["Label"])
    numeric = clean_numeric(combined.drop(columns=["Label"], errors="ignore"))
    labels = labels[numeric.index]
    numeric = numeric.reset_index(drop=True)
    X_all = numeric.values.astype(np.float32)

    strat_labels = labels.copy()
    for cls_id in range(NUM_CLASSES):
        if (strat_labels == cls_id).sum() < 2:
            strat_labels[strat_labels == cls_id] = 0

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, labels, test_size=(1 - TRAIN_RATIO),
        stratify=strat_labels, random_state=42,
    )
    log.info("Train: %d  Test: %d  (same split as original)", len(X_train), len(X_test))

    # Scale using the SAME scaler
    X_train_scaled = scaler.transform(X_train).astype(np.float32)
    X_benign = X_train_scaled[y_train == 0]
    log.info("Benign training samples: %d", len(X_benign))

    # Detect device
    try:
        import torch_directml
        if torch_directml.is_available():
            device = torch_directml.device()
            log.info("Device: DirectML")
        else:
            device = torch.device("cpu")
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Device: %s", device)

    # Train improved autoencoder
    ae = Autoencoder(input_dim, latent_dim=48).to(device)
    opt = optim.Adam(ae.parameters(), lr=args.lr, weight_decay=1e-5)
    crit = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(torch.tensor(X_benign)),
        batch_size=args.batch_size, shuffle=True,
    )

    log.info("Training improved AE (latent=48, skip connections) for %d epochs...", args.epochs)
    ae.train()
    for ep in range(1, args.epochs + 1):
        ep_loss = 0.0
        for (bx,) in loader:
            bx = bx.to(device)
            opt.zero_grad()
            loss = crit(ae(bx), bx)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        if ep % 5 == 0 or ep == 1:
            log.info("  Epoch %2d/%d  loss=%.6f", ep, args.epochs, ep_loss / len(loader))

    # Save
    state = {k: v.cpu() for k, v in ae.state_dict().items()}
    ae_path = os.path.join(exp_dir, "autoencoder_v3.pt")
    torch.save(state, ae_path)
    log.info("Saved -> %s", ae_path)

    # Calibrate threshold on benign training data
    ae.eval()
    scores = []
    for i in range(0, len(X_benign), 1024):
        b = torch.tensor(X_benign[i:i+1024]).to(device)
        with torch.no_grad():
            r = ae(b)
            scores.append(Autoencoder.reconstruction_error(b, r).cpu().numpy())
    scores = np.concatenate(scores)
    p95 = float(np.percentile(scores, 95))
    p99 = float(np.percentile(scores, 99))
    log.info("Benign AE scores: mean=%.4f  p95=%.4f  p99=%.4f", scores.mean(), p95, p99)

    # Test anomaly score distribution
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    ae_cpu = ae.cpu()
    ae_cpu.eval()

    test_scores = []
    for i in range(0, len(X_test_scaled), 1024):
        b = torch.tensor(X_test_scaled[i:i+1024])
        with torch.no_grad():
            test_scores.append(Autoencoder.reconstruction_error(b, ae_cpu(b)).numpy())
    test_scores = np.concatenate(test_scores)

    benign_scores = test_scores[y_test == 0]
    attack_scores = test_scores[y_test > 0]

    log.info("\nTest set anomaly score analysis:")
    log.info("  Benign  median=%.4f  p95=%.4f  p99=%.4f",
             np.median(benign_scores), np.percentile(benign_scores, 95),
             np.percentile(benign_scores, 99))
    log.info("  Attack  median=%.4f  p25=%.4f  p50=%.4f",
             np.median(attack_scores), np.percentile(attack_scores, 25),
             np.percentile(attack_scores, 50))

    ATTACK_FAMILIES = {
        "DDoS": [2,3,4,5,6], "Recon": [10], "Brute-Force": [7,11],
        "Web Attack": [12,13,14], "Botnet": [1],
    }
    for fam, cls_ids in ATTACK_FAMILIES.items():
        mask = np.isin(y_test, cls_ids)
        if mask.sum() > 0:
            fs = test_scores[mask]
            above_p99 = (fs > p99).sum()
            above_p95 = (fs > p95).sum()
            log.info("  %-15s  n=%6d  median=%.4f  above_p99=%5d (%.1f%%)  above_p95=%5d (%.1f%%)",
                     fam, mask.sum(), np.median(fs),
                     above_p99, 100*above_p99/mask.sum(),
                     above_p95, 100*above_p95/mask.sum())

    # Threshold sweep
    log.info("\nThreshold sweep:")
    for pct in [99, 97, 95, 93, 90, 85, 80]:
        thr = np.percentile(benign_scores, pct)
        fpr = (benign_scores > thr).mean()
        rec = (attack_scores > thr).mean()
        log.info("  p%02d=%.4f  FPR=%.4f  recall=%.4f", pct, thr, fpr, rec)

    log.info("\nDone! To test with the improved AE, update test_v3.py to load autoencoder_v3.pt")
    log.info("and use 'from models_v3.autoencoder_v3 import Autoencoder'")


if __name__ == "__main__":
    main()
