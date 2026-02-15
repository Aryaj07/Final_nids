"""
HALO NIDS -- AGILE v2.1  Paper-Quality Visualisations
======================================================
Generates publication-ready figures from test results.

Run AFTER test_v2.py:
    python generate_plots.py
    python generate_plots.py --results test_results_v2/test_YYYYMMDD_HHMMSS/all_metrics.json
"""

import os, sys, json, argparse, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")          # no GUI needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ---------- style ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

COLOURS = {
    "DDoS":          "#e74c3c",
    "PortScan":      "#e67e22",
    "WebAttacks":    "#f1c40f",
    "Infiltration":  "#9b59b6",
    "Monday_Benign": "#3498db",
    "Tuesday":       "#1abc9c",
    "Wednesday":     "#2ecc71",
    "Friday_Morning":"#34495e",
}

ATTACK_DS  = ["DDoS", "PortScan", "WebAttacks", "Infiltration"]
BENIGN_DS  = ["Monday_Benign", "Tuesday", "Wednesday", "Friday_Morning"]
ALL_DS     = ATTACK_DS + BENIGN_DS

PRIORITY_NAMES = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
PRIO_COLOURS   = ["#c0392b", "#e67e22", "#f1c40f", "#27ae60"]


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ======================================================================
# Figure 1 — Binary classification metrics bar chart
# ======================================================================
def fig_binary_metrics(results: dict, out_dir: str):
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels  = ["Accuracy", "Precision", "Recall", "F1-Score"]

    datasets = [d for d in ALL_DS if d in results]
    n = len(datasets)
    x = np.arange(n)
    w = 0.18

    fig, ax = plt.subplots(figsize=(14, 5.5))
    for i, (m, lab) in enumerate(zip(metrics, labels)):
        vals = [results[d]["binary"][m] for d in datasets]
        bars = ax.bar(x + i * w, vals, w, label=lab, zorder=3)
        for bar, v in zip(bars, vals):
            if v > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.set_title("Binary Classification Metrics per Dataset")
    ax.legend(loc="upper right", ncol=4)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_binary_metrics.png"))
    plt.close(fig)
    print("  [OK] fig1_binary_metrics.png")


# ======================================================================
# Figure 2 — FPR comparison across datasets
# ======================================================================
def fig_fpr(results: dict, out_dir: str):
    datasets = [d for d in ALL_DS if d in results]
    fprs = [results[d]["binary"]["fpr"] * 100 for d in datasets]
    colors = [COLOURS.get(d, "#999") for d in datasets]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(datasets, fprs, color=colors, edgecolor="white", zorder=3)
    for bar, v in zip(bars, fprs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{v:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("False Positive Rate (%)")
    ax.set_title("False Positive Rate per Dataset")
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.axhline(y=5, color="red", linestyle="--", linewidth=0.8, label="5% target")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_fpr_comparison.png"))
    plt.close(fig)
    print("  [OK] fig2_fpr_comparison.png")


# ======================================================================
# Figure 3 — Confusion matrices (2x4 grid)
# ======================================================================
def fig_confusion_matrices(results: dict, out_dir: str):
    datasets = [d for d in ALL_DS if d in results]
    n = len(datasets)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4.2 * rows))
    axes = np.array(axes).flatten()

    for idx, ds in enumerate(datasets):
        bm = results[ds]["binary"]
        cm = np.array([[bm["tn"], bm["fp"]],
                       [bm["fn"], bm["tp"]]])
        ax = axes[idx]
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                colour = "white" if val > cm.max() * 0.5 else "black"
                ax.text(j, i, f"{val:,}", ha="center", va="center",
                        fontsize=9, color=colour, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Benign", "Attack"])
        ax.set_yticklabels(["Benign", "Attack"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        f1 = bm["f1"]
        ax.set_title(f"{ds}\nF1={f1:.3f}", fontsize=10)

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Confusion Matrices per Dataset", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_confusion_matrices.png"))
    plt.close(fig)
    print("  [OK] fig3_confusion_matrices.png")


# ======================================================================
# Figure 4 — Priority distribution stacked bar chart
# ======================================================================
def fig_priority_distribution(results: dict, out_dir: str):
    datasets = [d for d in ALL_DS if d in results]
    n = len(datasets)

    data = {p: [] for p in PRIORITY_NAMES}
    for ds in datasets:
        pd_dist = results[ds].get("priority_distribution", {})
        total = sum(pd_dist.values()) if pd_dist else 1
        for p in PRIORITY_NAMES:
            data[p].append(pd_dist.get(p, 0) / total * 100)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    bottom = np.zeros(n)
    x = np.arange(n)
    for i, p in enumerate(PRIORITY_NAMES):
        vals = data[p]
        ax.bar(x, vals, bottom=bottom, color=PRIO_COLOURS[i], label=p,
               edgecolor="white", linewidth=0.5, zorder=3)
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel("Percentage of Flows (%)")
    ax.set_title("Alert Priority Distribution per Dataset")
    ax.legend(loc="upper right", ncol=4)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig4_priority_distribution.png"))
    plt.close(fig)
    print("  [OK] fig4_priority_distribution.png")


# ======================================================================
# Figure 5 — Recall vs FPR scatter (ROC-style operating points)
# ======================================================================
def fig_recall_vs_fpr(results: dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 6))

    for ds in ALL_DS:
        if ds not in results:
            continue
        bm = results[ds]["binary"]
        rec = bm["recall"]
        fpr = bm["fpr"]
        c = COLOURS.get(ds, "#999")
        ax.scatter(fpr * 100, rec * 100, s=140, c=c, edgecolors="black",
                   linewidths=0.8, zorder=5)
        # offset label to avoid overlap
        ox, oy = 0.4, -2.5
        if ds == "DDoS":
            oy = 2
        ax.annotate(ds, (fpr * 100, rec * 100), fontsize=8,
                    textcoords="offset points", xytext=(ox + 5, oy),
                    fontweight="bold")

    ax.set_xlabel("False Positive Rate (%)")
    ax.set_ylabel("Recall / Detection Rate (%)")
    ax.set_title("Detection Rate vs False Positive Rate")
    ax.set_xlim(-0.5, max(12, ax.get_xlim()[1]))
    ax.set_ylim(-5, 105)
    ax.axvline(x=5, color="red", linestyle="--", linewidth=0.8, alpha=0.6, label="5% FPR target")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig5_recall_vs_fpr.png"))
    plt.close(fig)
    print("  [OK] fig5_recall_vs_fpr.png")


# ======================================================================
# Figure 6 — Confidence gating effectiveness
# ======================================================================
def fig_confidence_gating(results: dict, out_dir: str):
    datasets = [d for d in ALL_DS if d in results and "confidence_gating" in results[d]]
    if not datasets:
        print("  [SKIP] fig6 -- no confidence gating data")
        return

    raw_vals    = []
    gated_vals  = []
    filtered    = []
    for ds in datasets:
        cg = results[ds]["confidence_gating"]
        raw_vals.append(cg["raw_attack_predictions"])
        gated_vals.append(cg["after_gating"])
        filtered.append(cg["filtered_out"])

    x = np.arange(len(datasets))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w / 2, raw_vals, w, label="Raw Attack Predictions", color="#e74c3c",
           edgecolor="white", zorder=3)
    ax.bar(x + w / 2, gated_vals, w, label="After Confidence Gate", color="#27ae60",
           edgecolor="white", zorder=3)

    for i, f in enumerate(filtered):
        if f > 0:
            ax.annotate(f"  -{f}", (x[i] + w / 2, gated_vals[i]),
                        fontsize=8, color="#e74c3c", fontweight="bold",
                        ha="left", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel("Number of Flows")
    ax.set_title("Confidence Gating: Raw vs Filtered Attack Predictions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig6_confidence_gating.png"))
    plt.close(fig)
    print("  [OK] fig6_confidence_gating.png")


# ======================================================================
# Figure 7 — ROC-AUC bar chart
# ======================================================================
def fig_roc_auc(results: dict, out_dir: str):
    datasets = [d for d in ALL_DS if d in results]
    aucs = [results[d]["binary"].get("roc_auc", 0) for d in datasets]
    colors = [COLOURS.get(d, "#999") for d in datasets]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(datasets, aucs, color=colors, edgecolor="white", zorder=3)
    for bar, v in zip(bars, aucs):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("ROC-AUC")
    ax.set_title("ROC-AUC Score per Dataset")
    ax.set_ylim(0, 1.12)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.axhline(y=0.5, color="grey", linestyle=":", linewidth=0.8, label="Random baseline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig7_roc_auc.png"))
    plt.close(fig)
    print("  [OK] fig7_roc_auc.png")


# ======================================================================
# Figure 8 — Detection summary radar / spider chart
# ======================================================================
def fig_radar(results: dict, out_dir: str):
    attack_ds = [d for d in ATTACK_DS if d in results and results[d]["binary"]["recall"] > 0]
    if len(attack_ds) < 2:
        print("  [SKIP] fig8 -- need >= 2 attack datasets with recall > 0")
        return

    metrics = ["accuracy", "precision", "recall", "f1"]
    labels  = ["Accuracy", "Precision", "Recall", "F1"]
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for ds in attack_ds:
        vals = [results[ds]["binary"][m] for m in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=1.8, label=ds, color=COLOURS.get(ds, "#999"))
        ax.fill(angles, vals, alpha=0.1, color=COLOURS.get(ds, "#999"))

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Attack Detection Performance (Radar)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig8_radar_chart.png"))
    plt.close(fig)
    print("  [OK] fig8_radar_chart.png")


# ======================================================================
# Figure 9 — Algorithm pipeline diagram
# ======================================================================
def fig_pipeline(out_dir: str):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (0.5,  1.5, "Phase 1\nFeature\nExtraction",  "#3498db"),
        (3.0,  2.5, "Stream A\nAutoencoder\n(Anomaly S)", "#e74c3c"),
        (3.0,  0.5, "Stream B\nTAGN\n(Class C)",      "#2ecc71"),
        (6.0,  1.5, "Phase 3\nCorrelation\n& Validation", "#9b59b6"),
        (9.0,  1.5, "Phase 4\nLLM Alert\nGeneration",  "#e67e22"),
        (11.5, 1.5, "Phase 5\nOutput\nDashboard",      "#1abc9c"),
    ]

    for x, y, text, col in boxes:
        rect = mpatches.FancyBboxPatch(
            (x, y - 0.6), 2.0, 1.2,
            boxstyle="round,pad=0.15", facecolor=col, alpha=0.85,
            edgecolor="black", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + 1.0, y, text, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="white")

    # arrows
    arrow_kw = dict(arrowstyle="->,head_width=0.3,head_length=0.2",
                    color="black", linewidth=1.5)
    ax.annotate("", xy=(3.0, 2.5), xytext=(2.5, 1.8), arrowprops=arrow_kw)
    ax.annotate("", xy=(3.0, 0.8), xytext=(2.5, 1.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(6.0, 1.8), xytext=(5.0, 2.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(6.0, 1.5), xytext=(5.0, 0.8), arrowprops=arrow_kw)
    ax.annotate("", xy=(9.0, 1.5), xytext=(8.0, 1.5), arrowprops=arrow_kw)
    ax.annotate("", xy=(11.5, 1.5), xytext=(11.0, 1.5), arrowprops=arrow_kw)

    ax.set_title("HALO NIDS -- AGILE Algorithm Pipeline", fontsize=14,
                 fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig9_pipeline_diagram.png"))
    plt.close(fig)
    print("  [OK] fig9_pipeline_diagram.png")


# ======================================================================
# Figure 10 — Summary table as image
# ======================================================================
def fig_summary_table(results: dict, out_dir: str):
    datasets = [d for d in ALL_DS if d in results]
    headers = ["Dataset", "Samples", "Accuracy", "Precision", "Recall", "F1", "FPR", "ROC-AUC"]

    rows = []
    for ds in datasets:
        bm = results[ds]["binary"]
        n_samples = results[ds]["n_samples"]
        rows.append([
            ds,
            f"{n_samples:,}",
            f"{bm['accuracy']:.4f}",
            f"{bm['precision']:.4f}",
            f"{bm['recall']:.4f}",
            f"{bm['f1']:.4f}",
            f"{bm['fpr']:.4f}",
            f"{bm.get('roc_auc', 0):.4f}",
        ])

    fig, ax = plt.subplots(figsize=(14, 0.5 + 0.45 * len(rows)))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    # header style
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # row colouring
    for i, ds in enumerate(datasets):
        is_attack = ds in ATTACK_DS
        bg = "#fde8e8" if is_attack else "#e8f6fd"
        for j in range(len(headers)):
            table[i + 1, j].set_facecolor(bg)

    ax.set_title("AGILE v2.1 -- Comprehensive Test Results Summary",
                 fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig10_summary_table.png"))
    plt.close(fig)
    print("  [OK] fig10_summary_table.png")


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--results", type=str, default=None,
                        help="Path to all_metrics.json")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory for figures")
    args = parser.parse_args()

    # find results
    if args.results:
        results_path = args.results
    else:
        base = "test_results_v2"
        if not os.path.exists(base):
            print("ERROR: test_results_v2/ not found. Run test_v2.py first.")
            sys.exit(1)
        dirs = sorted(glob.glob(os.path.join(base, "test_*")))
        if not dirs:
            print("ERROR: No test results found.")
            sys.exit(1)
        results_path = os.path.join(dirs[-1], "all_metrics.json")

    if not os.path.exists(results_path):
        print(f"ERROR: {results_path} not found.")
        sys.exit(1)

    results = load_results(results_path)
    print(f"Loaded results from: {results_path}")
    print(f"Datasets: {list(results.keys())}\n")

    # output dir
    if args.out:
        out_dir = args.out
    else:
        out_dir = os.path.join(os.path.dirname(results_path), "figures")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving figures to: {out_dir}\n")

    # generate all figures
    fig_binary_metrics(results, out_dir)
    fig_fpr(results, out_dir)
    fig_confusion_matrices(results, out_dir)
    fig_priority_distribution(results, out_dir)
    fig_recall_vs_fpr(results, out_dir)
    fig_confidence_gating(results, out_dir)
    fig_roc_auc(results, out_dir)
    fig_radar(results, out_dir)
    fig_pipeline(out_dir)
    fig_summary_table(results, out_dir)

    print(f"\nAll figures saved to: {out_dir}")
    print("Figures generated:")
    print("  fig1  - Binary classification metrics (bar chart)")
    print("  fig2  - False positive rate comparison")
    print("  fig3  - Confusion matrices (2x4 grid)")
    print("  fig4  - Alert priority distribution (stacked bars)")
    print("  fig5  - Recall vs FPR scatter (operating points)")
    print("  fig6  - Confidence gating effectiveness")
    print("  fig7  - ROC-AUC scores")
    print("  fig8  - Radar chart (attack detection)")
    print("  fig9  - Algorithm pipeline diagram")
    print("  fig10 - Summary results table")


if __name__ == "__main__":
    main()
