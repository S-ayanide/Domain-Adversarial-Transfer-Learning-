"""
Step 2: Experiment — Label Scarcity  (reproduces paper Figure 2)

Varies the proportion of labeled target data from 10% to 100%.
Trains DATL at each proportion and records accuracy, F1, AUC.

Output:
  results/tables/label_scarcity.json
  results/figures/figure2_label_scarcity.png
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from models import DATL
from trainer import train_datl

PROC_DIR = BASE_DIR / "data" / "processed"
RES_DIR  = BASE_DIR / "results"
FIG_DIR  = RES_DIR / "figures"
TAB_DIR  = RES_DIR / "tables"
LOG_DIR  = BASE_DIR / "logs"
for d in [FIG_DIR, TAB_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "02_label_scarcity.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ─── Hyper-parameters ─────────────────────────────────────────────────────────
HIDDEN_DIM   = 128
DROPOUT      = 0.3
EPOCHS       = 80
LR           = 1e-3
BATCH_SIZE   = 256
LABEL_RATIOS = [0.10, 0.20, 0.30, 0.50, 0.70, 1.00]   # Figure 2 x-axis

FEATURE_COLS = [
    "cpu_util_percent", "mem_util_percent", "mem_gps",
    "net_in", "net_out", "disk_io_percent",
]


def load_data():
    with open(PROC_DIR / "meta.json") as f:
        meta = json.load(f)
    src = pd.read_parquet(PROC_DIR / "source_domain.parquet")
    tgt = pd.read_parquet(PROC_DIR / "target_domain.parquet")

    feature_cols = [c for c in FEATURE_COLS if c in src.columns]
    if not feature_cols:
        feature_cols = [c for c in src.columns
                        if c not in {"label", "domain", "labeled", "machine_id",
                                     "time_stamp", "node_type", "failure_domain_1",
                                     "failure_domain_2", "mkpi", "mem_size",
                                     "cpu_num", "status"}]

    X_src = src[feature_cols].values.astype(np.float32)
    y_src = src["label"].values.astype(np.int64)
    X_tgt = tgt[feature_cols].values.astype(np.float32)
    y_tgt = tgt["label"].values.astype(np.int64)

    scaler = StandardScaler()
    X_src  = scaler.fit_transform(X_src).astype(np.float32)
    X_tgt  = scaler.transform(X_tgt).astype(np.float32)

    return X_src, y_src, X_tgt, y_tgt, int(meta["n_classes"])


def run_label_scarcity():
    logger.info("=" * 65)
    logger.info("  Experiment: Label Scarcity (Figure 2)")
    logger.info("=" * 65)

    X_src, y_src, X_tgt, y_tgt, n_classes = load_data()
    input_dim = X_src.shape[1]
    rng       = np.random.default_rng(42)

    records = []

    for ratio in LABEL_RATIOS:
        # Randomly select `ratio` fraction of target as labeled
        n_labeled = max(int(len(X_tgt) * ratio), n_classes * 2)
        indices   = rng.choice(len(X_tgt), n_labeled, replace=False)
        labeled   = np.zeros(len(X_tgt), dtype=bool)
        labeled[indices] = True

        logger.info(f"\nRatio={ratio:.0%}  labeled={labeled.sum()} / {len(labeled)}")

        model = DATL(input_dim, n_classes, HIDDEN_DIM, DROPOUT,
                     lambda1=0.1, lambda2=0.1, pseudo_threshold=0.85)
        m = train_datl(model, X_src, y_src, X_tgt, y_tgt, labeled,
                       n_classes, EPOCHS, LR, BATCH_SIZE)

        records.append({
            "labeled_ratio": ratio,
            "accuracy": m["accuracy"],
            "f1":       m["f1"],
            "auc":      m["auc"],
        })
        logger.info(f"  acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  auc={m['auc']:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(TAB_DIR / "label_scarcity.json", "w") as f:
        json.dump(records, f, indent=2)

    # ── Paper reference values (from Figure 2 description) ────────────────────
    paper_ref = {
        0.10: {"accuracy": 0.81, "f1": 0.76, "auc": 0.86},
        0.30: {"accuracy": 0.84, "f1": 0.80, "auc": 0.88},
        0.50: {"accuracy": 0.86, "f1": 0.82, "auc": 0.89},
        0.70: {"accuracy": 0.88, "f1": 0.84, "auc": 0.90},
        1.00: {"accuracy": 0.896,"f1": 0.854,"auc": 0.913},
    }

    # ── Plot ──────────────────────────────────────────────────────────────────
    ratios = [r["labeled_ratio"] for r in records]
    accs   = [r["accuracy"] for r in records]
    f1s    = [r["f1"]       for r in records]
    aucs   = [r["auc"]      for r in records]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ratios, accs, "o-", label="Accuracy (ours)",  color="#2196F3", linewidth=2)
    ax.plot(ratios, f1s,  "s-", label="F1-Score (ours)",  color="#4CAF50", linewidth=2)
    ax.plot(ratios, aucs, "^-", label="AUC (ours)",       color="#FF5722", linewidth=2)

    # Paper reference (dashed)
    ref_x    = sorted(paper_ref.keys())
    ref_acc  = [paper_ref[r]["accuracy"] for r in ref_x]
    ref_f1   = [paper_ref[r]["f1"]       for r in ref_x]
    ref_auc  = [paper_ref[r]["auc"]      for r in ref_x]
    ax.plot(ref_x, ref_acc, "o--", color="#2196F3", alpha=0.4, label="Accuracy (paper)")
    ax.plot(ref_x, ref_f1,  "s--", color="#4CAF50", alpha=0.4, label="F1-Score (paper)")
    ax.plot(ref_x, ref_auc, "^--", color="#FF5722", alpha=0.4, label="AUC (paper)")

    ax.set_xlabel("Proportion of labeled target data", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("DATL Robustness under Label Scarcity", fontsize=13, fontweight="bold")
    ax.set_xlim(0.05, 1.05)
    ax.set_ylim(0.60, 1.00)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "figure2_label_scarcity.png", dpi=150)
    plt.close()

    logger.info(f"\nFigure saved to {FIG_DIR}/figure2_label_scarcity.png")
    logger.info(f"Data saved to {TAB_DIR}/label_scarcity.json")
    return records


if __name__ == "__main__":
    run_label_scarcity()
