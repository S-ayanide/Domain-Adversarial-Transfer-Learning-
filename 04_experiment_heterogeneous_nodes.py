"""
Step 4: Experiment — Heterogeneous Nodes  (reproduces paper Figure 4)

Evaluates the DATL model (with domain adversarial mechanism) on different
node types in the target domain:
  - CPU-intensive nodes
  - Memory-intensive nodes
  - I/O-bound nodes
  - Mixed-load nodes

Shows how the domain adversarial strategy maintains performance across
heterogeneous hardware environments.

Output:
  results/tables/heterogeneous_nodes.json
  results/figures/figure4_heterogeneous_nodes.png
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

from models import DATL, DANN
from trainer import train_datl, train_adversarial, evaluate, to_tensor

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
        logging.FileHandler(LOG_DIR / "04_heterogeneous_nodes.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

HIDDEN_DIM = 128
DROPOUT    = 0.3
EPOCHS     = 150
LR         = 1e-3
BATCH_SIZE = 256

NODE_TYPES = ["cpu_heavy", "mem_heavy", "io_heavy", "mixed"]
NODE_LABELS = {
    "cpu_heavy": "CPU-Intensive",
    "mem_heavy": "Memory-Intensive",
    "io_heavy":  "I/O-Bound",
    "mixed":     "Mixed-Load",
}

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

    scaler = StandardScaler()
    X_src  = scaler.fit_transform(X_src).astype(np.float32)

    tgt_feature = tgt[feature_cols].values.astype(np.float32)
    tgt_feature = scaler.transform(tgt_feature).astype(np.float32)

    return X_src, y_src, tgt, tgt_feature, int(meta["n_classes"]), feature_cols


def run_heterogeneous_nodes():
    logger.info("=" * 65)
    logger.info("  Experiment: Heterogeneous Node Types (Figure 4)")
    logger.info("=" * 65)

    X_src, y_src, tgt_df, tgt_feature, n_classes, _ = load_data()
    input_dim = X_src.shape[1]

    # Check if node_type column exists
    if "node_type" not in tgt_df.columns:
        logger.warning("  node_type column missing — assigning node types from resource patterns")
        def assign_node_type(row):
            if row.get("cpu_util_percent", 0) > 70:   return "cpu_heavy"
            if row.get("mem_util_percent", 0) > 70:   return "mem_heavy"
            if row.get("disk_io_percent",  0) > 60:   return "io_heavy"
            return "mixed"
        feat_df = pd.read_parquet(PROC_DIR / "target_domain.parquet")
        if "cpu_util_percent" in feat_df.columns:
            tgt_df["node_type"] = feat_df.apply(assign_node_type, axis=1)
        else:
            # Fallback: random assignment
            rng = np.random.default_rng(42)
            tgt_df = tgt_df.copy()
            tgt_df["node_type"] = rng.choice(NODE_TYPES, size=len(tgt_df))

    tgt_labeled = tgt_df["labeled"].values.astype(bool)

    # ── Train models on the full target domain ────────────────────────────────
    logger.info("\nTraining models on full target domain ...")

    # DATL
    datl = DATL(input_dim, n_classes, HIDDEN_DIM, DROPOUT,
                lambda1=0.1, lambda2=0.1, pseudo_threshold=0.85)
    train_datl(datl, X_src, y_src, tgt_feature,
               tgt_df["label"].values.astype(np.int64), tgt_labeled,
               n_classes, EPOCHS, LR, BATCH_SIZE)

    # DANN (for comparison)
    dann = DANN(input_dim, n_classes, HIDDEN_DIM, DROPOUT)
    train_adversarial(dann, "DANN", X_src, y_src, tgt_feature,
                      tgt_df["label"].values.astype(np.int64), tgt_labeled,
                      n_classes, EPOCHS, LR, BATCH_SIZE)

    # ── Evaluate per node type ────────────────────────────────────────────────
    records = {"DATL (Ours)": [], "DANN": []}

    for node_type in NODE_TYPES:
        mask  = (tgt_df["node_type"] == node_type).values
        if mask.sum() < 10:
            logger.warning(f"  Skipping {node_type}: only {mask.sum()} samples")
            continue

        X_node = tgt_feature[mask]
        y_node = tgt_df["label"].values[mask].astype(np.int64)

        if len(np.unique(y_node)) < 2:
            logger.warning(f"  Skipping {node_type}: only 1 class present")
            continue

        datl_m = evaluate(datl, X_node, y_node, n_classes)
        dann_m = evaluate(dann, X_node, y_node, n_classes)

        records["DATL (Ours)"].append({
            "node_type": node_type,
            **datl_m,
        })
        records["DANN"].append({
            "node_type": node_type,
            **dann_m,
        })

        logger.info(f"  {NODE_LABELS[node_type]:<22}")
        logger.info(f"    DATL  acc={datl_m['accuracy']:.4f} f1={datl_m['f1']:.4f} auc={datl_m['auc']:.4f}")
        logger.info(f"    DANN  acc={dann_m['accuracy']:.4f} f1={dann_m['f1']:.4f} auc={dann_m['auc']:.4f}")

    with open(TAB_DIR / "heterogeneous_nodes.json", "w") as f:
        json.dump(records, f, indent=2)

    # ── Plot (grouped bar chart, mirroring Figure 4) ──────────────────────────
    node_labels = [NODE_LABELS[r["node_type"]] for r in records["DATL (Ours)"]]
    n_nodes     = len(node_labels)
    if n_nodes == 0:
        logger.warning("No node-type data to plot — skipping figure.")
        return records

    metrics = ["accuracy", "f1", "auc"]
    titles  = ["Accuracy", "F1-Score", "AUC"]
    colors_datl = ["#F44336", "#E91E63", "#FF5722"]
    colors_dann = ["#9E9E9E", "#757575", "#616161"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    bar_width = 0.35
    x         = np.arange(n_nodes)

    for ax, metric, title, c_d, c_n in zip(axes, metrics, titles,
                                            colors_datl, colors_dann):
        datl_vals = [r[metric] for r in records["DATL (Ours)"]]
        dann_vals = [r[metric] for r in records["DANN"]]

        ax.bar(x - bar_width/2, datl_vals, bar_width, label="DATL (Ours)",
               color=c_d, alpha=0.85, edgecolor="white")
        ax.bar(x + bar_width/2, dann_vals, bar_width, label="DANN",
               color=c_n, alpha=0.85, edgecolor="white")

        ax.set_xlabel("Node Type", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f"{title} across Node Types", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(node_labels, rotation=15, ha="right", fontsize=9)
        ax.set_ylim(0.50, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Domain Adversarial Model on Heterogeneous Nodes",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "figure4_heterogeneous_nodes.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"\nFigure saved to {FIG_DIR}/figure4_heterogeneous_nodes.png")
    logger.info(f"Data saved to {TAB_DIR}/heterogeneous_nodes.json")
    return records


if __name__ == "__main__":
    run_heterogeneous_nodes()
