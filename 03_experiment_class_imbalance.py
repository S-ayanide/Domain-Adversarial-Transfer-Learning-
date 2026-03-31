"""
Step 3: Experiment — Class Imbalance  (reproduces paper Figure 3)

Gradually increases the imbalance ratio (IR) of fault classes in the
source domain, then evaluates all models on the target domain.
IR = n_majority / n_minority.

Output:
  results/tables/class_imbalance.json
  results/figures/figure3_class_imbalance.png
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

from models import DATL, DANN, CDAN, FixBi, ToAlign
from trainer import train_datl, train_adversarial, train_fixbi

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
        logging.FileHandler(LOG_DIR / "03_class_imbalance.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

HIDDEN_DIM     = 128
DROPOUT        = 0.3
EPOCHS         = 150
LR             = 1e-3
BATCH_SIZE     = 256
IMBALANCE_RATIOS = [1, 2, 5, 10, 20]   # Figure 3 x-axis

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

    X_src_full = src[feature_cols].values.astype(np.float32)
    y_src_full = src["label"].values.astype(np.int64)
    X_tgt      = tgt[feature_cols].values.astype(np.float32)
    y_tgt      = tgt["label"].values.astype(np.int64)
    tgt_labeled = tgt["labeled"].values.astype(bool)

    scaler = StandardScaler()
    X_src_full = scaler.fit_transform(X_src_full).astype(np.float32)
    X_tgt      = scaler.transform(X_tgt).astype(np.float32)

    return X_src_full, y_src_full, X_tgt, y_tgt, tgt_labeled, int(meta["n_classes"])


def apply_imbalance(X: np.ndarray, y: np.ndarray,
                    imbalance_ratio: int, rng: np.random.Generator):
    """
    Down-sample minority classes so that
    n_majority / n_minority ≈ imbalance_ratio.
    """
    classes, counts = np.unique(y, return_counts=True)
    majority_class  = classes[counts.argmax()]
    n_majority      = counts.max()
    n_minority      = max(1, n_majority // imbalance_ratio)

    keep_indices = []
    for cls in classes:
        idx = np.where(y == cls)[0]
        if cls == majority_class:
            keep_indices.append(idx)
        else:
            n_keep = min(len(idx), n_minority)
            keep_indices.append(rng.choice(idx, n_keep, replace=False))

    keep = np.concatenate(keep_indices)
    rng.shuffle(keep)
    return X[keep], y[keep]


def run_class_imbalance():
    logger.info("=" * 65)
    logger.info("  Experiment: Class Imbalance (Figure 3)")
    logger.info("=" * 65)

    X_src_full, y_src_full, X_tgt, y_tgt, tgt_labeled, n_classes = load_data()
    input_dim = X_src_full.shape[1]
    rng       = np.random.default_rng(42)

    model_configs = {
        "DANN":        lambda: DANN(input_dim, n_classes, HIDDEN_DIM, DROPOUT),
        "CDAN":        lambda: CDAN(input_dim, n_classes, HIDDEN_DIM, DROPOUT),
        "FixBi":       lambda: FixBi(input_dim, n_classes, HIDDEN_DIM, DROPOUT),
        "ToAlign":     lambda: ToAlign(input_dim, n_classes, HIDDEN_DIM, DROPOUT),
        "DATL (Ours)": lambda: DATL(input_dim, n_classes, HIDDEN_DIM, DROPOUT,
                                     lambda1=0.1, lambda2=0.1, pseudo_threshold=0.85),
    }

    all_records = {name: [] for name in model_configs}

    for ir in IMBALANCE_RATIOS:
        logger.info(f"\nImbalance ratio = {ir}:1")
        X_src_im, y_src_im = apply_imbalance(X_src_full, y_src_full, ir, rng)
        logger.info(f"  Source after imbalance: {X_src_im.shape}, "
                    f"class counts: {dict(zip(*np.unique(y_src_im, return_counts=True)))}")

        for name, build in model_configs.items():
            model = build()
            if name == "DATL (Ours)":
                m = train_datl(model, X_src_im, y_src_im, X_tgt, y_tgt, tgt_labeled,
                               n_classes, EPOCHS, LR, BATCH_SIZE)
            elif name == "FixBi":
                m = train_fixbi(model, X_src_im, y_src_im, X_tgt, y_tgt, tgt_labeled,
                                n_classes, EPOCHS, LR, BATCH_SIZE)
            else:
                m = train_adversarial(model, name, X_src_im, y_src_im,
                                      X_tgt, y_tgt, tgt_labeled,
                                      n_classes, EPOCHS, LR, BATCH_SIZE)

            all_records[name].append({
                "imbalance_ratio": ir,
                "accuracy": m["accuracy"],
                "f1":       m["f1"],
                "auc":      m["auc"],
            })
            logger.info(f"  {name:<18} acc={m['accuracy']:.4f}  "
                        f"f1={m['f1']:.4f}  auc={m['auc']:.4f}")

    with open(TAB_DIR / "class_imbalance.json", "w") as f:
        json.dump(all_records, f, indent=2)

    # ── Plot (accuracy, mirroring Figure 3) ──────────────────────────────────
    colors = {"DANN": "#9E9E9E", "CDAN": "#2196F3", "FixBi": "#4CAF50",
              "ToAlign": "#FF9800", "DATL (Ours)": "#F44336"}
    markers = {"DANN": "o", "CDAN": "s", "FixBi": "^", "ToAlign": "D", "DATL (Ours)": "*"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ["accuracy", "f1", "auc"]
    titles  = ["Accuracy", "F1-Score", "AUC"]

    for ax, metric, title in zip(axes, metrics, titles):
        for name, records in all_records.items():
            xs = [r["imbalance_ratio"] for r in records]
            ys = [r[metric]            for r in records]
            ax.plot(xs, ys, marker=markers[name], label=name,
                    color=colors[name], linewidth=2, markersize=6)
        ax.set_xlabel("Imbalance Ratio (IR)", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f"{title} vs Class Imbalance", fontsize=12, fontweight="bold")
        ax.set_xscale("log")
        ax.set_xticks(IMBALANCE_RATIOS)
        ax.set_xticklabels([f"{ir}:1" for ir in IMBALANCE_RATIOS])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle("Performance under Class Imbalance", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "figure3_class_imbalance.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"\nFigure saved to {FIG_DIR}/figure3_class_imbalance.png")
    logger.info(f"Data saved to {TAB_DIR}/class_imbalance.json")
    return all_records


if __name__ == "__main__":
    run_class_imbalance()
