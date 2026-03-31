"""
Step 1: Train All Models → Reproduce Table 1

Trains all five methods on source→target domain adaptation:
  DANN, CDAN, FixBi, ToAlign, DATL (proposed)

Outputs:
  results/tables/table1_comparison.json    – raw metrics
  results/tables/table1_comparison.txt     – formatted table (matches paper Table 1)
  checkpoints/<model>.pt                   – saved model weights

Usage:
  python3 01_train_all_models.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE_DIR   = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from models import DATL, DANN, CDAN, FixBi, ToAlign
from trainer import train_datl, train_adversarial, train_fixbi, evaluate

PROC_DIR   = BASE_DIR / "data" / "processed"
CKPT_DIR   = BASE_DIR / "checkpoints"
RES_DIR    = BASE_DIR / "results" / "tables"
LOG_DIR    = BASE_DIR / "logs"
for d in [CKPT_DIR, RES_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "01_train_all_models.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ─── Hyper-parameters ─────────────────────────────────────────────────────────
HIDDEN_DIM = 128
DROPOUT    = 0.3
EPOCHS     = 200
LR         = 1e-3
BATCH_SIZE = 256
LAMBDA_ADV = 0.1    # adversarial weight for baselines
LAMBDA1    = 0.1    # MMD weight for DATL
LAMBDA2    = 0.1    # adversarial weight for DATL
PSEUDO_THR = 0.85   # confidence threshold for pseudo-labels


FEATURE_COLS = [
    "cpu_util_percent", "mem_util_percent", "mem_gps",
    "net_in", "net_out", "disk_io_percent",
]


def load_data():
    meta_path = PROC_DIR / "meta.json"
    if not meta_path.exists():
        logger.error("Processed data not found! Run 00_prepare_data.py first.")
        sys.exit(1)

    with open(meta_path) as f:
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

    logger.info(f"Using {len(feature_cols)} features: {feature_cols}")

    X_src = src[feature_cols].values.astype(np.float32)
    y_src = src["label"].values.astype(np.int64)

    X_tgt = tgt[feature_cols].values.astype(np.float32)
    y_tgt = tgt["label"].values.astype(np.int64)
    tgt_labeled = tgt["labeled"].values.astype(bool)

    # Standardise (fit on source only)
    scaler = StandardScaler()
    X_src  = scaler.fit_transform(X_src).astype(np.float32)
    X_tgt  = scaler.transform(X_tgt).astype(np.float32)

    n_classes   = int(meta["n_classes"])
    input_dim   = X_src.shape[1]

    logger.info(f"Source: {X_src.shape}, classes={n_classes}")
    logger.info(f"Target: {X_tgt.shape}, labeled={tgt_labeled.sum()} / {len(tgt_labeled)}")
    return X_src, y_src, X_tgt, y_tgt, tgt_labeled, n_classes, input_dim


def run_all_models():
    logger.info("=" * 65)
    logger.info("  Training all domain adaptation models (Table 1 replication)")
    logger.info("=" * 65)

    X_src, y_src, X_tgt, y_tgt, tgt_labeled, n_classes, input_dim = load_data()

    results = {}
    t0 = time.time()

    # ── 1. DANN ───────────────────────────────────────────────────────────────
    logger.info("\n[1/5] Training DANN ...")
    model = DANN(input_dim, n_classes, HIDDEN_DIM, DROPOUT)
    m = train_adversarial(model, "DANN", X_src, y_src, X_tgt, y_tgt, tgt_labeled,
                          n_classes, EPOCHS, LR, BATCH_SIZE, LAMBDA_ADV,
                          CKPT_DIR / "dann.pt")
    results["DANN"] = m
    logger.info(f"  DANN  → acc={m['accuracy']*100:.1f}  f1={m['f1']*100:.1f}  auc={m['auc']*100:.1f}")

    # ── 2. CDAN ───────────────────────────────────────────────────────────────
    logger.info("\n[2/5] Training CDAN ...")
    model = CDAN(input_dim, n_classes, HIDDEN_DIM, DROPOUT)
    m = train_adversarial(model, "CDAN", X_src, y_src, X_tgt, y_tgt, tgt_labeled,
                          n_classes, EPOCHS, LR, BATCH_SIZE, LAMBDA_ADV,
                          CKPT_DIR / "cdan.pt")
    results["CDAN"] = m
    logger.info(f"  CDAN  → acc={m['accuracy']*100:.1f}  f1={m['f1']*100:.1f}  auc={m['auc']*100:.1f}")

    # ── 3. FixBi ──────────────────────────────────────────────────────────────
    logger.info("\n[3/5] Training FixBi ...")
    model = FixBi(input_dim, n_classes, HIDDEN_DIM, DROPOUT)
    m = train_fixbi(model, X_src, y_src, X_tgt, y_tgt, tgt_labeled,
                    n_classes, EPOCHS, LR, BATCH_SIZE, LAMBDA_ADV,
                    CKPT_DIR / "fixbi.pt")
    results["FixBi"] = m
    logger.info(f"  FixBi → acc={m['accuracy']*100:.1f}  f1={m['f1']*100:.1f}  auc={m['auc']*100:.1f}")

    # ── 4. ToAlign ────────────────────────────────────────────────────────────
    logger.info("\n[4/5] Training ToAlign ...")
    model = ToAlign(input_dim, n_classes, HIDDEN_DIM, DROPOUT)
    m = train_adversarial(model, "ToAlign", X_src, y_src, X_tgt, y_tgt, tgt_labeled,
                          n_classes, EPOCHS, LR, BATCH_SIZE, LAMBDA_ADV,
                          CKPT_DIR / "toalign.pt")
    results["ToAlign"] = m
    logger.info(f"  ToAlign → acc={m['accuracy']*100:.1f}  f1={m['f1']*100:.1f}  auc={m['auc']*100:.1f}")

    # ── 5. DATL (proposed) ────────────────────────────────────────────────────
    logger.info("\n[5/5] Training DATL (proposed) ...")
    model = DATL(input_dim, n_classes, HIDDEN_DIM, DROPOUT,
                 lambda1=LAMBDA1, lambda2=LAMBDA2, pseudo_threshold=PSEUDO_THR)
    m = train_datl(model, X_src, y_src, X_tgt, y_tgt, tgt_labeled,
                   n_classes, EPOCHS, LR, BATCH_SIZE,
                   save_path=CKPT_DIR / "datl.pt")
    results["DATL (Ours)"] = m
    logger.info(f"  DATL  → acc={m['accuracy']*100:.1f}  f1={m['f1']*100:.1f}  auc={m['auc']*100:.1f}")

    elapsed = time.time() - t0
    logger.info(f"\nTotal training time: {elapsed/60:.1f} minutes")

    # ── Save raw results ──────────────────────────────────────────────────────
    with open(RES_DIR / "table1_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Print formatted table ─────────────────────────────────────────────────
    header = f"\n{'Method':<20}  {'Accuracy':>9}  {'F1-Score':>9}  {'AUC':>9}"
    sep    = "-" * 54
    lines  = [header, sep]
    paper_results = {
        "DANN":        {"accuracy": 84.2, "f1": 81.0, "auc": 87.2},
        "CDAN":        {"accuracy": 85.7, "f1": 82.3, "auc": 88.5},
        "FixBi":       {"accuracy": 86.3, "f1": 83.1, "auc": 89.2},
        "ToAlign":     {"accuracy": 87.1, "f1": 83.7, "auc": 89.9},
        "DATL (Ours)": {"accuracy": 89.6, "f1": 85.4, "auc": 91.3},
    }
    for method, m in results.items():
        lines.append(
            f"  {method:<18}  {m['accuracy']*100:>8.1f}  "
            f"{m['f1']*100:>8.1f}  {m['auc']*100:>8.1f}"
        )
    lines.append(sep)
    lines.append("\n  Paper results (for reference):")
    lines.append(sep)
    for method, m in paper_results.items():
        lines.append(
            f"  {method:<18}  {m['accuracy']:>8.1f}  "
            f"{m['f1']:>8.1f}  {m['auc']:>8.1f}"
        )
    lines.append(sep)
    table_str = "\n".join(lines)

    logger.info(table_str)

    with open(RES_DIR / "table1_comparison.txt", "w") as f:
        f.write(table_str + "\n")

    logger.info(f"\nResults saved to {RES_DIR}/table1_comparison.*")
    return results


if __name__ == "__main__":
    run_all_models()
