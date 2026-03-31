"""
Master script — runs the full replication pipeline end-to-end.

Steps:
  0. Prepare data     (download Alibaba 2018 or generate synthetic)
  1. Train all models → Table 1
  2. Experiment: Label scarcity   → Figure 2
  3. Experiment: Class imbalance  → Figure 3
  4. Experiment: Heterogeneous nodes → Figure 4
  5. Print final summary

Usage:
  python3 run_all.py                    # full pipeline
  python3 run_all.py --skip-data        # skip data prep (already done)
  python3 run_all.py --only-table1      # only Table 1

Outputs land in results/tables/ and results/figures/.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent
LOG_DIR  = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "run_all.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def banner(title: str):
    logger.info("\n" + "=" * 65)
    logger.info(f"  {title}")
    logger.info("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Full replication pipeline")
    parser.add_argument("--skip-data",      action="store_true",
                        help="Skip Step 0 (data already prepared)")
    parser.add_argument("--only-table1",    action="store_true",
                        help="Only run Step 1 (Table 1 comparison)")
    parser.add_argument("--skip-imbalance", action="store_true",
                        help="Skip the class imbalance experiment (it is slow)")
    args = parser.parse_args()

    t_start = time.time()

    # ── Step 0: Data preparation ──────────────────────────────────────────────
    if not args.skip_data:
        banner("Step 0/4: Data Preparation")
        import importlib, importlib.util
        spec = importlib.util.spec_from_file_location(
            "prepare", BASE_DIR / "00_prepare_data.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()
    else:
        logger.info("Skipping Step 0 (--skip-data)")

    # ── Step 1: Train all models (Table 1) ────────────────────────────────────
    banner("Step 1/4: Training All Models (Table 1)")
    from importlib import import_module
    import importlib.util

    def run_module(filename):
        spec = importlib.util.spec_from_file_location("_mod", BASE_DIR / filename)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    mod1 = run_module("01_train_all_models.py")
    table1_results = mod1.run_all_models()

    if args.only_table1:
        logger.info("--only-table1 flag set; stopping after Table 1.")
        _print_summary(table1_results, {}, {}, time.time() - t_start)
        return

    # ── Step 2: Label scarcity ────────────────────────────────────────────────
    banner("Step 2/4: Label Scarcity Experiment (Figure 2)")
    mod2 = run_module("02_experiment_label_scarcity.py")
    scarcity_results = mod2.run_label_scarcity()

    # ── Step 3: Class imbalance ────────────────────────────────────────────────
    if not args.skip_imbalance:
        banner("Step 3/4: Class Imbalance Experiment (Figure 3)")
        mod3 = run_module("03_experiment_class_imbalance.py")
        imbalance_results = mod3.run_class_imbalance()
    else:
        logger.info("Skipping Step 3 (--skip-imbalance)")
        imbalance_results = {}

    # ── Step 4: Heterogeneous nodes ────────────────────────────────────────────
    banner("Step 4/4: Heterogeneous Nodes Experiment (Figure 4)")
    mod4 = run_module("04_experiment_heterogeneous_nodes.py")
    hetero_results = mod4.run_heterogeneous_nodes()

    # ── Summary ────────────────────────────────────────────────────────────────
    _print_summary(table1_results, scarcity_results, hetero_results,
                   time.time() - t_start)


def _print_summary(table1, scarcity, hetero, elapsed):
    banner("FINAL SUMMARY")
    logger.info("\n  TABLE 1 — Comparative Results")
    logger.info(f"  {'Method':<20}  {'Acc':>6}  {'F1':>6}  {'AUC':>6}")
    logger.info("  " + "-" * 42)
    for method, m in table1.items():
        logger.info(f"  {method:<20}  {m['accuracy']*100:>5.1f}  "
                    f"{m['f1']*100:>5.1f}  {m['auc']*100:>5.1f}")

    if scarcity:
        logger.info("\n  FIGURE 2 — Label Scarcity (DATL only)")
        logger.info(f"  {'Ratio':>6}  {'Acc':>6}  {'F1':>6}  {'AUC':>6}")
        logger.info("  " + "-" * 32)
        for r in scarcity:
            logger.info(f"  {r['labeled_ratio']:>5.0%}  "
                        f"{r['accuracy']*100:>5.1f}  "
                        f"{r['f1']*100:>5.1f}  "
                        f"{r['auc']*100:>5.1f}")

    logger.info(f"\n  Total wall-clock time: {elapsed/60:.1f} minutes")
    logger.info("\n  All results in replicate/results/")
    logger.info("  Figures:  results/figures/")
    logger.info("  Tables:   results/tables/")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
