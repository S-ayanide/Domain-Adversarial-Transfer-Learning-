# Replication: Domain-Adversarial Transfer Learning for Fault Root Cause Identification

Full replication of:

> Fang, B. & Gao, D. (2025). *Domain-Adversarial Transfer Learning for Fault Root Cause
> Identification in Cloud Computing Systems.* 2025 4th International Conference on Robotics,
> Artificial Intelligence and Intelligent Control (RAIIC). IEEE.

---

## What This Replication Covers

| Paper output | Script | Result file |
|---|---|---|
| Table 1 – Comparative results | `01_train_all_models.py` | `results/tables/table1_comparison.txt` |
| Figure 2 – Label scarcity | `02_experiment_label_scarcity.py` | `results/figures/figure2_label_scarcity.png` |
| Figure 3 – Class imbalance | `03_experiment_class_imbalance.py` | `results/figures/figure3_class_imbalance.png` |
| Figure 4 – Heterogeneous nodes | `04_experiment_heterogeneous_nodes.py` | `results/figures/figure4_heterogeneous_nodes.png` |

---

## Dataset

The paper uses the **Alibaba Cluster Trace 2018** dataset
(https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018).

### Option A — Use the real data (recommended)

```bash
cd data/raw
wget http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/machine_meta.tar.gz
wget http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/machine_usage.tar.gz
tar -xzf machine_meta.tar.gz
tar -xzf machine_usage.tar.gz
```

Then run `python3 00_prepare_data.py` — it will detect the CSV files automatically.

### Option B — Synthetic data (default if CSVs are absent)

`00_prepare_data.py` automatically generates **synthetic telemetry** that faithfully mirrors
the Alibaba 2018 `machine_usage.csv` schema (CPU%, memory%, memory bandwidth, network in/out,
disk I/O) so the full pipeline runs end-to-end without downloading the 1.7 GB file.

---

## Model Architecture

The proposed **DATL** model (Figure 1 of the paper):

```
Input x  →  Feature Extractor F(·; θ_f)  →  feat
                    │
                    ├──→  Classifier C(·; θ_c)    →  class logits
                    │
                    └──→  Domain Discriminator D(·; θ_d)  →  domain label
                          (via Gradient Reversal Layer)
```

**Total loss** (paper Eq. 4):

```
L_total = L_s  +  λ₁ · L_mmd  +  λ₂ · L_adv
```

- `L_s`   — cross-entropy on source labels (Eq. 1)
- `L_mmd` — Maximum Mean Discrepancy between feature distributions (Eq. 2)
- `L_adv` — adversarial domain discrimination loss via GRL (Eq. 3)
- **Pseudo-label mechanism**: high-confidence target predictions (p ≥ δ = 0.85)
  are added to the source training set each `pseudo_update_freq` epochs.

---

## File Structure

```
replicate/
├── 00_prepare_data.py              # Step 0: download / generate data
├── 01_train_all_models.py          # Step 1: train all 5 models → Table 1
├── 02_experiment_label_scarcity.py # Step 2: vary labeled % → Figure 2
├── 03_experiment_class_imbalance.py# Step 3: vary class IR → Figure 3
├── 04_experiment_heterogeneous_nodes.py # Step 4: per-node-type → Figure 4
├── run_all.py                      # One-shot: runs all steps
├── models.py                       # All neural architectures
├── trainer.py                      # Training + evaluation engine
├── requirements.txt
├── data/
│   ├── raw/                        # Downloaded CSVs go here
│   └── processed/                  # source_domain.parquet, target_domain.parquet
├── checkpoints/                    # Saved model weights (.pt)
├── results/
│   ├── tables/                     # .json and .txt result files
│   └── figures/                    # figure2.png, figure3.png, figure4.png
└── logs/                           # Training logs
```

---

## Quick Start

```bash
cd replicate

# Install dependencies
pip install -r requirements.txt

# Run everything (uses synthetic data by default)
python3 run_all.py

# Or run step by step:
python3 00_prepare_data.py
python3 01_train_all_models.py
python3 02_experiment_label_scarcity.py
python3 03_experiment_class_imbalance.py
python3 04_experiment_heterogeneous_nodes.py
```

---

## Implemented Baselines

| Method | Reference | Description |
|--------|-----------|-------------|
| **DANN** | Ganin et al. (2016) | Domain-adversarial training with GRL |
| **CDAN** | Long et al. (2018) | Conditional adversarial — discriminator conditioned on classifier output |
| **FixBi** | Na et al. (2021, CVPR) | Bipartite dual-classifier consistency regularization |
| **ToAlign** | Wei et al. (2021, NeurIPS) | Task-oriented feature decomposition + adversarial alignment |
| **DATL** (proposed) | Fang & Gao (2025) | MMD + adversarial + pseudo-labels |

---

## Key Hyper-parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `hidden_dim` | 128 | Feature extractor hidden size |
| `epochs` | 100 | Training epochs (Table 1); 80 (experiments) |
| `lr` | 1e-3 | Adam learning rate |
| `lambda1` | 0.1 | MMD loss weight (DATL) |
| `lambda2` | 0.1 | Adversarial loss weight |
| `pseudo_threshold` | 0.85 | Min confidence for pseudo-label inclusion |
| `batch_size` | 256 | Mini-batch size |
