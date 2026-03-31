"""
Step 0: Download and Prepare Alibaba Cluster Trace 2018

Downloads machine_meta.csv and machine_usage.csv from:
  http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces

If download fails, generates realistic synthetic data matching the dataset schema
so the rest of the pipeline still runs end-to-end.

Output:
  data/processed/source_domain.parquet   - labeled data from stable high-load nodes
  data/processed/target_domain.parquet   - partially labeled data from different nodes
  data/processed/meta.json               - domain split metadata
"""

import os
import sys
import json
import tarfile
import urllib.request
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from io import BytesIO

BASE_DIR   = Path(__file__).parent
RAW_DIR    = BASE_DIR / "data" / "raw"
PROC_DIR   = BASE_DIR / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

ALIBABA_BASE = "http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces"

# ─── Feature columns we use (matches paper description) ───────────────────────
FEATURE_COLS = [
    "cpu_util_percent",    # CPU usage
    "mem_util_percent",    # Memory consumption
    "mem_gps",             # Memory bandwidth
    "net_in",              # Network in
    "net_out",             # Network out
    "disk_io_percent",     # Disk I/O
]

# Fault types derived from resource patterns (multi-class)
# 0=normal, 1=cpu_fault, 2=memory_fault, 3=disk_fault, 4=network_fault, 5=mixed_fault
FAULT_NAMES = {
    0: "Normal",
    1: "CPU_Overload",
    2: "Memory_Leak",
    3: "Disk_IO_Fault",
    4: "Network_Fault",
    5: "Mixed_Fault",
}
N_CLASSES = len(FAULT_NAMES)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Attempt real download
# ──────────────────────────────────────────────────────────────────────────────

def try_download(filename: str, dest: Path, timeout: int = 30) -> bool:
    url = f"{ALIBABA_BASE}/{filename}"
    print(f"  Trying {url} ...")
    try:
        req = urllib.request.urlopen(url, timeout=timeout)
        data = req.read(1024)        # read just 1 KB to verify the URL works
        if len(data) < 100:
            return False
        print(f"  URL accessible – downloading full file to {dest} ...")
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def load_machine_meta(raw_dir: Path) -> Optional[pd.DataFrame]:
    dest = raw_dir / "machine_meta.tar.gz"
    csv  = raw_dir / "machine_meta.csv"
    if csv.exists():
        return pd.read_csv(csv, header=None,
                           names=["machine_id","time_stamp","failure_domain_1",
                                  "failure_domain_2","cpu_num","mem_size","status"])
    if not dest.exists():
        ok = try_download("machine_meta.tar.gz", dest)
        if not ok:
            return None
    try:
        with tarfile.open(dest, "r:gz") as tf:
            tf.extractall(raw_dir)
        f = list(raw_dir.glob("machine_meta.csv"))
        if f:
            return pd.read_csv(f[0], header=None,
                               names=["machine_id","time_stamp","failure_domain_1",
                                      "failure_domain_2","cpu_num","mem_size","status"])
    except Exception as e:
        print(f"  Could not extract machine_meta: {e}")
    return None


def load_machine_usage_sample(raw_dir: Path,
                              target_rows: int = 500_000,
                              seed: int = 42) -> Optional[pd.DataFrame]:
    """
    Load a RANDOM sample of rows from machine_usage.csv spread across
    the full file (not just the first N rows, which are all the same machine).

    Uses reservoir-style probability sampling via skiprows callable so only
    one pass over the file is needed.
    """
    col_names = ["machine_id", "time_stamp", "cpu_util_percent",
                 "mem_util_percent", "mem_gps", "mkpi",
                 "net_in", "net_out", "disk_io_percent"]

    def _load(csv_path: Path) -> pd.DataFrame:
        # Estimate total row count from file size (each row ≈ 70 bytes)
        file_bytes   = csv_path.stat().st_size
        est_total    = max(file_bytes // 70, target_rows * 2)
        sample_prob  = min(1.0, (target_rows * 1.5) / est_total)
        print(f"  Sampling ~{target_rows:,} rows (p={sample_prob:.4f}) "
              f"from {file_bytes/1e9:.2f} GB file ...")

        rng_skip = np.random.default_rng(seed)

        # skiprows callable: keep row 0 (header check disabled via header=None),
        # then sample each subsequent row probabilistically
        def _skip(i):
            return i > 0 and rng_skip.random() > sample_prob

        df = pd.read_csv(csv_path, header=None, skiprows=_skip,
                         names=col_names, low_memory=False)
        print(f"  Loaded {len(df):,} rows (sampled across full file)")
        return df

    csv = raw_dir / "machine_usage.csv"
    if csv.exists():
        return _load(csv)

    dest = raw_dir / "machine_usage.tar.gz"
    if not dest.exists():
        print("  machine_usage.tar.gz not found; skipping download (1.7 GB).")
        print("  Run fetchData.sh in data/raw/ to download the real file.")
        return None
    try:
        print("  Extracting machine_usage.tar.gz (may take a while) ...")
        with tarfile.open(dest, "r:gz") as tf:
            tf.extractall(raw_dir)
        f = list(raw_dir.glob("machine_usage.csv"))
        if f:
            return _load(f[0])
    except Exception as e:
        print(f"  Could not extract machine_usage: {e}")
    return None


# ──────────────────────────────────────────────────────────────────────────────
# 2. Assign fault labels from resource patterns
#    (used when real data is available)
# ──────────────────────────────────────────────────────────────────────────────

def assign_fault_labels(df: pd.DataFrame,
                        thresholds: Optional[dict] = None) -> pd.Series:
    """
    Percentile-based fault labeling.

    Thresholds are computed from the actual data distribution (85th percentile
    for CPU/mem/disk, 80th for combined network) so that fault classes are
    present regardless of absolute scale — critical for real Alibaba data where
    memory utilisation is consistently high across all machines.

    If `thresholds` is provided (pre-computed), those values are used directly
    so that source and target share the same decision boundaries.

    Class mapping:
      0 = Normal
      1 = CPU_Overload   (cpu > p85)
      2 = Memory_Leak    (mem > p85)
      3 = Disk_IO_Fault  (disk > p85)
      4 = Network_Fault  (net_in+net_out > p80)
      5 = Mixed_Fault    (two or more of the above)
    """
    if thresholds is None:
        thresholds = {
            "cpu":  float(np.percentile(df["cpu_util_percent"].clip(0, 100), 85)),
            "mem":  float(np.percentile(df["mem_util_percent"].clip(0, 100), 85)),
            "disk": float(np.percentile(df["disk_io_percent"].clip(0, 100),  85)),
            "net":  float(np.percentile(
                (df["net_in"] + df["net_out"]).clip(0, 200), 80)),
        }

    cpu_fault  = df["cpu_util_percent"]               > thresholds["cpu"]
    mem_fault  = df["mem_util_percent"]               > thresholds["mem"]
    disk_fault = df["disk_io_percent"]                > thresholds["disk"]
    net_fault  = (df["net_in"] + df["net_out"])       > thresholds["net"]

    labels = np.zeros(len(df), dtype=int)
    multi  = (cpu_fault.astype(int) + mem_fault.astype(int)
              + disk_fault.astype(int) + net_fault.astype(int)) >= 2
    labels[multi]               = 5
    labels[cpu_fault  & ~multi] = 1
    labels[mem_fault  & ~multi] = 2
    labels[disk_fault & ~multi] = 3
    labels[net_fault  & ~multi] = 4
    return pd.Series(labels, index=df.index), thresholds


# ──────────────────────────────────────────────────────────────────────────────
# 3. Synthetic data generator
#    Closely mirrors Alibaba 2018 machine_usage statistics and paper description
# ──────────────────────────────────────────────────────────────────────────────

def generate_synthetic_alibaba(
    n_source: int = 12_000,
    n_target: int = 8_000,
    seed: int = 42,
    source_high_load: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic cloud telemetry matching Alibaba 2018 schema.

    Source domain: stable high-load nodes (paper description: "stable nodes
    operating under high load, large number of labeled fault events")
    Target domain: different node group (paper description: "nodes that differ
    in structure or runtime environment")
    """
    rng = np.random.default_rng(seed)
    print("  Generating synthetic data (mirrors Alibaba 2018 schema) ...")

    def _make_node_batch(n, node_type="high_load", domain_id=0):
        """
        node_type:
          high_load   – source domain: 30–95% CPU, 40–90% mem
          mem_heavy   – memory-intensive target nodes
          io_heavy    – I/O-bound target nodes
          mixed       – mixed workload target nodes
          cpu_heavy   – CPU-intensive target nodes
        """
        rows = []
        n_machines = max(50, n // 200)
        machine_ids = [f"m_{domain_id}_{i:04d}" for i in range(n_machines)]

        for _ in range(n):
            mid = rng.choice(machine_ids)
            if node_type == "high_load":
                cpu  = rng.uniform(30, 95)
                mem  = rng.uniform(40, 90)
                disk = rng.uniform(5, 70)
                net  = rng.uniform(5, 80)
            elif node_type == "cpu_heavy":
                cpu  = rng.uniform(60, 100)
                mem  = rng.uniform(20, 60)
                disk = rng.uniform(5, 40)
                net  = rng.uniform(5, 50)
            elif node_type == "mem_heavy":
                cpu  = rng.uniform(20, 60)
                mem  = rng.uniform(60, 100)
                disk = rng.uniform(5, 40)
                net  = rng.uniform(5, 50)
            elif node_type == "io_heavy":
                cpu  = rng.uniform(20, 70)
                mem  = rng.uniform(20, 70)
                disk = rng.uniform(50, 100)
                net  = rng.uniform(20, 80)
            else:  # mixed
                cpu  = rng.uniform(20, 90)
                mem  = rng.uniform(20, 90)
                disk = rng.uniform(10, 80)
                net  = rng.uniform(10, 80)

            mem_gps  = mem * 0.6 + rng.uniform(-5, 5)
            net_out  = net * 0.8 + rng.uniform(-5, 5)
            mkpi     = int(rng.uniform(100, 3000))
            ts       = rng.uniform(0, 691200)   # 8 days in seconds
            rows.append({
                "machine_id":        mid,
                "time_stamp":        ts,
                "cpu_util_percent":  float(np.clip(cpu, 0, 100)),
                "mem_util_percent":  float(np.clip(mem, 0, 100)),
                "mem_gps":           float(np.clip(mem_gps, 0, 100)),
                "mkpi":              mkpi,
                "net_in":            float(np.clip(net, 0, 100)),
                "net_out":           float(np.clip(net_out, 0, 100)),
                "disk_io_percent":   float(np.clip(disk, 0, 100)),
                "failure_domain_1":  domain_id,
                "node_type":         node_type,
            })
        return pd.DataFrame(rows)

    # Source domain: stable high-load nodes (labeled)
    src = _make_node_batch(n_source, "high_load", domain_id=0)
    src["label"], thr = assign_fault_labels(src)
    src["domain"]  = "source"
    src["labeled"] = True

    # Target domain: different node types (partially labeled – only 30%)
    n_per_type = n_target // 4
    tgt_parts = []
    for i, ntype in enumerate(["cpu_heavy", "mem_heavy", "io_heavy", "mixed"]):
        part = _make_node_batch(n_per_type, ntype, domain_id=i + 1)
        tgt_parts.append(part)
    tgt = pd.concat(tgt_parts, ignore_index=True)
    tgt["label"], _ = assign_fault_labels(tgt, thresholds=thr)
    tgt["domain"]   = "target"
    labeled_mask    = rng.random(len(tgt)) < 0.30
    tgt["labeled"]  = labeled_mask

    return src, tgt


# ──────────────────────────────────────────────────────────────────────────────
# 4. Build source/target from real Alibaba data
# ──────────────────────────────────────────────────────────────────────────────

def build_domains_from_real(meta: pd.DataFrame, usage: pd.DataFrame):
    """
    Split real Alibaba data into source and target domains.

    Per paper:
      Source = stable nodes under high load (first half of failure domains)
      Target = nodes differing in structure (second half of failure domains)

    Fault thresholds are computed on the FULL dataset first, then applied to
    both domains so they share consistent decision boundaries.
    """
    print("  Building domains from real Alibaba 2018 data ...")

    # Merge machine metadata (failure domain, CPU count, memory size)
    meta_last = (meta.sort_values("time_stamp")
                     .groupby("machine_id")
                     .last()
                     .reset_index()[["machine_id", "failure_domain_1",
                                     "failure_domain_2", "cpu_num", "mem_size"]])
    df = usage.merge(meta_last, on="machine_id", how="left")
    df["failure_domain_1"] = df["failure_domain_1"].fillna(0).astype(int)

    # Remove abnormal sentinel values (-1, 101) and clip to [0, 100]
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace(-1, np.nan).replace(101, np.nan)
            df[col] = df[col].clip(0, 100)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    print(f"  After cleaning: {len(df):,} rows across "
          f"{df['machine_id'].nunique():,} machines")

    # Compute thresholds on the full dataset so source & target share them
    _, thresholds = assign_fault_labels(df)
    print(f"  Fault thresholds (percentile-based): {thresholds}")
    df["label"], _ = assign_fault_labels(df, thresholds=thresholds)

    # Domain split: sort unique failure_domain_1 values, first half → source
    unique_domains = sorted(df["failure_domain_1"].unique())
    split_point    = max(1, len(unique_domains) // 2)
    src_domains    = unique_domains[:split_point]
    tgt_domains    = unique_domains[split_point:] if len(unique_domains) > 1 \
                     else unique_domains          # fallback: same domain

    print(f"  Source failure_domains: {src_domains[:5]}{'...' if len(src_domains)>5 else ''}")
    print(f"  Target failure_domains: {tgt_domains[:5]}{'...' if len(tgt_domains)>5 else ''}")

    src = df[df["failure_domain_1"].isin(src_domains)].copy()
    tgt = df[df["failure_domain_1"].isin(tgt_domains)].copy()

    # Ensure both splits have all 6 classes; if target is empty use a random split
    if len(tgt) == 0 or tgt["label"].nunique() < 2:
        print("  WARNING: Domain split produced empty/single-class target. "
              "Falling back to random 60/40 machine split.")
        all_machines = df["machine_id"].unique()
        rng_split    = np.random.default_rng(42)
        rng_split.shuffle(all_machines)
        split_m      = int(len(all_machines) * 0.6)
        src = df[df["machine_id"].isin(all_machines[:split_m])].copy()
        tgt = df[df["machine_id"].isin(all_machines[split_m:])].copy()

    src["domain"]  = "source"
    src["labeled"] = True
    tgt["domain"]  = "target"
    rng = np.random.default_rng(42)
    tgt["labeled"] = rng.random(len(tgt)) < 0.30

    # Node type from data-adaptive percentile thresholds
    cpu_p70  = float(df["cpu_util_percent"].quantile(0.70))
    mem_p70  = float(df["mem_util_percent"].quantile(0.70))
    disk_p60 = float(df["disk_io_percent"].quantile(0.60))

    def _node_type(row):
        if row["cpu_util_percent"]  > cpu_p70:  return "cpu_heavy"
        if row["mem_util_percent"]  > mem_p70:  return "mem_heavy"
        if row["disk_io_percent"]   > disk_p60: return "io_heavy"
        return "mixed"

    for d in [src, tgt]:
        d["node_type"] = d.apply(_node_type, axis=1)

    return src.reset_index(drop=True), tgt.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Step 0: Preparing Alibaba Cluster Trace 2018 Data")
    print("=" * 65)

    # Try to load real data first
    print("\n[1/3] Attempting to load Alibaba 2018 raw data ...")
    meta  = load_machine_meta(RAW_DIR)
    usage = load_machine_usage_sample(RAW_DIR, target_rows=500_000)

    if meta is not None and usage is not None:
        print(f"  Loaded machine_meta: {len(meta):,} rows")
        print(f"  Loaded machine_usage: {len(usage):,} rows")
        print("\n[2/3] Building source and target domains from real data ...")
        src, tgt = build_domains_from_real(meta, usage)
        data_source = "real"
    else:
        print("  Real data not available – using synthetic data.")
        print("  (Download machine_usage.tar.gz to use actual Alibaba 2018 data)")
        print("\n[2/3] Generating synthetic data ...")
        src, tgt = generate_synthetic_alibaba(n_source=12_000, n_target=8_000)
        data_source = "synthetic"

    print(f"\n  Source domain: {len(src):,} samples")
    print(f"  Target domain: {len(tgt):,} samples")

    # Class distribution
    print("\n  Source fault distribution:")
    for k, v in src["label"].value_counts().sort_index().items():
        print(f"    {FAULT_NAMES[k]:>16s} (class {k}): {v:>5,}  ({v/len(src)*100:.1f}%)")
    print("\n  Target fault distribution:")
    for k, v in tgt["label"].value_counts().sort_index().items():
        print(f"    {FAULT_NAMES[k]:>16s} (class {k}): {v:>5,}  ({v/len(tgt)*100:.1f}%)")

    print("\n[3/3] Saving processed data ...")
    src.to_parquet(PROC_DIR / "source_domain.parquet", index=False)
    tgt.to_parquet(PROC_DIR / "target_domain.parquet", index=False)

    # Save metadata
    meta_info = {
        "data_source":  data_source,
        "n_source":     int(len(src)),
        "n_target":     int(len(tgt)),
        "n_classes":    N_CLASSES,
        "fault_names":  FAULT_NAMES,
        "feature_cols": FEATURE_COLS,
        "target_labeled_pct": float((tgt["labeled"].sum() / len(tgt)) * 100),
        "source_label_dist":  {str(k): int(v) for k, v in
                               src["label"].value_counts().sort_index().items()},
        "target_label_dist":  {str(k): int(v) for k, v in
                               tgt["label"].value_counts().sort_index().items()},
    }
    with open(PROC_DIR / "meta.json", "w") as f:
        json.dump(meta_info, f, indent=2)

    print(f"\n  Saved source_domain.parquet ({len(src):,} rows)")
    print(f"  Saved target_domain.parquet ({len(tgt):,} rows)")
    print(f"  Saved meta.json")
    print(f"\n  Data source: {data_source.upper()}")
    print("\n  Done. Run 01_train_models.py next.")
    print("=" * 65)


if __name__ == "__main__":
    main()
