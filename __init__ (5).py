"""
baselines/hive_static.py
------------------------
Static Hive-style partitioning baseline.
Uses a fixed partition key chosen by DBA heuristic (most common WHERE column).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from collections import Counter

from environment.repartitioner import Repartitioner
from environment.cost_model import CostModel

DATA_DIR = Path(__file__).parent.parent / "data"


def pick_hive_partition_key(workload: list[dict]) -> str:
    """
    Simple heuristic: pick the column that appears most frequently
    in WHERE clauses across all queries.
    """
    counter = Counter()
    for entry in workload:
        for pred in entry.get("predicates", []):
            counter[pred] += 1
    if not counter:
        return "sale_year"
    return counter.most_common(1)[0][0]


def run_hive_baseline(workload_path: str = None) -> dict:
    """Apply static Hive partitioning and measure cost."""
    workload_path = workload_path or str(DATA_DIR / "workload_log.json")

    with open(workload_path) as f:
        workload = json.load(f)

    partition_key = pick_hive_partition_key(workload)
    print(f"[hive_static] Chosen partition key: {partition_key}")

    # Map key to granularity
    gran_map = {
        "sale_year": "year",
        "sale_month": "month",
        "sale_day": "day",
        "category": "value",
        "region": "value",
        "store_id": "hash_16",
        "customer_id": "hash_32",
    }
    granularity = gran_map.get(partition_key, "value")

    repartitioner = Repartitioner(str(DATA_DIR / "sales_flat.parquet"))
    result = repartitioner.apply(
        partition_key=partition_key,
        granularity=granularity,
        output_dir=str(DATA_DIR / "baseline_hive"),
    )

    cost_model = CostModel()
    flat_cost = cost_model.measure(str(DATA_DIR / "sales_flat.parquet"))
    hive_cost = cost_model.measure(result["output_path"])

    improvement = (
        (flat_cost["total_bytes_scanned"] - hive_cost["total_bytes_scanned"])
        / max(flat_cost["total_bytes_scanned"], 1.0)
    )

    return {
        "method": "hive_static",
        "partition_key": partition_key,
        "granularity": granularity,
        "n_files": result["n_files"],
        "total_bytes_mb": result["total_bytes"] / 1e6,
        "flat_bytes_scanned_mb": flat_cost["total_bytes_scanned"] / 1e6,
        "hive_bytes_scanned_mb": hive_cost["total_bytes_scanned"] / 1e6,
        "bytes_improvement_pct": improvement * 100,
        "flat_avg_latency_ms": flat_cost["avg_latency_ms"],
        "hive_avg_latency_ms": hive_cost["avg_latency_ms"],
    }


if __name__ == "__main__":
    flat = DATA_DIR / "sales_flat.parquet"
    if not flat.exists():
        print("Run workload/generator.py first.")
        sys.exit(1)

    results = run_hive_baseline()
    print("\n=== Hive Static Baseline Results ===")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
