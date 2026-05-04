"""
baselines/zorder.py
-------------------
Z-ordering (space-filling curve sort) baseline using delta-rs.
Z-ordering sorts data along multiple dimensions, which improves
data skipping across many query predicates.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import shutil
import time
import json

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

from environment.cost_model import CostModel

DATA_DIR = Path(__file__).parent.parent / "data"


def z_order_sort_key(table: pa.Table, z_cols: list[str], n_bits: int = 16) -> np.ndarray:
    """
    Compute Z-order (Morton code) sort keys for the given columns.
    Each column value is quantized to n_bits bits and interleaved.
    """
    n_rows = len(table)
    codes = np.zeros(n_rows, dtype=np.int64)

    for bit_pos, col_name in enumerate(z_cols):
        col = table.column(col_name)
        try:
            arr = col.cast(pa.float64()).to_pylist()
            arr = np.array([v if v is not None else 0.0 for v in arr])
            # Normalize to [0, 2^n_bits)
            arr_min, arr_max = arr.min(), arr.max()
            if arr_max > arr_min:
                normalized = ((arr - arr_min) / (arr_max - arr_min) * (2**n_bits - 1)).astype(np.int64)
            else:
                normalized = np.zeros(n_rows, dtype=np.int64)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            # Categorical: encode as ordinal
            unique_vals = sorted(set(col.to_pylist()))
            val_to_idx = {v: i for i, v in enumerate(unique_vals)}
            raw = [val_to_idx.get(v, 0) for v in col.to_pylist()]
            arr = np.array(raw, dtype=np.int64)
            arr_max = max(arr.max(), 1)
            normalized = (arr / arr_max * (2**n_bits - 1)).astype(np.int64)

        # Interleave bits (Morton encoding approximation)
        for bit in range(n_bits):
            codes |= ((normalized >> bit) & 1).astype(np.int64) << (bit * len(z_cols) + bit_pos)

    return codes


def run_zorder_baseline(
    z_cols: list[str] = None,
    workload_path: str = None,
) -> dict:
    """Apply Z-order sorting and measure cost reduction."""
    z_cols = z_cols or ["sale_year", "category", "region"]
    workload_path = workload_path or str(DATA_DIR / "workload_log.json")

    print(f"[zorder] Z-ordering on columns: {z_cols}")

    # Read flat parquet
    flat_path = DATA_DIR / "sales_flat.parquet"
    table = pq.read_table(str(flat_path))

    # Compute Z-order keys
    t0 = time.perf_counter()
    print("[zorder] Computing Morton codes...")
    z_keys = z_order_sort_key(table, z_cols)

    # Sort table by Z-order keys
    sort_indices = np.argsort(z_keys)
    sorted_table = table.take(sort_indices.tolist())

    # Write sorted Parquet (chunked into ~128MB row groups)
    out_path = DATA_DIR / "baseline_zorder"
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir()

    pq.write_table(
        sorted_table,
        str(out_path / "data_zorder.parquet"),
        compression="snappy",
        row_group_size=200_000,  # ~200k rows per row group for good skipping
    )
    elapsed = time.perf_counter() - t0
    print(f"[zorder] Z-order completed in {elapsed:.2f}s")

    # Measure cost
    cost_model = CostModel()
    flat_cost = cost_model.measure(str(flat_path))
    zorder_cost = cost_model.measure(str(out_path / "data_zorder.parquet"))

    improvement = (
        (flat_cost["total_bytes_scanned"] - zorder_cost["total_bytes_scanned"])
        / max(flat_cost["total_bytes_scanned"], 1.0)
    )

    n_files = len(list(out_path.rglob("*.parquet")))
    total_bytes = sum(f.stat().st_size for f in out_path.rglob("*.parquet"))

    return {
        "method": "z_order",
        "z_cols": z_cols,
        "n_files": n_files,
        "total_bytes_mb": total_bytes / 1e6,
        "elapsed_s": elapsed,
        "flat_bytes_scanned_mb": flat_cost["total_bytes_scanned"] / 1e6,
        "zorder_bytes_scanned_mb": zorder_cost["total_bytes_scanned"] / 1e6,
        "bytes_improvement_pct": improvement * 100,
        "flat_avg_latency_ms": flat_cost["avg_latency_ms"],
        "zorder_avg_latency_ms": zorder_cost["avg_latency_ms"],
    }


if __name__ == "__main__":
    flat = DATA_DIR / "sales_flat.parquet"
    if not flat.exists():
        print("Run workload/generator.py first.")
        sys.exit(1)

    results = run_zorder_baseline()
    print("\n=== Z-Order Baseline Results ===")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
