"""
environment/repartitioner.py
-----------------------------
Physically rewrites Parquet files with a new partition scheme using PyArrow.
Supports: value partitioning, hash bucketing, temporal granularity partitioning.
"""

import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

DATA_DIR = Path(__file__).parent.parent / "data"
WORKING_DIR = DATA_DIR / "working"
WORKING_DIR.mkdir(exist_ok=True)


class Repartitioner:
    """
    Applies a partition action to a Parquet dataset.
    Reads from source, rewrites to a new directory, returns the cost (time + size).
    """

    def __init__(self, source_path: Optional[str] = None):
        self.source_path = source_path or str(DATA_DIR / "sales_flat.parquet")

    def apply(
        self,
        partition_key: str,
        granularity: str,
        sort_cols: Optional[list[str]] = None,
        output_dir: Optional[str] = None,
    ) -> dict:
        """
        Re-partition the dataset and return stats.

        Returns:
            dict with keys: output_path, n_files, total_bytes, elapsed_s
        """
        t0 = time.perf_counter()

        output_dir = output_dir or str(WORKING_DIR / f"part_{partition_key}_{granularity}")
        out_path = Path(output_dir)
        if out_path.exists():
            shutil.rmtree(out_path)
        out_path.mkdir(parents=True)

        # Read source
        table = self._read_source()

        # Apply granularity transform
        table = self._apply_granularity(table, partition_key, granularity)

        # Sort within partitions
        if sort_cols:
            valid_sort = [c for c in sort_cols if c in table.schema.names]
            if valid_sort:
                table = table.sort_by([(c, "ascending") for c in valid_sort])

        # Write partitioned dataset
        if partition_key == "none" or partition_key not in table.schema.names:
            # Single flat file
            pq.write_table(table, str(out_path / "data.parquet"), compression="snappy")
        else:
            derived_col = self._get_partition_col_name(partition_key, granularity)
            pq.write_to_dataset(
                table,
                root_path=str(out_path),
                partition_cols=[derived_col],
                compression="snappy",
            )

        elapsed = time.perf_counter() - t0

        # Gather stats
        parquet_files = list(out_path.rglob("*.parquet"))
        total_bytes = sum(f.stat().st_size for f in parquet_files)
        n_files = len(parquet_files)

        return {
            "output_path": str(out_path),
            "n_files": n_files,
            "total_bytes": total_bytes,
            "elapsed_s": elapsed,
            "avg_file_bytes": total_bytes / max(n_files, 1),
        }

    def _read_source(self) -> pa.Table:
        p = Path(self.source_path)
        if p.is_dir():
            return pq.read_table(str(p), use_threads=True)
        return pq.read_table(str(p))

    def _apply_granularity(self, table: pa.Table, partition_key: str, granularity: str) -> pa.Table:
        """Add or transform the partition column according to granularity."""
        if partition_key == "none":
            return table

        derived_col = self._get_partition_col_name(partition_key, granularity)

        if granularity in ("year", "month", "day"):
            # Temporal: re-use existing columns
            if partition_key == "sale_date" and granularity == "year":
                col = pc.utf8_slice_codeunits(table.column("sale_date"), 0, 4)
                table = table.append_column(derived_col, col)
            elif partition_key == "sale_year":
                table = table.append_column(derived_col, table.column("sale_year").cast(pa.string()))
            elif partition_key == "sale_month":
                table = table.append_column(derived_col, table.column("sale_month").cast(pa.string()))
            else:
                # Fall back to value partitioning
                table = table.append_column(derived_col, table.column(partition_key))

        elif granularity.startswith("hash_"):
            n_buckets = int(granularity.split("_")[1])
            raw = table.column(partition_key)
            # Hash into buckets
            hashed = pa.array(
                [str(hash(str(v.as_py())) % n_buckets) for v in raw],
                type=pa.string(),
            )
            table = table.append_column(derived_col, hashed)

        elif granularity == "value":
            # Partition by distinct value (good for low-cardinality categoricals)
            table = table.append_column(derived_col, table.column(partition_key))

        return table

    def _get_partition_col_name(self, partition_key: str, granularity: str) -> str:
        return f"_part_{partition_key}_{granularity}"

    def estimate_repartition_cost(self, total_bytes: float) -> float:
        """
        Estimate the I/O cost of re-partitioning (bytes written ≈ bytes read).
        Returns cost in 'bytes' units for reward normalization.
        """
        return total_bytes  # re-partition writes ~same bytes as it reads


if __name__ == "__main__":
    r = Repartitioner()
    print("Testing re-partitioner...")

    flat = DATA_DIR / "sales_flat.parquet"
    if not flat.exists():
        print("Run workload/generator.py first.")
    else:
        for pk, gran in [("sale_year", "year"), ("category", "value"), ("store_id", "hash_16")]:
            result = r.apply(pk, gran)
            print(f"  {pk}/{gran}: {result['n_files']} files, {result['total_bytes']/1e6:.1f} MB, {result['elapsed_s']:.2f}s")
