"""
tests/test_core.py
------------------
Unit tests for core components.
Run with: pytest tests/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest


# -------------------------------------------------------------------------
# State Encoder Tests
# -------------------------------------------------------------------------

class TestStateEncoder:
    def test_encode_shape(self):
        from workload.encoder import StateEncoder, PartitionScheme
        encoder = StateEncoder()
        scheme = PartitionScheme(partition_key="sale_year", granularity="year")
        state = encoder.encode(scheme)
        assert state.shape == (encoder.observation_space_dim,)
        assert state.dtype == np.float32

    def test_encode_bounds(self):
        from workload.encoder import StateEncoder, PartitionScheme
        encoder = StateEncoder()
        scheme = PartitionScheme(partition_key="category", granularity="value", n_files=10, total_bytes=1e8)
        col_freq = {"category": 0.9, "revenue": 0.7}
        state = encoder.encode(scheme, col_access_freq=col_freq)
        assert np.all(state >= 0.0)
        assert np.all(state <= 1.0 + 1e-6)

    def test_partition_key_one_hot(self):
        from workload.encoder import StateEncoder, PartitionScheme, PARTITION_KEYS
        encoder = StateEncoder()
        for pk in PARTITION_KEYS:
            scheme = PartitionScheme(partition_key=pk, granularity="year")
            state = encoder.encode(scheme)
            pk_slice = state[:len(PARTITION_KEYS)]
            assert pk_slice.sum() == pytest.approx(1.0, abs=1e-5), f"Failed for {pk}"


# -------------------------------------------------------------------------
# Workload Logger Tests
# -------------------------------------------------------------------------

class TestWorkloadLogger:
    def test_stats_empty(self):
        from workload.logger import WorkloadLogger
        logger = WorkloadLogger()
        stats = logger.get_stats()
        assert stats.n_queries == 0
        assert stats.avg_latency_ms == 0.0

    def test_predicate_inference(self):
        from workload.logger import WorkloadLogger
        logger = WorkloadLogger()
        sql = "SELECT * FROM sales WHERE sale_year = 2022 AND category = 'food'"
        preds = logger._infer_predicates(sql)
        assert "sale_year" in preds
        assert "category" in preds

    def test_access_col_inference(self):
        from workload.logger import WorkloadLogger
        logger = WorkloadLogger()
        sql = "SELECT region, SUM(revenue) FROM sales GROUP BY region"
        cols = logger._infer_access_cols(sql)
        assert "region" in cols
        assert "revenue" in cols

    def test_stats_vector_shape(self):
        from workload.logger import WorkloadLogger
        logger = WorkloadLogger()
        vec = logger.get_stats_vector()
        assert vec.ndim == 1
        assert vec.dtype == np.float32


# -------------------------------------------------------------------------
# Cost Model Tests
# -------------------------------------------------------------------------

class TestCostModel:
    def test_estimate_bytes_fallback(self):
        from environment.cost_model import CostModel
        cm = CostModel()
        # With empty explain text, should return fallback
        result = cm._estimate_bytes.__func__(cm, "")
        assert result > 0

    def test_compare_structure(self):
        """compare() should return expected keys."""
        from environment.cost_model import CostModel
        cm = CostModel()
        # We can't test with real files in unit test, so just check the structure
        # by patching measure
        original_measure = cm.measure

        def mock_measure(path, *args, **kwargs):
            return {"total_bytes_scanned": 1000.0, "total_latency_ms": 100.0, "avg_latency_ms": 50.0, "avg_bytes_per_query": 500.0, "n_queries": 2}

        cm.measure = mock_measure
        result = cm.compare("path_a", "path_b")
        assert "bytes_improvement_pct" in result
        assert "latency_improvement_pct" in result
        assert result["bytes_improvement_pct"] == pytest.approx(0.0, abs=1e-6)
        cm.measure = original_measure


# -------------------------------------------------------------------------
# Repartitioner Tests
# -------------------------------------------------------------------------

class TestRepartitioner:
    def test_get_partition_col_name(self):
        from environment.repartitioner import Repartitioner
        r = Repartitioner()
        name = r._get_partition_col_name("sale_year", "year")
        assert name == "_part_sale_year_year"

    def test_estimate_cost(self):
        from environment.repartitioner import Repartitioner
        r = Repartitioner()
        cost = r.estimate_repartition_cost(1e9)
        assert cost == 1e9


# -------------------------------------------------------------------------
# Action / Partition Space Tests
# -------------------------------------------------------------------------

class TestActionSpace:
    def test_partition_keys_unique(self):
        from workload.encoder import PARTITION_KEYS
        assert len(PARTITION_KEYS) == len(set(PARTITION_KEYS))

    def test_granularities_unique(self):
        from workload.encoder import GRANULARITIES
        assert len(GRANULARITIES) == len(set(GRANULARITIES))

    def test_reward_formula(self):
        """Test the reward formula directly."""
        # 50% bytes saved, 10% repartition cost → reward ≈ 0.49
        bytes_before = 100.0
        bytes_after = 50.0
        repartition_bytes = 100.0
        cost_weight = 0.1
        scan_saving = (bytes_before - bytes_after) / bytes_before
        penalty = cost_weight * (repartition_bytes / bytes_before)
        reward = scan_saving - penalty
        assert reward == pytest.approx(0.4, abs=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
