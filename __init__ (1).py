"""
environment/partition_env.py
-----------------------------
Gymnasium custom environment for RL-based partition optimization.

Observation: state vector (partition scheme + workload stats)
Action:      (partition_key_idx, granularity_idx, sort_col_idx)
Reward:      normalized bytes-scanned reduction minus repartition cost
"""

import json
import shutil
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Import local modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from workload.encoder import (
    StateEncoder, PartitionScheme,
    PARTITION_KEYS, GRANULARITIES, ALL_COLUMNS,
)
from workload.logger import WorkloadLogger
from environment.repartitioner import Repartitioner
from environment.cost_model import CostModel
from environment.simulated_cost_model import SimulatedCostModel

DATA_DIR = Path(__file__).parent.parent / "data"
WORKING_DIR = DATA_DIR / "working"


class PartitionEnv(gym.Env):
    """
    RL Environment for lakehouse partition optimization.

    The agent selects a (partition_key, granularity, sort_col) triple.
    The environment re-partitions the dataset, runs probe queries, and
    returns a reward proportional to scan cost reduction.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        source_parquet: Optional[str] = None,
        workload_path: Optional[str] = None,
        max_steps: int = 20,
        repartition_cost_weight: float = 0.1,
        fast_mode: bool = True,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.fast_mode = fast_mode
        self.source_parquet = source_parquet or str(DATA_DIR / "sales_flat.parquet")
        self.workload_path = workload_path or str(DATA_DIR / "workload_log.json")
        self.max_steps = max_steps
        self.repartition_cost_weight = repartition_cost_weight
        self.render_mode = render_mode

        # Load workload
        try:
            with open(self.workload_path) as f:
                self.workload = json.load(f)
        except FileNotFoundError:
            self.workload = []

        # Components
        self.encoder = StateEncoder()
        self.repartitioner = Repartitioner(self.source_parquet)
        self.cost_model = SimulatedCostModel() if fast_mode else CostModel()
        self.logger = WorkloadLogger(window=50)

        # Action space: Discrete (n_partition_keys * n_granularities * n_sort_cols)
        # We encode as a MultiDiscrete for interpretability
        n_pk = len(PARTITION_KEYS)
        n_gran = len(GRANULARITIES)
        n_sort = len(ALL_COLUMNS) + 1  # +1 for "no sort"
        self.action_space = spaces.MultiDiscrete([n_pk, n_gran, n_sort])

        # Observation space
        obs_dim = self.encoder.observation_space_dim
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # State
        self._current_scheme: Optional[PartitionScheme] = None
        self._current_path: Optional[str] = None
        self._baseline_cost: Optional[float] = None
        self._step_count = 0
        self._episode_rewards: list[float] = []
        self._best_reward = -np.inf
        self._best_scheme: Optional[PartitionScheme] = None

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._episode_rewards = []

        # Start from flat (no partition) scheme
        self._current_scheme = PartitionScheme(
            partition_key="none",
            granularity="year",
            sort_cols=[],
            n_files=1,
            total_bytes=self._get_source_bytes(),
        )
        self._current_path = self.source_parquet

        # Measure baseline cost
        if self.fast_mode:
            self._baseline_cost = self.cost_model.baseline_cost()
        else:
            baseline = self.cost_model.measure(self.source_parquet)
            self._baseline_cost = baseline["total_bytes_scanned"]

        # Warm up workload logger
        if self.workload:
            self.logger.load_table(self.source_parquet)
            for entry in self.workload[:10]:
                try:
                    self.logger.execute(
                        entry["sql"],
                        predicates=entry.get("predicates"),
                        access_cols=entry.get("access_cols"),
                    )
                except Exception:
                    pass

        obs = self._get_observation()
        info = {"baseline_cost": self._baseline_cost}
        return obs, info

    def step(self, action):
        pk_idx, gran_idx, sort_idx = int(action[0]), int(action[1]), int(action[2])
        partition_key = PARTITION_KEYS[pk_idx]
        granularity = GRANULARITIES[gran_idx]
        sort_col = ALL_COLUMNS[sort_idx] if sort_idx < len(ALL_COLUMNS) else None

        self._step_count += 1

        # Apply re-partition (fast: simulated; slow: real I/O)
        if self.fast_mode:
            stats = self.logger.get_stats()
            result = self.cost_model.measure_simulated(
                partition_key=partition_key,
                granularity=granularity,
                workload_predicate_freq=stats.predicate_freq,
            )
            new_bytes = result["total_bytes_scanned"]
            new_path = self._current_path  # no physical change
        else:
            result = self.repartitioner.apply(
                partition_key=partition_key,
                granularity=granularity,
                sort_cols=[sort_col] if sort_col else [],
                output_dir=str(WORKING_DIR / f"step_{self._step_count}"),
            )
            new_path = result["output_path"]
            new_cost = self.cost_model.measure(new_path)
            new_bytes = new_cost["total_bytes_scanned"]

        # Compute reward
        repartition_bytes = result.get("total_bytes", self._baseline_cost) if not self.fast_mode else self._baseline_cost * 0.05
        reward = self._compute_reward(
            bytes_before=self._baseline_cost,
            bytes_after=new_bytes,
            repartition_bytes=repartition_bytes,
        )

        # Update state
        self._current_scheme = PartitionScheme(
            partition_key=partition_key,
            granularity=granularity,
            sort_cols=[sort_col] if sort_col else [],
            n_files=result.get("n_files", 1),
            total_bytes=result.get("total_bytes", self._baseline_cost),
            avg_file_bytes=result.get("avg_file_bytes", self._baseline_cost),
        )
        self._current_path = new_path

        if reward > self._best_reward:
            self._best_reward = reward
            self._best_scheme = self._current_scheme

        self._episode_rewards.append(reward)

        terminated = self._step_count >= self.max_steps
        truncated = False

        obs = self._get_observation()
        info = {
            "action": {"partition_key": partition_key, "granularity": granularity, "sort_col": sort_col},
            "bytes_before": self._baseline_cost,
            "bytes_after": new_bytes,
            "reward": reward,
            "step": self._step_count,
            "best_reward": self._best_reward,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        scheme = self._current_scheme
        print(
            f"Step {self._step_count:3d} | "
            f"partition={scheme.partition_key}/{scheme.granularity} | "
            f"sort={scheme.sort_cols} | "
            f"files={scheme.n_files} | "
            f"reward={self._episode_rewards[-1] if self._episode_rewards else 0:.3f}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        bytes_before: float,
        bytes_after: float,
        repartition_bytes: float,
    ) -> float:
        """
        reward = (bytes_before - bytes_after) / bytes_before
                 - repartition_cost_weight * (repartition_bytes / bytes_before)

        Positive: scan savings exceed repartition I/O cost.
        Negative: repartition was too expensive or made things worse.
        """
        if bytes_before <= 0:
            return 0.0

        scan_saving = (bytes_before - bytes_after) / bytes_before
        repartition_penalty = self.repartition_cost_weight * (repartition_bytes / bytes_before)
        return float(scan_saving - repartition_penalty)

    def _get_observation(self) -> np.ndarray:
        stats = self.logger.get_stats()
        return self.encoder.encode(
            scheme=self._current_scheme,
            col_access_freq=stats.col_access_freq,
            predicate_freq=stats.predicate_freq,
            avg_latency_ms=stats.avg_latency_ms,
        )

    def _get_source_bytes(self) -> float:
        p = Path(self.source_parquet)
        if p.is_file():
            return float(p.stat().st_size)
        return sum(f.stat().st_size for f in p.rglob("*.parquet"))

    def close(self):
        # Cleanup working directory
        if WORKING_DIR.exists():
            shutil.rmtree(WORKING_DIR, ignore_errors=True)


if __name__ == "__main__":
    import sys

    flat = DATA_DIR / "sales_flat.parquet"
    if not flat.exists():
        print("Run workload/generator.py first to create the dataset.")
        sys.exit(1)

    print("Testing PartitionEnv...")
    env = PartitionEnv(render_mode="human", max_steps=5)
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Baseline cost: {info['baseline_cost']/1e6:.1f} MB")
    print(f"Action space: {env.action_space}")

    total_reward = 0
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            break

    print(f"\nTotal reward over 5 steps: {total_reward:.3f}")
    print(f"Best scheme found: {env._best_scheme}")
    env.close()
