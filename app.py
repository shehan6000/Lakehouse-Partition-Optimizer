"""
tracking/mlflow_logger.py
--------------------------
Centralized MLflow logging utilities for all experiment runs.
"""

import mlflow
import numpy as np
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).parent.parent
MLRUNS_DIR = ROOT / "mlruns"


def setup_mlflow(experiment_name: str = "partition-optimizer", tracking_uri: str = None):
    if tracking_uri:
        uri = tracking_uri
    else:
        uri = "sqlite:///" + str(ROOT / "mlflow.db")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    return mlflow.get_experiment_by_name(experiment_name)


def log_baseline_run(method: str, results: dict):
    """Log a baseline (Hive/Z-order) run to MLflow."""
    with mlflow.start_run(run_name=f"baseline_{method}"):
        mlflow.log_param("method", method)
        for k, v in results.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)
            elif isinstance(v, list):
                mlflow.log_param(k, str(v))
            else:
                mlflow.log_param(k, str(v))


def log_rl_training_step(
    step: int,
    mean_reward: float,
    std_reward: float,
    best_reward: float,
    partition_key: Optional[str] = None,
    granularity: Optional[str] = None,
):
    mlflow.log_metrics(
        {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "best_reward": best_reward,
        },
        step=step,
    )
    if partition_key:
        mlflow.log_param("best_partition_key", partition_key)
    if granularity:
        mlflow.log_param("best_granularity", granularity)


def log_benchmark_comparison(comparison: dict):
    """Log full benchmark comparison table to MLflow."""
    with mlflow.start_run(run_name="benchmark_comparison"):
        for method, metrics in comparison.items():
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"{method}_{k}", v)


class EpisodeLogger:
    """Callback-friendly logger for per-episode RL stats."""

    def __init__(self, run_name: str = "ppo-training"):
        self.run_name = run_name
        self._rewards = []
        self._run = None

    def start(self):
        self._run = mlflow.start_run(run_name=self.run_name)

    def log_episode(self, episode: int, reward: float, info: dict):
        self._rewards.append(reward)
        mlflow.log_metric("episode_reward", reward, step=episode)
        if info.get("bytes_improvement_pct") is not None:
            mlflow.log_metric("bytes_improvement_pct", info["bytes_improvement_pct"], step=episode)

    def finish(self):
        if self._rewards:
            mlflow.log_metrics(
                {
                    "final_mean_reward": float(np.mean(self._rewards)),
                    "final_max_reward": float(np.max(self._rewards)),
                }
            )
        if self._run:
            mlflow.end_run()
