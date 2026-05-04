"""
agent/train.py
--------------
PPO training loop using Stable-Baselines3.
Tracks experiments with MLflow.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
import yaml
import numpy as np
import mlflow
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from environment.partition_env import PartitionEnv

ROOT = Path(__file__).parent.parent
CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


class MLflowCallback(BaseCallback):
    """Log SB3 training metrics to MLflow every N steps."""

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            # Log rollout stats
            if len(self.model.ep_info_buffer) > 0:
                ep_rewards = [ep["r"] for ep in self.model.ep_info_buffer]
                ep_lengths = [ep["l"] for ep in self.model.ep_info_buffer]
                mlflow.log_metrics(
                    {
                        "mean_ep_reward": float(np.mean(ep_rewards)),
                        "mean_ep_length": float(np.mean(ep_lengths)),
                    },
                    step=self.num_timesteps,
                )
        return True


def make_env(cfg: dict, eval_mode: bool = False) -> PartitionEnv:
    env = PartitionEnv(
        source_parquet=str(ROOT / cfg["paths"]["source_parquet"]),
        workload_path=str(ROOT / cfg["paths"]["workload_log"]),
        max_steps=cfg["env"]["max_steps"],
        repartition_cost_weight=cfg["env"]["repartition_cost_weight"],
        fast_mode=cfg["env"].get("fast_mode", True),
    )
    return Monitor(env)


def train(cfg: dict = None):
    if cfg is None:
        cfg = load_config()

    ppo_cfg = cfg["ppo"]
    mlf_cfg = cfg["mlflow"]

    # Setup save dirs
    model_save_path = ROOT / cfg["paths"]["model_save"]
    log_dir = ROOT / cfg["paths"]["log_dir"]
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # MLflow tracking URI: sqlite:/// URIs are used as-is; plain paths get file:// prefix
    tracking_uri = mlf_cfg["tracking_uri"]
    if "://" not in tracking_uri:
        tracking_uri = (ROOT / tracking_uri).resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlf_cfg["experiment_name"])

    warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")
    with mlflow.start_run(run_name=mlf_cfg["run_name"]):
        # Log all hyperparameters
        mlflow.log_params({
            "algorithm": "PPO",
            "learning_rate": ppo_cfg["learning_rate"],
            "n_steps": ppo_cfg["n_steps"],
            "batch_size": ppo_cfg["batch_size"],
            "n_epochs": ppo_cfg["n_epochs"],
            "gamma": ppo_cfg["gamma"],
            "ent_coef": ppo_cfg["ent_coef"],
            "total_timesteps": ppo_cfg["total_timesteps"],
            "max_env_steps": cfg["env"]["max_steps"],
            "repartition_cost_weight": cfg["env"]["repartition_cost_weight"],
        })

        # Build envs
        train_env = DummyVecEnv([lambda: make_env(cfg)])
        eval_env = DummyVecEnv([lambda: make_env(cfg, eval_mode=True)])

        # Build PPO model
        policy_kwargs = ppo_cfg.get("policy_kwargs", {})
        model = PPO(
            policy=ppo_cfg["policy"],
            env=train_env,
            learning_rate=ppo_cfg["learning_rate"],
            n_steps=ppo_cfg["n_steps"],
            batch_size=ppo_cfg["batch_size"],
            n_epochs=ppo_cfg["n_epochs"],
            gamma=ppo_cfg["gamma"],
            gae_lambda=ppo_cfg["gae_lambda"],
            clip_range=ppo_cfg["clip_range"],
            ent_coef=ppo_cfg["ent_coef"],
            vf_coef=ppo_cfg["vf_coef"],
            max_grad_norm=ppo_cfg["max_grad_norm"],
            policy_kwargs=policy_kwargs,
            verbose=1,
        )

        # Callbacks
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(model_save_path.parent / "best"),
            log_path=str(log_dir / "eval"),
            eval_freq=cfg["eval"]["eval_freq"],
            n_eval_episodes=cfg["eval"]["n_eval_episodes"],
            deterministic=cfg["eval"]["deterministic"],
            verbose=1,
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=10_000,
            save_path=str(model_save_path.parent / "checkpoints"),
            name_prefix="ppo_partition",
        )
        mlflow_callback = MLflowCallback(log_freq=1000)

        # Train
        print(f"\n{'='*60}")
        print(f"  Training PPO for {ppo_cfg['total_timesteps']:,} timesteps")
        print(f"{'='*60}\n")

        model.learn(
            total_timesteps=ppo_cfg["total_timesteps"],
            callback=[eval_callback, checkpoint_callback, mlflow_callback],
            progress_bar=True,
        )

        # Save final model
        model.save(str(model_save_path))
        mlflow.log_artifact(str(model_save_path) + ".zip")
        print(f"\nModel saved to {model_save_path}.zip")

    return model


if __name__ == "__main__":
    data_check = ROOT / "data" / "sales_flat.parquet"
    if not data_check.exists():
        print("Run workload/generator.py first.")
        sys.exit(1)
    train()
