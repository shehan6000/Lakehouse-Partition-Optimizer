"""
agent/evaluate.py
-----------------
Evaluation harness: loads a trained PPO model and runs it against
the test workload, reporting scan cost reduction vs baselines.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from environment.partition_env import PartitionEnv
from environment.cost_model import CostModel
from workload.encoder import PARTITION_KEYS, GRANULARITIES, ALL_COLUMNS

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "agent" / "saved_models" / "best" / "best_model"


def evaluate_agent(model_path: str = None, n_episodes: int = 10) -> dict:
    model_path = model_path or str(MODEL_PATH)
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    env = PartitionEnv(
        source_parquet=str(ROOT / "data" / "sales_flat.parquet"),
        workload_path=str(ROOT / "data" / "workload_log.json"),
        max_steps=20,
    )

    episode_rewards = []
    best_actions = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0
        baseline = info["baseline_cost"]
        done = False
        ep_best_action = None
        ep_best_reward = -np.inf

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

            if reward > ep_best_reward:
                ep_best_reward = reward
                ep_best_action = info["action"]

        episode_rewards.append(ep_reward)
        best_actions.append(ep_best_action)
        print(f"  Episode {ep+1:2d}: reward={ep_reward:.3f} | best_action={ep_best_action}")

    results = {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "best_actions": best_actions,
    }
    print(f"\nMean reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    return results


def decode_action(action: np.ndarray) -> dict:
    pk_idx, gran_idx, sort_idx = int(action[0]), int(action[1]), int(action[2])
    return {
        "partition_key": PARTITION_KEYS[pk_idx],
        "granularity": GRANULARITIES[gran_idx],
        "sort_col": ALL_COLUMNS[sort_idx] if sort_idx < len(ALL_COLUMNS) else None,
    }


if __name__ == "__main__":
    if not MODEL_PATH.with_suffix(".zip").exists():
        print(f"No model found at {MODEL_PATH}.zip")
        print("Run agent/train.py first.")
        sys.exit(1)
    evaluate_agent()
