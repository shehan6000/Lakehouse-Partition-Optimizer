"""
orchestration/scheduler.py
---------------------------
APScheduler-based periodic re-partition trigger.
Monitors workload drift and triggers the RL agent to re-optimize
when performance degrades beyond a threshold.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
import time
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from environment.cost_model import CostModel
from workload.logger import WorkloadLogger

DATA_DIR = Path(__file__).parent.parent / "data"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [scheduler] %(message)s")
logger = logging.getLogger(__name__)

# Threshold: if scan cost increases by >20% vs last checkpoint, trigger re-optimization
DRIFT_THRESHOLD = 0.20


class PartitionScheduler:
    """
    Periodic job that:
    1. Measures current scan cost on recent workload
    2. Compares to last checkpoint cost
    3. If degraded beyond threshold, triggers RL agent re-optimization
    """

    def __init__(self):
        self.cost_model = CostModel()
        self.last_cost: float = None
        self.current_parquet_path = str(DATA_DIR / "sales_flat.parquet")
        self.scheduler = BackgroundScheduler()
        self.check_count = 0

    def check_and_optimize(self):
        """Main job: check for workload drift and re-optimize if needed."""
        self.check_count += 1
        logger.info(f"Check #{self.check_count}: measuring current scan cost...")

        try:
            cost = self.cost_model.measure(self.current_parquet_path)
            current_cost = cost["total_bytes_scanned"]

            if self.last_cost is None:
                self.last_cost = current_cost
                logger.info(f"Baseline cost set: {current_cost/1e6:.1f} MB")
                return

            drift = (current_cost - self.last_cost) / max(self.last_cost, 1.0)
            logger.info(f"Cost drift: {drift*100:.1f}% (threshold: {DRIFT_THRESHOLD*100:.0f}%)")

            if drift > DRIFT_THRESHOLD:
                logger.warning(f"Drift {drift*100:.1f}% exceeds threshold! Triggering re-optimization...")
                self._trigger_reoptimization()
                self.last_cost = current_cost
            else:
                logger.info("Cost within acceptable range. No re-optimization needed.")

        except Exception as e:
            logger.error(f"Check failed: {e}")

    def _trigger_reoptimization(self):
        """Trigger the RL agent to find a new optimal partition scheme."""
        logger.info("Loading RL model for re-optimization...")

        model_path = Path(__file__).parent.parent / "agent" / "saved_models" / "best" / "best_model.zip"
        if not model_path.exists():
            logger.warning("No trained model found. Run agent/train.py first.")
            return

        try:
            from stable_baselines3 import PPO
            from environment.partition_env import PartitionEnv
            import numpy as np

            model = PPO.load(str(model_path.with_suffix("")))
            env = PartitionEnv(
                source_parquet=self.current_parquet_path,
                workload_path=str(DATA_DIR / "workload_log.json"),
                max_steps=10,
            )

            obs, _ = env.reset()
            best_reward = -np.inf
            best_info = None

            for _ in range(10):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                if reward > best_reward:
                    best_reward = reward
                    best_info = info
                if terminated or truncated:
                    break

            if best_info:
                logger.info(
                    f"Re-optimization complete. Best action: {best_info['action']} "
                    f"(reward={best_reward:.3f})"
                )

                # Log the re-optimization event
                event = {
                    "timestamp": datetime.now().isoformat(),
                    "trigger": "drift",
                    "best_reward": best_reward,
                    "best_action": best_info["action"],
                }
                events_path = DATA_DIR / "scheduler_events.json"
                events = []
                if events_path.exists():
                    with open(events_path) as f:
                        events = json.load(f)
                events.append(event)
                with open(events_path, "w") as f:
                    json.dump(events, f, indent=2)

        except Exception as e:
            logger.error(f"Re-optimization failed: {e}")

    def start(self, interval_minutes: int = 60):
        """Start the scheduler with given check interval."""
        self.scheduler.add_job(
            self.check_and_optimize,
            trigger=IntervalTrigger(minutes=interval_minutes),
            id="partition_check",
            name="Partition performance check",
            replace_existing=True,
        )
        self.scheduler.start()
        logger.info(f"Scheduler started. Checking every {interval_minutes} minutes.")

        try:
            while True:
                time.sleep(10)
        except (KeyboardInterrupt, SystemExit):
            self.scheduler.shutdown()
            logger.info("Scheduler stopped.")

    def run_once(self):
        """Run a single check (useful for testing)."""
        self.check_and_optimize()


if __name__ == "__main__":
    flat = DATA_DIR / "sales_flat.parquet"
    if not flat.exists():
        print("Run workload/generator.py first.")
        sys.exit(1)

    scheduler = PartitionScheduler()
    print("Running single check (use --daemon for continuous mode)...")
    scheduler.run_once()

    if "--daemon" in sys.argv:
        interval = int(sys.argv[sys.argv.index("--daemon") + 1]) if "--daemon" in sys.argv and len(sys.argv) > sys.argv.index("--daemon") + 1 else 60
        scheduler.start(interval_minutes=interval)
