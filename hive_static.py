# agent/config.yaml
# PPO Hyperparameters for partition optimization agent

ppo:
  learning_rate: 3.0e-4
  n_steps: 128          # reduced: fewer steps per rollout = faster iteration
  batch_size: 64
  n_epochs: 5           # reduced from 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  clip_range_vf: null
  normalize_advantage: true
  ent_coef: 0.02        # slightly higher entropy for faster exploration
  vf_coef: 0.5
  max_grad_norm: 0.5

  policy: "MlpPolicy"
  policy_kwargs:
    net_arch:
      - 128
      - 128           # smaller network: faster forward pass

  total_timesteps: 10000  # sufficient for concept validation
  n_envs: 1

env:
  max_steps: 10         # reduced from 20: agent decides faster
  repartition_cost_weight: 0.1
  fast_mode: true       # KEY: use simulated cost model, skip physical rewrite

eval:
  n_eval_episodes: 3    # reduced from 10
  eval_freq: 2000       # evaluate less often
  deterministic: true

mlflow:
  experiment_name: "partition-optimizer"
  run_name: "ppo-fast"
  tracking_uri: "sqlite:///mlflow.db"

paths:
  source_parquet: "data/sales_flat.parquet"
  workload_log: "data/workload_log.json"
  model_save: "agent/saved_models/ppo_partition"
  log_dir: "agent/logs"
