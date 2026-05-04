# 🦆 Lakehouse Partition Optimizer

An RL-based system that learns optimal Parquet partitioning strategies for lakehouse query performance.

## Architecture

```
Query Workload → Workload Logger → State Encoder → RL Agent (PPO)
                                                         ↓
                                              Gymnasium Environment
                                                         ↓
                                              Re-partitioner (PyArrow)
                                                         ↓
                                              DuckDB Cost Measurement
```

## Stack

| Layer | Tool |
|---|---|
| Query Engine | DuckDB |
| Data Format | Apache Parquet (PyArrow) |
| Table Format | delta-rs |
| RL Framework | Stable-Baselines3 (PPO) |
| RL Environment | Gymnasium |
| Experiment Tracking | MLflow |
| Workload Generator | TPC-DS style (custom Python) |
| Orchestration | APScheduler |
| Dashboard | Streamlit |

## Quick Start

```bash
pip install -r requirements.txt

# Phase 1: Generate data + start workload logging
python workload/generator.py

# Phase 2: Test the Gym environment
python environment/partition_env.py

# Phase 3: Train the PPO agent
python agent/train.py

# Phase 4: Run benchmarks vs baselines
python benchmarks/run_benchmark.py

# Optional: Launch Streamlit dashboard
streamlit run dashboard/app.py
