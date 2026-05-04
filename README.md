


**PartitionRL** is a reinforcement learning system that learns optimal Apache Parquet partitioning strategies to minimize query scan costs in a lakehouse environment. Rather than relying on static heuristics like Hive-style partitioning or Z-ordering, a PPO agent continuously adapts to the actual query workload — re-partitioning data when the expected reduction in bytes scanned outweighs the cost of re-writing files.

The agent observes a state encoding of the current workload (query predicates, column access patterns, data distribution) and selects from a discrete action space of partitioning schemes. Each action is evaluated by physically re-partitioning the dataset with PyArrow, executing representative queries via DuckDB, and measuring bytes scanned before and after. This scan-reduction ratio forms the reward signal, with a penalty applied when re-partition I/O cost exceeds the savings.

The environment is implemented as a Gymnasium-compatible interface, making it straightforward to swap in alternative RL algorithms. Training runs are tracked with MLflow, and a Streamlit dashboard visualises agent behaviour, reward curves, and benchmark comparisons against static baselines.

---

And a compact one-liner for the repo description field:

> PPO agent that learns adaptive Parquet partitioning strategies to minimise DuckDB query scan costs in a lakehouse.The long paragraph goes in the body of your README (below the existing architecture section), and the one-liner goes in the **About** field on the GitHub repo page (the small description shown on your profile and at the top of the repo). This project reads really well as a PhD portfolio piece — the closed-loop feedback between the cost model and the agent is a clean research story.
