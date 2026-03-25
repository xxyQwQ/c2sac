<h1 align="center">Cross-Task Conservative Soft Actor-Critic (C2SAC)</h1>
<p align="center">Project of IERG5350 Reinforcement Learning, Spring 2026, CUHK</p>

This project studies offline reinforcement learning on multiple walker subtasks. The refactor unifies the original training scripts into a single configurable trainer, updates the project naming to **C2SAC**, uses the two walker subtasks `walk` and `run`, and adds a broader baseline suite for course experiments.

## Requirements

Create a Python environment for this project with Python 3.10 and the pinned dependencies in [`requirements.txt`](./requirements.txt). The current setup uses a newer PyTorch build with CUDA 12.8 while keeping the rest of the compatibility-sensitive RL stack aligned with this repo.

```bash
conda create -n c2sac python=3.10
conda activate c2sac
pip install -r requirements.txt
```

## Datasets

The trainer expects datasets under [`datasets`](./datasets) with the following names:

- `walker-walk-medium`
- `walker-walk-replay`
- `walker-run-medium`
- `walker-run-replay`

## Agents

The refactored project supports:

- `bc`: behavior cloning
- `gail`: generative adversarial imitation learning
- `bcq`: batch-constrained Q-learning
- `cql`: conservative Q-learning
- `c2sac`: cross-task conservative soft actor-critic

## Experiments

Everything now runs through the shared trainer with the shared cross-task interface:

```bash
python trainer.py agent={agent_name} setting.dataset_name={dataset_name}
```

You can directly run all the experiments with the provided script:

```bash
python scripts/experiment.py
```
