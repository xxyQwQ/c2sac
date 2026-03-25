<h1 align="center">Cross-task Conservative Soft Actor-Critic (C2SAC)</h1>
<p align="center">Final project refactor for AIST5030 Generative Artificial Intelligence, Spring 2026, CUHK</p>

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

## Training

Everything now runs through the shared trainer with the shared cross-task interface:

```bash
python trainer.py agent=bc setting.dataset_name=medium
python trainer.py agent=gail setting.dataset_name=replay
python trainer.py agent=bcq setting.dataset_name=medium
python trainer.py agent=cql setting.dataset_name=replay
python trainer.py agent=c2sac setting.dataset_name=medium
```

## Logging

Training logs are written to each checkpoint directory. Optional Weights & Biases logging can be enabled with:

```bash
python trainer.py agent=cql logging.use_wandb=true
```

The default W&B entity is `VisualCamp`. You can still override it at launch time with `logging.entity=...` if needed.

## Experiments

Batch experiment commands live in [`scripts/experiment.py`](./scripts/experiment.py). Checkpoints are stored under `checkpoints/`, and the notebook in [`scripts/plot.ipynb`](./scripts/plot.ipynb) can still be used for result visualization after adapting plots to the new agent names.
