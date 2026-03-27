<h1 align="center">Cross-Task Conservative Soft Actor-Critic (C2SAC)</h1>
<p align="center">Project of IERG5350 Reinforcement Learning, Spring 2026, CUHK</p>

This project focuses on the challenge of multi-task offline reinforcement learning. We propose a novel method named **C2SAC** inspired by conservative Q-learning (CQL) and conservative data sharing (CDS). We evaluate C2SAC in the walker environment, jointly learning to walk and run from offline trajectories. We compare C2SAC with multiple baselines, including behavioral cloning (BC), generative adversarial imitation learning (GAIL), batch-constrained Q-learning (BCQ), and conservative Q-learning (CQL). The results demonstrate that C2SAC effectively leverages cross-task data to achieve superior performance in both tasks.

## Requirements

Create a conda environment:

```bash
conda create -n c2sac python=3.10
conda activate c2sac
```

Install a compatible version of PyTorch:

```bash
pip install torch torchvision
```

Reinstall some tools to avoid conflicts:

```bash
pip install setuptools==65.5.0 pip==21.0
pip install wheel==0.38.0
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Datasets

The datasets are collected from TD-3 agents trained on the `walker-walk` and `walker-run` tasks, containing two subsets of trajectories. The `medium` subset is sampled from a policy with medium performance and the `replay` subset is sampled from the replay buffer during training.

## Agents

There are five agents implemented in this project, including behavioral cloning (BC), generative adversarial imitation learning (GAIL), batch-constrained Q-learning (BCQ), conservative Q-learning (CQL), and our proposed method C2SAC.

## Experiments

Run the following command to train a specific agent on a specific dataset:

```bash
python trainer.py agent={agent_name} setting.dataset_name={dataset_name}
# agent_name: bc, gail, bcq, cql, c2sac
# dataset_name: medium, replay
```

You can directly run all the experiments with the provided script:

```bash
python scripts/experiment.py
```
