import os
import sys
import random
import inspect

import hydra
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from utils.env import make
from utils.task import validate_task_names
from utils.torch import convert_batch_to_tensor
from utils.logger import FileLogger, WandbLogger
from utils.dataset import SharingDataset, collate_batch
from agents import BCAgent, GAILAgent, BCQAgent, CQLAgent, C2SACAgent


def _to_plain_config(config):
    return OmegaConf.to_container(config, resolve=True)


def _build_dataset_paths(dataset_root, task_names, dataset_name):
    return [os.path.join(dataset_root, f'walker-{task_name}-{dataset_name}') for task_name in task_names]


def _validate_dataset_paths(dataset_paths):
    missing_paths = [path for path in dataset_paths if not os.path.isdir(path)]
    if missing_paths:
        raise FileNotFoundError(f'Missing dataset directories: {missing_paths}')


def _make_environments(task_names, seed):
    return [make(f'walker_{task_name}', seed=seed) for task_name in task_names]


def _evaluate_policy(device, environment, policy, episodes):
    lengths = []
    returns = []
    with torch.no_grad():
        for _ in tqdm(range(episodes), desc='Evaluating', leave=False):
            episode_length = 0
            episode_return = 0.0
            step = environment.reset()
            while not step.last():
                state = torch.from_numpy(step.observation).to(device)
                action = policy(state).cpu().numpy()
                step = environment.step(action)
                episode_length += 1
                episode_return += step.reward
            lengths.append(episode_length)
            returns.append(episode_return)
    return {
        'Episode Length': lengths,
        'Average Return': returns
    }


def _evaluate_multi_task(device, environment, agent, task_index, episodes):
    task = torch.tensor([task_index], dtype=torch.long).to(device)
    return _evaluate_policy(device, environment, lambda state: agent.take_action(task, state), episodes)


def _supports_dataset_sharing(config):
    return config.agent.name == 'c2sac'


def _summarize_metrics(metrics):
    return ' | '.join(f'{key}: {np.mean(value):7.2f}' for key, value in metrics.items())


def _train_agent_batch(agent, batch, warmup):
    signature = inspect.signature(agent.train_batch)
    if 'warmup' in signature.parameters:
        return agent.train_batch(batch, warmup=warmup)
    return agent.train_batch(batch)


def _make_agent(config):
    name = config.agent.name
    parameter = dict(config.agent.parameter)
    state_dims = config.setting.state_dims
    action_dims = config.setting.action_dims
    num_tasks = len(config.setting.task_names)
    if name == 'bc':
        return BCAgent(num_tasks, state_dims, action_dims, **parameter)
    if name == 'gail':
        return GAILAgent(num_tasks, state_dims, action_dims, **parameter)
    if name == 'bcq':
        return BCQAgent(num_tasks, state_dims, action_dims, **parameter)
    if name == 'cql':
        return CQLAgent(num_tasks, state_dims, action_dims, **parameter)
    if name == 'c2sac':
        return C2SACAgent(num_tasks, state_dims, action_dims, **parameter)
    raise ValueError(f'Unsupported agent: {name}')


def _make_dataset(config):
    task_names = list(config.setting.task_names)
    dataset_paths = _build_dataset_paths(
        config.setting.dataset_root,
        task_names,
        config.setting.dataset_name
    )
    _validate_dataset_paths(dataset_paths)
    return SharingDataset(dataset_paths)


def _set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _create_dataloader(dataset, strategy):
    return DataLoader(
        dataset,
        batch_size=strategy.batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )


def _train_one_epoch(agent, loader, device, epoch, strategy):
    metrics = {}
    warmup = epoch <= strategy.warmup_epochs
    for batch in tqdm(loader, desc='Training', leave=False):
        batch = convert_batch_to_tensor(batch, device)
        feedback = _train_agent_batch(agent, batch, warmup=warmup)
        for key, value in feedback.items():
            metrics.setdefault(key, []).append(value)
    return metrics


def _log_metrics(prefix, metrics, metric_logger, step, normalize_keys=True):
    def _format_key(key):
        return key.lower().replace(' ', '_') if normalize_keys else key

    metric_logger.log(
        {f'{prefix}/{_format_key(key)}': np.mean(value) for key, value in metrics.items()},
        step=step
    )


def _evaluate_tasks(config, strategy, device, agent, environments, metric_logger, epoch):
    scores = []
    for task_index, (task_name, environment) in enumerate(zip(config.setting.task_names, environments)):
        metrics = _evaluate_multi_task(device, environment, agent, task_index, strategy.sample_episodes)
        scores.append(np.mean(metrics['Average Return']))
        print(f'Evaluate | Epoch: {epoch:4d} | Task: walker_{task_name:5} | {_summarize_metrics(metrics)} |')
        _log_metrics(f'eval/{task_name}', metrics, metric_logger, step=epoch)
    return float(np.mean(scores))


def _run_dataset_sharing(dataset, strategy, device, agent, metric_logger, epoch):
    metrics = dataset.update(device, agent, strategy.rank_percentage)
    print(f'Update | Epoch: {epoch:4d} | Dataset Size: {len(dataset):6d} | {_summarize_metrics(metrics)} |')
    _log_metrics('update', metrics, metric_logger, step=epoch)


def _run_final_evaluation(config, strategy, device, agent, environments):
    for task_index, (task_name, environment) in enumerate(zip(config.setting.task_names, environments)):
        metrics = _evaluate_multi_task(device, environment, agent, task_index, strategy.sample_episodes)
        print(f'Test | Task: walker_{task_name:5} | {_summarize_metrics(metrics)} |')


def run(config):
    checkpoint_path = os.path.abspath(str(config.checkpoint))
    os.makedirs(checkpoint_path, exist_ok=True)
    device = torch.device('cuda' if config.device == 'gpu' and torch.cuda.is_available() else 'cpu')
    sys.stdout = FileLogger(os.path.join(checkpoint_path, 'trainer.log'))
    config.checkpoint = checkpoint_path
    config.device = str(device)
    print(OmegaConf.to_yaml(config))

    _set_random_seed(config.seed)
    agent_strategy = config.agent.get('strategy', OmegaConf.create({}))
    strategy = OmegaConf.merge(config.strategy, agent_strategy)
    config.setting.task_names = validate_task_names(config.setting.task_names)
    task_names = config.setting.task_names

    agent = _make_agent(config)
    dataset = _make_dataset(config)
    loader = _create_dataloader(dataset, strategy)
    environments = _make_environments(task_names, config.seed)
    agent.to_device(device)

    metric_logger = WandbLogger(
        enabled=config.logging.use_wandb,
        project=config.logging.project,
        entity=config.logging.entity,
        run_name=str(config.logging.run_name),
        config=_to_plain_config(config),
        directory=checkpoint_path
    )

    best = float('-inf')
    best_model_path = os.path.join(checkpoint_path, 'model.pth')
    for epoch in range(1, strategy.num_epochs + 1):
        train_metrics = _train_one_epoch(agent, loader, device, epoch, strategy)
        print(f'Train | Epoch: {epoch:4d} | {_summarize_metrics(train_metrics)} |')
        _log_metrics('train', train_metrics, metric_logger, step=epoch, normalize_keys=False)

        if epoch % strategy.eval_interval == 0:
            mean_score = _evaluate_tasks(config, strategy, device, agent, environments, metric_logger, epoch)
            metric_logger.log({'eval/overall/average_return': mean_score}, step=epoch)
            if mean_score > best:
                best = mean_score
                agent.save_model(best_model_path)

        if _supports_dataset_sharing(config) and strategy.share_interval > 0 and epoch % strategy.share_interval == 0:
            _run_dataset_sharing(dataset, strategy, device, agent, metric_logger, epoch)

    if not os.path.exists(best_model_path):
        agent.save_model(best_model_path)
    agent.load_model(best_model_path)
    _run_final_evaluation(config, strategy, device, agent, environments)
    metric_logger.finish()


@hydra.main(version_base=None, config_path='./configs', config_name='trainer')
def main(config):
    run(config)


if __name__ == '__main__':
    main()
