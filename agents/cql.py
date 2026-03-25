from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.distributions import Normal
from torch.distributions.transforms import TanhTransform
from torch.distributions.transformed_distribution import TransformedDistribution

from utils.dataset import prepare_task_indices
from utils.torch import extend_and_repeat_tensor, soft_update_target_network


class ScalarParameter(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value, dtype=torch.float32))

    def forward(self):
        return self.value


class CQLActor(nn.Module):
    def __init__(self, num_tasks, state_dims, action_dims, num_layers=2, hidden_dims=256):
        super().__init__()
        self.embedding = nn.Embedding(num_tasks, state_dims)
        layers = [nn.Linear(2 * state_dims, hidden_dims), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dims, hidden_dims), nn.ReLU()]
        self.backbone = nn.Sequential(*layers)
        self.mu = nn.Linear(hidden_dims, action_dims)
        self.log_std = nn.Linear(hidden_dims, action_dims)
        self.log_std_scale = ScalarParameter(1.0)
        self.log_std_bias = ScalarParameter(-1.0)

    def _distribution(self, task, state, repeat=None):
        embedding = torch.cat([state, self.embedding(task)], dim=-1)
        if repeat is not None:
            embedding = extend_and_repeat_tensor(embedding, 1, repeat)
        hidden = self.backbone(embedding)
        mu = self.mu(hidden)
        log_std = self.log_std_scale() * self.log_std(hidden) + self.log_std_bias()
        std = torch.exp(torch.clamp(log_std, -20, 2))
        return TransformedDistribution(Normal(mu, std), TanhTransform(cache_size=1)), mu

    def forward(self, task, state, deterministic=False, repeat=None):
        dist, mu = self._distribution(task, state, repeat=repeat)
        action = torch.tanh(mu) if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def log_prob(self, task, state, action, repeat=None):
        dist, _ = self._distribution(task, state, repeat=repeat)
        action = action.clamp(min=-1 + 1e-6, max=1 - 1e-6)
        return dist.log_prob(action).sum(dim=-1)


class CQLCritic(nn.Module):
    def __init__(self, num_tasks, state_dims, action_dims, num_layers=2, hidden_dims=256):
        super().__init__()
        self.embedding = nn.Embedding(num_tasks, state_dims)
        layers = [nn.Linear(2 * state_dims + action_dims, hidden_dims), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dims, hidden_dims), nn.ReLU()]
        self.backbone = nn.Sequential(*layers)
        self.value = nn.Linear(hidden_dims, 1)

    def forward(self, task, state, action, repeat=None):
        embedding = torch.cat([state, self.embedding(task)], dim=-1)
        if repeat is not None:
            embedding = extend_and_repeat_tensor(embedding, 1, repeat)
            embedding = embedding.reshape(-1, embedding.shape[-1])
            action = action.reshape(-1, action.shape[-1])
        hidden = self.backbone(torch.cat([embedding, action], dim=-1))
        value = self.value(hidden)
        if repeat is not None:
            value = value.reshape(-1, repeat)
        return value


class CQLAgent:
    def __init__(
        self,
        num_tasks,
        state_dims,
        action_dims,
        num_layers=3,
        hidden_dims=256,
        actor_lr=1e-4,
        critic_lr=3e-4,
        discount_factor=0.99,
        update_rate=0.01,
        num_samples=5,
        penalty_weight=5.0,
        **kwargs
    ):
        del kwargs
        self.num_tasks = num_tasks
        self.action_dims = action_dims
        self.discount_factor = discount_factor
        self.update_rate = update_rate
        self.num_samples = num_samples
        self.penalty_weight = penalty_weight
        self.target_entropy = -action_dims

        self.log_alpha = ScalarParameter(0.0)
        self.alpha_optimizer = Adam(self.log_alpha.parameters(), lr=actor_lr)

        self.actor = CQLActor(num_tasks, state_dims, action_dims, num_layers, hidden_dims)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)

        self.critic_1 = CQLCritic(num_tasks, state_dims, action_dims, num_layers, hidden_dims)
        self.critic_2 = CQLCritic(num_tasks, state_dims, action_dims, num_layers, hidden_dims)
        self.critic_optimizer = Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=critic_lr
        )

        self.target_critic_1 = deepcopy(self.critic_1)
        self.target_critic_2 = deepcopy(self.critic_2)

    def _task(self, task, state):
        return prepare_task_indices(task, state, self.num_tasks)

    @property
    def modules(self):
        return {
            'log_alpha': self.log_alpha,
            'actor': self.actor,
            'critic_1': self.critic_1,
            'critic_2': self.critic_2,
            'target_critic_1': self.target_critic_1,
            'target_critic_2': self.target_critic_2
        }

    def to_device(self, device):
        for module in self.modules.values():
            module.to(device)

    def save_model(self, path):
        torch.save({name: module.state_dict() for name, module in self.modules.items()}, path)

    def load_model(self, path):
        weights = torch.load(path)
        for name, module in self.modules.items():
            module.load_state_dict(weights[name])

    def take_action(self, task, state=None):
        if state is None:
            state = task
            task = None
        task = self._task(task, state.unsqueeze(0))
        with torch.no_grad():
            action, _ = self.actor(task, state.unsqueeze(0), deterministic=True)
        return action.squeeze(0).detach()

    def _soft_update_targets(self):
        soft_update_target_network(self.critic_1, self.target_critic_1, self.update_rate)
        soft_update_target_network(self.critic_2, self.target_critic_2, self.update_rate)

    def train_batch(self, batch, warmup=False):
        state = batch['state']
        task = self._task(batch['task'], state)
        action = batch['action']
        reward = batch['reward'].unsqueeze(-1) if batch['reward'].ndim == 1 else batch['reward']
        next_state = batch['next_state']
        done = batch['done'].unsqueeze(-1) if batch['done'].ndim == 1 else batch['done']

        sampled_action, log_pi = self.actor(task, state)
        alpha = self.log_alpha().exp()
        alpha_loss = -(self.log_alpha() * (log_pi + self.target_entropy).detach()).mean()

        if warmup:
            behavior_log_prob = self.actor.log_prob(task, state, action)
            actor_loss = (alpha * log_pi - behavior_log_prob).mean()
        else:
            policy_value = torch.min(
                self.critic_1(task, state, sampled_action),
                self.critic_2(task, state, sampled_action)
            )
            actor_loss = (alpha * log_pi - policy_value).mean()

        with torch.no_grad():
            next_action, _ = self.actor(task, next_state)
            target_value = torch.min(
                self.target_critic_1(task, next_state, next_action),
                self.target_critic_2(task, next_state, next_action)
            )
            td_target = reward + self.discount_factor * (1 - done) * target_value

        pred_value_1 = self.critic_1(task, state, action)
        pred_value_2 = self.critic_2(task, state, action)
        critic_1_loss = F.mse_loss(pred_value_1, td_target)
        critic_2_loss = F.mse_loss(pred_value_2, td_target)

        batch_size = state.shape[0]
        random_action = action.new_empty(batch_size, self.num_samples, self.action_dims).uniform_(-1, 1)
        current_action, current_log_prob = self.actor(task, state, repeat=self.num_samples)
        next_action, next_log_prob = self.actor(task, next_state, repeat=self.num_samples)
        current_action = current_action.detach()
        current_log_prob = current_log_prob.detach()
        next_action = next_action.detach()
        next_log_prob = next_log_prob.detach()

        random_value_1 = self.critic_1(task, state, random_action, repeat=self.num_samples)
        random_value_2 = self.critic_2(task, state, random_action, repeat=self.num_samples)
        current_value_1 = self.critic_1(task, state, current_action, repeat=self.num_samples)
        current_value_2 = self.critic_2(task, state, current_action, repeat=self.num_samples)
        next_value_1 = self.target_critic_1(task, next_state, next_action, repeat=self.num_samples)
        next_value_2 = self.target_critic_2(task, next_state, next_action, repeat=self.num_samples)

        random_density = np.log(0.5 ** self.action_dims)
        union_value_1 = torch.cat(
            [random_value_1 - random_density, next_value_1 - next_log_prob, current_value_1 - current_log_prob],
            dim=1
        )
        union_value_2 = torch.cat(
            [random_value_2 - random_density, next_value_2 - next_log_prob, current_value_2 - current_log_prob],
            dim=1
        )
        conservative_loss_1 = self.penalty_weight * (torch.logsumexp(union_value_1, dim=1) - pred_value_1.squeeze(-1)).mean()
        conservative_loss_2 = self.penalty_weight * (torch.logsumexp(union_value_2, dim=1) - pred_value_2.squeeze(-1)).mean()
        critic_loss = critic_1_loss + critic_2_loss + conservative_loss_1 + conservative_loss_2

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self._soft_update_targets()

        return {
            'Alpha Value': alpha.item(),
            'Actor Loss': actor_loss.item(),
            'Critic Loss': (critic_1_loss + critic_2_loss).item(),
            'CQL Loss': (conservative_loss_1 + conservative_loss_2).item()
        }
