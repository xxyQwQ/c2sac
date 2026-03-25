from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from utils.dataset import prepare_task_indices
from utils.torch import extend_and_repeat_tensor, soft_update_target_network


class BCQVAE(nn.Module):
    def __init__(self, num_tasks, state_dims, action_dims, hidden_dims=256, latent_dims=None):
        super().__init__()
        latent_dims = latent_dims or action_dims * 2
        self.latent_dims = latent_dims
        self.embedding = nn.Embedding(num_tasks, state_dims)
        self.encoder = nn.Sequential(
            nn.Linear(2 * state_dims + action_dims, hidden_dims), nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims), nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dims, latent_dims)
        self.log_sigma = nn.Linear(hidden_dims, latent_dims)
        self.decoder = nn.Sequential(
            nn.Linear(2 * state_dims + latent_dims, hidden_dims), nn.ReLU(),
            nn.Linear(hidden_dims, action_dims), nn.Tanh()
        )

    def forward(self, task, state, action):
        task_embedding = self.embedding(task)
        latent = self.encoder(torch.cat([state, task_embedding, action], dim=-1))
        mu = self.mu(latent)
        log_sigma = torch.clamp(self.log_sigma(latent), -4, 15)
        sigma = torch.exp(log_sigma)
        z = mu + sigma * torch.randn_like(sigma)
        reconstruction = self.decode(task, state, z)
        return reconstruction, mu, sigma

    def decode(self, task, state, z=None):
        if z is None:
            z = torch.randn(*state.shape[:-1], self.latent_dims, device=state.device)
            z = torch.clamp(z, -0.5, 0.5)
        task_embedding = self.embedding(task)
        return self.decoder(torch.cat([state, task_embedding, z], dim=-1))


class BCQPerturbationActor(nn.Module):
    def __init__(self, num_tasks, state_dims, action_dims, hidden_dims=256, perturbation_scale=0.05):
        super().__init__()
        self.perturbation_scale = perturbation_scale
        self.embedding = nn.Embedding(num_tasks, state_dims)
        self.network = nn.Sequential(
            nn.Linear(2 * state_dims + action_dims, hidden_dims), nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims), nn.ReLU(),
            nn.Linear(hidden_dims, action_dims), nn.Tanh()
        )

    def forward(self, task, state, action):
        task_embedding = self.embedding(task)
        perturbation = self.perturbation_scale * self.network(torch.cat([state, task_embedding, action], dim=-1))
        return torch.clamp(action + perturbation, -1.0, 1.0)


class BCQCritic(nn.Module):
    def __init__(self, num_tasks, state_dims, action_dims, hidden_dims=256):
        super().__init__()
        self.embedding = nn.Embedding(num_tasks, state_dims)
        self.q1 = nn.Sequential(
            nn.Linear(2 * state_dims + action_dims, hidden_dims), nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims), nn.ReLU(),
            nn.Linear(hidden_dims, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(2 * state_dims + action_dims, hidden_dims), nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims), nn.ReLU(),
            nn.Linear(hidden_dims, 1)
        )

    def forward(self, task, state, action):
        task_embedding = self.embedding(task)
        embedding = torch.cat([state, task_embedding, action], dim=-1)
        return self.q1(embedding), self.q2(embedding)

    def q1_value(self, task, state, action):
        task_embedding = self.embedding(task)
        return self.q1(torch.cat([state, task_embedding, action], dim=-1))


class BCQAgent:
    def __init__(
        self,
        num_tasks,
        state_dims,
        action_dims,
        hidden_dims=256,
        actor_lr=1e-4,
        critic_lr=3e-4,
        vae_lr=1e-4,
        discount_factor=0.99,
        update_rate=0.01,
        num_action_samples=5,
        perturbation_scale=0.05,
        **kwargs
    ):
        del kwargs
        self.num_tasks = num_tasks
        self.discount_factor = discount_factor
        self.update_rate = update_rate
        self.num_action_samples = num_action_samples

        self.vae = BCQVAE(num_tasks, state_dims, action_dims, hidden_dims)
        self.vae_optimizer = Adam(self.vae.parameters(), lr=vae_lr)

        self.actor = BCQPerturbationActor(num_tasks, state_dims, action_dims, hidden_dims, perturbation_scale)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.target_actor = deepcopy(self.actor)

        self.critic = BCQCritic(num_tasks, state_dims, action_dims, hidden_dims)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.target_critic = deepcopy(self.critic)

    def _task(self, task, state):
        return prepare_task_indices(task, state, self.num_tasks)

    @property
    def modules(self):
        return {
            'vae': self.vae,
            'actor': self.actor,
            'critic': self.critic,
            'target_actor': self.target_actor,
            'target_critic': self.target_critic
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

    def _sample_action_candidates(self, task, state, use_target=False):
        actor = self.target_actor if use_target else self.actor
        repeated_task = extend_and_repeat_tensor(task, 1, self.num_action_samples)
        repeated_state = extend_and_repeat_tensor(state, 1, self.num_action_samples)
        sampled_action = self.vae.decode(repeated_task, repeated_state)
        perturbed_action = actor(repeated_task, repeated_state, sampled_action)
        return repeated_task, repeated_state, perturbed_action

    def take_action(self, task, state=None):
        if state is None:
            state = task
            task = None
        with torch.no_grad():
            task = self._task(task, state.unsqueeze(0))
            state = state.unsqueeze(0)
            repeated_task, repeated_state, action = self._sample_action_candidates(task, state)
            q1 = self.critic.q1_value(
                repeated_task.reshape(-1),
                repeated_state.reshape(-1, repeated_state.shape[-1]),
                action.reshape(-1, action.shape[-1])
            ).reshape(1, self.num_action_samples)
            index = q1.argmax(dim=1)
            action = action[0, index.item()]
        return action.detach()

    def soft_update(self):
        soft_update_target_network(self.actor, self.target_actor, self.update_rate)
        soft_update_target_network(self.critic, self.target_critic, self.update_rate)

    def train_batch(self, batch):
        state = batch['state']
        task = self._task(batch['task'], state)
        action = batch['action']
        reward = batch['reward'].unsqueeze(-1) if batch['reward'].ndim == 1 else batch['reward']
        next_state = batch['next_state']
        done = batch['done'].unsqueeze(-1) if batch['done'].ndim == 1 else batch['done']

        reconstruction, mu, sigma = self.vae(task, state, action)
        reconstruction_loss = F.mse_loss(reconstruction, action)
        kl_loss = -0.5 * (1 + torch.log(sigma.pow(2) + 1e-6) - mu.pow(2) - sigma.pow(2)).mean()
        vae_loss = reconstruction_loss + 0.5 * kl_loss
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()

        with torch.no_grad():
            repeated_task, repeated_next_state, next_action = self._sample_action_candidates(task, next_state, use_target=True)
            flat_task = repeated_task.reshape(-1)
            flat_state = repeated_next_state.reshape(-1, repeated_next_state.shape[-1])
            flat_action = next_action.reshape(-1, next_action.shape[-1])
            target_q1, target_q2 = self.target_critic(flat_task, flat_state, flat_action)
            target_q = torch.min(target_q1, target_q2).reshape(next_state.shape[0], self.num_action_samples)
            target_q = target_q.max(dim=1, keepdim=True).values
            td_target = reward + self.discount_factor * (1 - done) * target_q

        pred_q1, pred_q2 = self.critic(task, state, action)
        critic_loss = F.mse_loss(pred_q1, td_target) + F.mse_loss(pred_q2, td_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        sampled_action = self.vae.decode(task, state)
        perturbed_action = self.actor(task, state, sampled_action)
        actor_loss = -self.critic.q1_value(task, state, perturbed_action).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update()

        return {
            'VAE Loss': vae_loss.item(),
            'Actor Loss': actor_loss.item(),
            'Critic Loss': critic_loss.item()
        }
