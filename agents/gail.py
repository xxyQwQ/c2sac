import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from utils.dataset import prepare_task_indices


class GAILPolicy(nn.Module):
    def __init__(self, num_tasks, state_dims, action_dims, num_layers=2, hidden_dims=256):
        super().__init__()
        self.embedding = nn.Embedding(num_tasks, state_dims)
        layers = [nn.Linear(2 * state_dims, hidden_dims), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dims, hidden_dims), nn.ReLU()]
        layers += [nn.Linear(hidden_dims, action_dims), nn.Tanh()]
        self.policy = nn.Sequential(*layers)

    def forward(self, task, state):
        embedding = torch.cat([state, self.embedding(task)], dim=-1)
        return self.policy(embedding)


class GAILDiscriminator(nn.Module):
    def __init__(self, num_tasks, state_dims, action_dims, num_layers=2, hidden_dims=256):
        super().__init__()
        self.embedding = nn.Embedding(num_tasks, state_dims)
        layers = [nn.Linear(2 * state_dims + action_dims, hidden_dims), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dims, hidden_dims), nn.ReLU()]
        layers += [nn.Linear(hidden_dims, 1)]
        self.discriminator = nn.Sequential(*layers)

    def forward(self, task, state, action):
        embedding = torch.cat([state, self.embedding(task), action], dim=-1)
        return self.discriminator(embedding)


class GAILAgent:
    def __init__(
        self,
        num_tasks,
        state_dims,
        action_dims,
        num_layers=3,
        hidden_dims=256,
        policy_lr=1e-4,
        discriminator_lr=3e-4,
        bc_weight=1.0,
        adv_weight=1.0,
        **kwargs
    ):
        del kwargs
        self.num_tasks = num_tasks
        self.policy = GAILPolicy(num_tasks, state_dims, action_dims, num_layers, hidden_dims)
        self.discriminator = GAILDiscriminator(num_tasks, state_dims, action_dims, num_layers, hidden_dims)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=policy_lr)
        self.discriminator_optimizer = Adam(self.discriminator.parameters(), lr=discriminator_lr)
        self.bc_weight = bc_weight
        self.adv_weight = adv_weight

    def _task(self, task, state):
        return prepare_task_indices(task, state, self.num_tasks)

    @property
    def modules(self):
        return {
            'policy': self.policy,
            'discriminator': self.discriminator
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
            action = self.policy(task, state.unsqueeze(0))
        return action.squeeze(0).detach()

    def train_batch(self, batch):
        state = batch['state']
        task = self._task(batch['task'], state)
        expert_action = batch['action']
        fake_action = self.policy(task, state)

        real_logits = self.discriminator(task, state, expert_action)
        fake_logits = self.discriminator(task, state, fake_action.detach())
        discriminator_loss = (
            F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits)) +
            F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        )

        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        fake_logits = self.discriminator(task, state, fake_action)
        adversarial_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
        bc_loss = F.mse_loss(fake_action, expert_action)
        policy_loss = self.adv_weight * adversarial_loss + self.bc_weight * bc_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return {
            'Policy Loss': policy_loss.item(),
            'BC Loss': bc_loss.item(),
            'Discriminator Loss': discriminator_loss.item()
        }
