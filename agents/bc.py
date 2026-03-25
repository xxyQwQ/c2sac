import torch
from torch import nn
from torch.optim import Adam

from utils.dataset import prepare_task_indices


class BCPolicy(nn.Module):
    def __init__(self, num_tasks, state_dims, action_dims, num_layers=2, hidden_dims=256):
        super().__init__()
        self.embedding = nn.Embedding(num_tasks, state_dims)
        layers = [nn.Linear(2 * state_dims, hidden_dims), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dims, hidden_dims), nn.ReLU()]
        layers += [nn.Linear(hidden_dims, action_dims)]
        self.policy = nn.Sequential(*layers)

    def forward(self, task, state):
        embedding = torch.cat([state, self.embedding(task)], dim=-1)
        return self.policy(embedding)


class BCAgent:
    def __init__(
        self,
        num_tasks,
        state_dims,
        action_dims,
        num_layers=2,
        hidden_dims=256,
        policy_lr=1e-3,
        **kwargs
    ):
        del kwargs
        self.num_tasks = num_tasks
        self.policy = BCPolicy(num_tasks, state_dims, action_dims, num_layers, hidden_dims)
        self.optimizer = Adam(self.policy.parameters(), lr=policy_lr)
        self.criterion = nn.MSELoss()

    def _task(self, task, state):
        return prepare_task_indices(task, state, self.num_tasks)

    @property
    def modules(self):
        return {
            'policy': self.policy
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
        action = batch['action']
        prediction = self.policy(task, state)
        loss = self.criterion(prediction, action)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            'MSE Loss': loss.item()
        }
