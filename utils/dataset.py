import glob

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


DATA_KEYS = ('task', 'state', 'action', 'reward', 'next_state', 'done')


def collate_batch(batch):
    return {
        key: np.stack([item[key] for item in batch], axis=0)
        for key in DATA_KEYS
    }


def prepare_task_indices(task, state, num_tasks):
    if task is None:
        return torch.zeros(state.shape[0], dtype=torch.long, device=state.device)
    if task.ndim == 0:
        task = task.unsqueeze(0)
    task = task.to(device=state.device, dtype=torch.long)
    if task.numel() == 1 and state.shape[0] != 1:
        task = task.expand(state.shape[0])
    if task.shape[0] != state.shape[0]:
        raise ValueError(
            f'Task batch size must match state batch size, got {task.shape[0]} and {state.shape[0]}.'
        )
    if torch.any(task < 0) or torch.any(task >= num_tasks):
        raise ValueError(f'Task indices must be in [0, {num_tasks - 1}], got: {task.detach().cpu().tolist()}')
    return task


class SharingDataset(Dataset):
    def __init__(self, dataset_paths):
        self.stores = []
        for task_index, dataset_path in enumerate(dataset_paths):
            store = {key: [] for key in DATA_KEYS}
            path_list = sorted(glob.glob(f'{dataset_path}/*.npz'))
            if not path_list:
                raise FileNotFoundError(f'No dataset shards found in {dataset_path}')
            for path in path_list:
                with open(path, 'rb') as file:
                    record = np.load(file)
                    record = {key: record[key] for key in record.keys()}
                    state = record['observation'][:-1]
                    action = record['action'][1:]
                    reward = record['reward'][1:]
                    next_state = record['observation'][1:]
                    done = np.zeros_like(reward)
                    task = np.full(state.shape[0], task_index, dtype=np.int64)
                    store['task'].append(task)
                    store['state'].append(state)
                    store['action'].append(action)
                    store['reward'].append(reward)
                    store['next_state'].append(next_state)
                    store['done'].append(done)
            for key in store.keys():
                store[key] = np.concatenate(store[key], axis=0)
            self.stores.append(store)
        self.num_tasks = len(self.stores)
        self.buffer = {key: [] for key in DATA_KEYS}
        for store in self.stores:
            for key in store.keys():
                self.buffer[key].append(store[key])
        for key in self.buffer.keys():
            self.buffer[key] = np.concatenate(self.buffer[key], axis=0)

    def __len__(self):
        return self.buffer['state'].shape[0]

    def __getitem__(self, index):
        return {key: self.buffer[key][index] for key in DATA_KEYS}

    def update(self, device, agent, percent=0.1):
        self.buffer = {key: [] for key in DATA_KEYS}
        for store in self.stores:
            for key in store.keys():
                self.buffer[key].append(store[key])
        values = []
        for task_index, task_store in enumerate(self.stores):
            value_list = []
            task = torch.tensor(task_index, dtype=torch.long).to(device)
            for item_index in tqdm(range(task_store['state'].shape[0]), desc='Updating', leave=False):
                state = torch.from_numpy(task_store['state'][item_index]).to(device)
                action = torch.from_numpy(task_store['action'][item_index]).to(device)
                value = agent.compute_value(task, state, action).cpu().numpy()
                value_list.append(value)
            values.append(np.percentile(value_list, 100 * (1 - percent)))
        for source_index in range(len(self.stores)):
            store = self.stores[source_index]
            for target_index in range(len(self.stores)):
                if source_index == target_index:
                    continue
                task = torch.tensor(target_index, dtype=torch.long).to(device)
                for item_index in tqdm(range(store['state'].shape[0]), desc='Updating', leave=False):
                    state = torch.from_numpy(store['state'][item_index]).to(device)
                    action = torch.from_numpy(store['action'][item_index]).to(device)
                    value = agent.compute_value(task, state, action).cpu().numpy()
                    if value > values[target_index]:
                        for key in store.keys():
                            if key == 'task':
                                self.buffer[key].append(np.array([target_index]))
                            else:
                                self.buffer[key].append(store[key][item_index][np.newaxis])
        for key in self.buffer.keys():
            self.buffer[key] = np.concatenate(self.buffer[key], axis=0)
        return {'Critical Value': values}
