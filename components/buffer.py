from typing import Dict, List

import torch


class BatchBuffer:
    """Replay buffer storing simulated samples as a dict of CPU tensors."""

    def __init__(self, buffer_size: int):
        self.buffer_size: int = buffer_size
        self.batches: Dict[str, List] = {}

    def add(self, batch: dict):
        for k, v in batch.items():
            self.batches.setdefault(k, []).append(v)

    def reset(self):
        self.batches = {}

    def build_dataset(self, duplicates=1):
        total_data = {}
        for k, v in self.batches.items():
            data = torch.cat(v)
            total_data[k] = data[-self.buffer_size:]  # keep the most recent samples
            self.batches[k] = [total_data[k]]
        return BufferDataset(total_data, duplicates)

    def __len__(self):
        if len(self.batches) == 0:
            return 0
        return len(next(iter(self.batches.values())))


class BufferDataset(torch.utils.data.Dataset):
    def __init__(self, total_data: dict, duplicates: int):
        super().__init__()
        assert len(total_data.keys()) > 0
        keys = list(total_data.keys())
        self.len = len(total_data[keys[0]])
        for v in total_data.values():
            assert len(v) == self.len
        self.total_data = total_data
        self.duplicates = duplicates  # expand factor

    def __getitem__(self, idx):
        return {k: v[idx % self.len] for k, v in self.total_data.items()}

    def __len__(self):
        return self.len * self.duplicates
