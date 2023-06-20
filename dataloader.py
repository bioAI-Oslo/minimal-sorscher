import numpy as np
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(
            self, agent, place_cells, num_samples, seq_len=20, **kwargs
    ):
        self.agent = agent
        self.place_cells = place_cells
        self.seq_len = seq_len
        # since data is generated num_samples is essentially infinite.
        # In practice, this defines the iterator object termination criterion which
        # will be interpreted as an epoch.
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index=None):
        """
        Parameters:
            index: Unused
        Returns:
            velocities (self.seq_len,2): cartesian velocities
            init_pc_positions (npcs,): initial place cell activities
            labels (self.seq_len, npcs): true place cell activities
            positions (self.seq_len, 2): true cartesian positions (not used for training)
        """
        self.agent.reset()
        for _ in range(self.seq_len):
            self.agent.step()
        velocities = torch.tensor(self.agent.velocities[1:], dtype=torch.float32)
        positions = torch.tensor(self.agent.positions, dtype=torch.float32)
        pc_positions = self.place_cells.softmax_response(positions)
        init_pc_positions, labels = pc_positions[0], pc_positions[1:]
        return velocities, init_pc_positions, labels, positions



