import torch.nn as nn
import torch.nn.functional as F


class DuelingQNetwork(nn.Module):
    """Dueling Q-network (https://arxiv.org/abs/1511.06581)"""

    def __init__(self, state_size, action_size, hide_size_1=128, hide_size_2=128):
        super(DuelingQNetwork, self).__init__()

        # value network
        self.fc1_val = nn.Linear(state_size, hide_size_1)
        self.fc2_val = nn.Linear(hide_size_1, hide_size_2)
        self.fc4_val = nn.Linear(hide_size_2, 1)

        # advantage network
        self.fc1_adv = nn.Linear(state_size, hide_size_1)
        self.fc2_adv = nn.Linear(hide_size_1, hide_size_2)
        self.fc4_adv = nn.Linear(hide_size_2, action_size)

    def forward(self, x):
        val = F.relu(self.fc1_val(x))
        val = F.relu(self.fc2_val(val))
        val = self.fc4_val(val)

        # advantage calculation
        adv = F.relu(self.fc1_adv(x))
        adv = F.relu(self.fc2_adv(adv))
        adv = self.fc4_adv(adv)

        return val + adv - adv.mean()
