"""
Test model definitions at module level for pickling.
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Simple CNN model for testing."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 56 * 56, 10)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SimpleMLP(nn.Module):
    """Simple MLP model for testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
        self.relu = nn.ReLU()

    def forward(self, input_ids=None, **kwargs):
        if input_ids is not None:
            x = input_ids.float()
        else:
            x = kwargs.get("x")
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


# Register safe globals for PyTorch 2.6+
try:
    torch.serialization.add_safe_globals([SimpleCNN, SimpleMLP])
except Exception:
    pass  # Older PyTorch versions
