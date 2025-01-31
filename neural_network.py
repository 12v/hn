import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 4),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(4, 1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
