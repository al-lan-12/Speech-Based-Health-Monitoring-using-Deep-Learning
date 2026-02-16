import torch
import torch.nn as nn

class DualHeadClassifier(nn.Module):
    def __init__(self, input_dim=768, emotion_classes=5, binary_classes=2):
        super(DualHeadClassifier, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU()
        )
        self.emotion_head = nn.Linear(256, emotion_classes)
        self.binary_head = nn.Linear(256, binary_classes)

    def forward(self, x):
        x = self.shared(x)
        return self.emotion_head(x), self.binary_head(x)
