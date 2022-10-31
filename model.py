import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import efficientnet_b2
class EfficientNet_B2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = efficientnet_b2(weights='DEFAULT')
        self.backbone.classifier = nn.Sequential(
            nn.Linear(1408, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        output = self.backbone(x)
        return output
    
from torchvision.models import efficientnet_b6
class EfficientNet_B6(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = efficientnet_b6(weights='DEFAULT')
        self.backbone.classifier = nn.Sequential(
            nn.Linear(2304, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        output = self.backbone(x)
        return output