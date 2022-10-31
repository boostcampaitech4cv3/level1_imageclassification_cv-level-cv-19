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
    
# Swin_b
from torchvision.models import swin_b, swin_t
class Swin_b(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = swin_b(weights='IMAGENET1K_V1')
        self.backbone.head = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return x

class Swin_b_Deep(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = swin_b(weights='DEFAULT')
        self.backbone.head = nn.Sequential(
        nn.Linear(1024, 512),
        nn.LeakyReLU(),
        nn.BatchNorm1d(512),
        nn.Linear(512, 128),
        nn.LeakyReLU(),
        nn.BatchNorm1d(128),
        nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        return x

from torchvision.models import swin_b
class Swin_T_Deep(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = swin_t(weights='DEFAULT')
        self.backbone.head = nn.Sequential(
        nn.Linear(1024, 512),
        nn.LeakyReLU(),
        nn.BatchNorm1d(512),
        nn.Linear(512, 128),
        nn.LeakyReLU(),
        nn.BatchNorm1d(128),
        nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        return x
from torchvision.models import resnext50_32x4d
class ResNext50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = resnext50_32x4d(weights='IMAGENET1K_V2')
        self.features = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
        self.classifier = nn.Sequential(nn.Linear(2048, 1024), nn.LeakyReLU(0.2), nn.BatchNorm1d(1024),
                                        nn.Linear(1024, 512), nn.LeakyReLU(0.2), nn.BatchNorm1d(512),
                                        nn.Linear(512, 128), nn.LeakyReLU(0.2), nn.BatchNorm1d(128),
                                        nn.Linear(128, num_classes)
                                        )
        
    def forward(self, x):
        x = self.features(x)
        mask = self.mask_classifier(x)
        gender = self.gender_classifier(x)
        age = self.age_classifier(x)
        return mask, gender, age

# Convnext_tiny
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
class ConvNext_Tiny(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.backbone = convnext_tiny(weights = ConvNeXt_Tiny_Weights.DEFAULT)
        self.backbone.Linear = nn.Linear(768, num_classes)
        
    def forward(self, x):
        out = self.backbone(x)
        return out

'''# Convnext_samll
from torchvision.models import convnext_small, ConvNeXt_Small_Weights
class ConvNext_Small(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.backbone = convnext_small(weights = ConvNeXt_Small_Weights.DEFAULT)
        self.backbone.Linear = nn.Linear(768, num_classes)
        
    def forward(self, x):
        out = self.backbone(x)
        return out'''

from torchvision.models import resnext101_32x8d
class ResNext101(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = resnext101_32x8d(weights='IMAGENET1K_V2')
        self.features = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
        self.classifier = nn.Sequential(nn.Linear(2048, 1024), nn.LeakyReLU(0.2), nn.BatchNorm1d(1024),
                                        nn.Linear(1024, 512), nn.LeakyReLU(0.2), nn.BatchNorm1d(512),
                                        nn.Linear(512, 128), nn.LeakyReLU(0.2), nn.BatchNorm1d(128),
                                        nn.Linear(128, num_classes)
                                        )
        
    def forward(self, x):
        x = self.features(x)
        out = self.classifier(x)
        return out