import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)

# resnet 50
from torchvision.models import resnet50, ResNet50_Weights
class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(2048, num_classes)
        
        # freeze except classifier
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.fc.weight.requires_grad = True
        self.backbone.fc.bias.requires_grad = True

    def forward(self, x):
        out = self.backbone(x)
        return out

# resnet 101
from torchvision.models import resnet101
class ResNet101(nn.Module):
    def __init__(self, num_classes): 
        super().__init__()
        
        self.backbone = resnet101(pretrained=True)
        self.fc = nn.Linear(2048, num_classes)
        
        # freeze except classifier
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.fc.weight.requires_grad = True
        self.backbone.fc.bias.requires_grad = True
    
    def forward(self, x):
        out = self.backbone(x)
        return out

# AlexNet
class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(-1, 256*6*6)
        x = self.classifier(x)
        return x

# Swin_b
from torchvision.models import swin_b
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
        self.backbone = swin_b(weights='IMAGENET1K_V1')
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

# Multiple Output Model Template
from torchvision.models import resnext101_64x4d
class MultiHeadBaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnext101_64x4d(weights='IMAGENET1K_V1')
        self.features = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
        self.mask_classifier = nn.Sequential(nn.Linear(2048, 3))
        self.gender_classifier = nn.Sequential(nn.Linear(2048, 2))
        self.age_classifier = nn.Sequential(nn.Linear(2048, 3))
 
    def forward(self, x):
        x = self.features(x)
        mask = self.mask_classifier(x)
        gender = self.gender_classifier(x)
        age = self.age_classifier(x)
        return mask, gender, age

# ViT_B_16
from torchvision.models import vit_b_16
class ViT_B_16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = vit_b_16(weights='DEFAULT')
        for p in self.backbone.parameters():
            p.requires_grad=True
        self.backbone.head = nn.Linear(768, num_classes)

    def forward(self, x):
        output = self.backbone(x)
        return output


# densenet 121
import torch
class DenseNet121(nn.Module):
    def __init__(self, num_classes = 18):
        super().__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.14.0', 'densenet121', pretrained=True)
        fc1 = nn.Linear(1024,512)
        self.backbone.classifier = fc1
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.classifier.weight.requires_grad = True
        self.bn1 = nn.BatchNorm1d(512)
        self.classifier2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.classifier3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.classifier4 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.relu(x)
        x = self.bn1(x)
        #x = self.dropout(x)
        x = self.classifier2(x)
        x = self.relu(x)
        x = self.bn2(x)
        #x = self.dropout(x)
        x = self.classifier3(x)
        x = self.relu(x)
        x = self.bn3(x)
        output = self.classifier4(x)
        return output
    
# densenet201
class DenseNet201(nn.Module):
    def __init__(self, num_classes = 18):
        super().__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.14.0', 'densenet201', pretrained=True)
        fc1 = nn.Linear(1920,1024)
        self.backbone.classifier = fc1
        # for parameter in self.backbone.parameters():
        #     parameter.requires_grad = False
        # self.backbone.classifier.weight.requires_grad = True
        self.bn1 = nn.BatchNorm1d(1024)
        self.classifier2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.classifier3 = nn.Linear(512, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.classifier4 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.classifier2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.classifier3(x)
        x = self.relu(x)
        x = self.bn3(x)
        output = self.classifier4(x)
        return output

# efficientNetb0
class EfficientNet_B0(nn.Module):
    def __init__(self,num_classes = 18):
        super(EfficientNet_B0, self).__init__()
        self.backbone = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
        self.backbone.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        output = self.backbone(x)
        return output

# EfficientNet_B2
from torchvision.models import efficientnet_b2
class EfficientNet_B2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = efficientnet_b2(weights='DEFAULT')
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(1408, num_classes)
        )

    def forward(self, x):
        output = self.backbone(x)
        return output

# EfficientNet_B2 Deepclassifier
from torchvision.models import efficientnet_b2
class EfficientNet_B2_Deep(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = efficientnet_b2(weights='DEFAULT')
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(1408, 512),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        output = self.backbone(x)
        return output
    
# EfficientNet_B2 Deepclassifier with BN
from torchvision.models import efficientnet_b2
class EfficientNet_B2_Deep_BN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = efficientnet_b2(weights='DEFAULT')
        self.backbone.classifier = nn.Sequential(
            nn.Linear(1408, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        output = self.backbone(x)
        return output


from torchvision.models import resnext50_32x4d
class MultiHeadResNext50(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnext50_32x4d(weights='IMAGENET1K_V1')
        self.features = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
        self.mask_classifier = nn.Sequential(nn.Linear(2048, 3))
        self.gender_classifier = nn.Sequential(nn.Linear(2048, 2))
        self.age_classifier = nn.Sequential(nn.Linear(2048, 3))
 
    def forward(self, x):
        x = self.features(x)
        mask = self.mask_classifier(x)
        gender = self.gender_classifier(x)
        age = self.age_classifier(x)
        return mask, gender, age