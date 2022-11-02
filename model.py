import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import efficientnet_b0
class EfficientNet_B0(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = efficientnet_b0(weights='DEFAULT')
        self.backbone.classifier = nn.Sequential(
            nn.Linear(1280, 1024),
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

from torchvision.models import swin_t
class Swin_T_Deep(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = swin_t(weights='DEFAULT')
        self.backbone.head = nn.Sequential(
        nn.Linear(768, 512),
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
        out = self.classifier(x)
        return out

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


# densenet 121
import torch
class DenseNet121(nn.Module):
    def __init__(self, num_classes = 18):
        super().__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.14.0', 'densenet121', pretrained=True)
        fc1 = nn.Linear(1024,512)
        self.backbone.classifier = fc1
        '''for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.classifier.weight.requires_grad = True'''
        self.bn1 = nn.BatchNorm1d(512)
        self.classifier2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.classifier3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.relu(x)
        x = self.bn1(x)
        #x = self.dropout(x)
        x = self.classifier2(x)
        x = self.relu(x)
        x = self.bn2(x)
        #x = self.dropout(x)
        output = self.classifier3(x)
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
        self.classifier2 = nn.Linear(1024, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.classifier3 = nn.Linear(256, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.classifier4 = nn.Linear(64, num_classes)
        self.leaky_relu = nn.LeakyReLU()
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.leaky_relu(x)
        x = self.bn1(x)
        #x = self.dropout(x)
        x = self.classifier2(x)
        x = self.leaky_relu(x)
        x = self.bn2(x)
        #x = self.dropout(x)
        x = self.classifier3(x)
        x = self.leaky_relu(x)
        x = self.bn3(x)
        output = self.classifier4(x)
        return output

class MultiHeadResNext50(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnext101_32x8d(weights='IMAGENET1K_V1')
        self.features = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
        self.mask_classifier = nn.Sequential(nn.Linear(2048, 1024), nn.LeakyReLU(0.2), nn.BatchNorm1d(1024),
                                        nn.Linear(1024, 512), nn.LeakyReLU(0.2), nn.BatchNorm1d(512),
                                        nn.Linear(512, 128), nn.LeakyReLU(0.2), nn.BatchNorm1d(128),
                                        nn.Linear(128, 3)
                                        )
        self.gender_classifier = nn.Sequential(nn.Linear(2048, 1024), nn.LeakyReLU(0.2), nn.BatchNorm1d(1024),
                                        nn.Linear(1024, 512), nn.LeakyReLU(0.2), nn.BatchNorm1d(512),
                                        nn.Linear(512, 128), nn.LeakyReLU(0.2), nn.BatchNorm1d(128),
                                        nn.Linear(128, 2)
                                        )
        self.age_classifier = nn.Sequential(nn.Linear(2048, 1024), nn.LeakyReLU(0.2), nn.BatchNorm1d(1024),
                                        nn.Linear(1024, 512), nn.LeakyReLU(0.2), nn.BatchNorm1d(512),
                                        nn.Linear(512, 128), nn.LeakyReLU(0.2), nn.BatchNorm1d(128),
                                        nn.Linear(128, 3)
                                        )

    def forward(self, x):
        x = self.features(x)
        mask = self.mask_classifier(x)
        gender = self.gender_classifier(x)
        age = self.age_classifier(x)
        return mask, gender, age


from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
class ConvNext_Tiny(nn.Module): 
    def __init__(self,num_classes): 
        super().__init__()
        self.backbone = convnext_tiny(weights = ConvNeXt_Tiny_Weights.DEFAULT)
        self.backbone.classifier = nn.Sequential(
            self.LayerNorm2d((768,), eps=1e-06, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=768, out_features=256, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(in_features=256, out_features=64, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(in_features=64, out_features=num_classes, bias=True)
        )
    
    def forward(self, x):
        out = self.backbone(x)
        return out
    
    class LayerNorm2d(nn.LayerNorm):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            x = x.permute(0, 3, 1, 2)
            return x

from torchvision.models import convnext_small, ConvNeXt_Small_Weights
class ConvNext_Small(nn.Module): 
    def __init__(self,num_classes): 
        super().__init__()
        self.backbone = convnext_small(weights = ConvNeXt_Small_Weights.DEFAULT)
        self.backbone.classifier = nn.Sequential(
            self.LayerNorm2d((768,), eps=1e-06, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=768, out_features=256, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(in_features=256, out_features=64, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(in_features=64, out_features=num_classes, bias=True)
        )
    
    def forward(self, x):
        out = self.backbone(x)
        return out
    
    class LayerNorm2d(nn.LayerNorm):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            x = x.permute(0, 3, 1, 2)
            return x


class ConvNext_Small_Shallow(nn.Module): 
    def __init__(self,num_classes): 
        super().__init__()
        self.backbone = convnext_small(weights = ConvNeXt_Small_Weights.DEFAULT)
        self.backbone.classifier = nn.Sequential(
            self.LayerNorm2d((768,), eps=1e-06, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=768, out_features=64, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(in_features=64, out_features=num_classes, bias=True)
        )
    
    def forward(self, x):
        out = self.backbone(x)
        return out
    
    class LayerNorm2d(nn.LayerNorm):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            x = x.permute(0, 3, 1, 2)
            return x


class ConvNext_Small_Shallow_without_BN(nn.Module): 
    def __init__(self,num_classes): 
        super().__init__()
        self.backbone = convnext_small(weights = ConvNeXt_Small_Weights.DEFAULT)
        self.backbone.classifier = nn.Sequential(
            self.LayerNorm2d((768,), eps=1e-06, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=768, out_features=64, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=num_classes, bias=True)
        )
    
    def forward(self, x):
        out = self.backbone(x)
        return out
    
    class LayerNorm2d(nn.LayerNorm):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            x = x.permute(0, 3, 1, 2)
            return x


from torchvision.models import convnext_base, ConvNeXt_Base_Weights
class ConvNext_Base(nn.Module): 
    def __init__(self,num_classes): 
        super().__init__()
        self.backbone = convnext_base(weights = ConvNeXt_Base_Weights.DEFAULT)
        self.backbone.classifier = nn.Sequential(
            self.LayerNorm2d((1024,), eps=1e-06, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=1024, out_features=512, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(in_features=256, out_features=64, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(in_features=64, out_features=num_classes, bias=True)
        )
    
    def forward(self, x):
        out = self.backbone(x)
        return out
    
    class LayerNorm2d(nn.LayerNorm):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            x = x.permute(0, 3, 1, 2)
            return x




    
from torchvision.models import convnext_large, ConvNeXt_Large_Weights
class ConvNext_Large(nn.Module): 
    def __init__(self,num_classes): 
        super().__init__()
        self.backbone = convnext_large(weights = ConvNeXt_Large_Weights.DEFAULT)
        self.backbone.classifier = nn.Sequential(
            self.LayerNorm2d((1536,), eps=1e-06, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=1536, out_features=512, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(in_features=256, out_features=64, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(in_features=64, out_features=num_classes, bias=True)
        )
    
    def forward(self, x):
        out = self.backbone(x)
        return out
    
    class LayerNorm2d(nn.LayerNorm):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            x = x.permute(0, 3, 1, 2)
            return x