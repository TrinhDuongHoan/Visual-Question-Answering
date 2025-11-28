import torch.nn as nn
import timm

class ViTEncoder(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224"):
        super().__init__()
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool=""
        )
        self.out_dim = self.backbone.num_features

    def forward(self, x):
        return self.backbone(x) 