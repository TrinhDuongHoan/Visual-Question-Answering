import torch
import torch.nn as nn
from transformers import ViTModel

class ImageEncoder(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224", freeze=True):
        super(ImageEncoder, self).__init__()
        print(f"Loading Image Encoder: {model_name}...")
        
        self.model = ViTModel.from_pretrained(model_name)
        
        # Nếu dataset nhỏ, nên đóng băng (freeze) các lớp feature extraction
        # để tránh làm hỏng weight đã pre-train.
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
                
    def forward(self, pixel_values):
        # pixel_values shape: (Batch_Size, 3, 224, 224)
        outputs = self.model(pixel_values)
        
        # Lấy pooler_output (CLS token) đại diện cho cả tấm ảnh
        # Shape: (Batch_Size, hidden_size) -> thường là 768
        return outputs.pooler_output