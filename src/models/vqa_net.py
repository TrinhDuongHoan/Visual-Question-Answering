import torch
import torch.nn as nn
from src.models.image_encoder import ViTEncoder
from src.models.text_encoder import PhoBERTEncoder
from src.models.decoder import FusionDecoder

class VQANet(nn.Module):
    def __init__(self, cfg, vocab):
        super().__init__()
        self.cfg = cfg
        self.vocab = vocab
        ctx_dim = cfg.DEC_HIDDEN_SIZE
        
        # 1. Encoders
        self.vit = ViTEncoder(cfg.VISION_NAME)
        self.text_encoder = PhoBERTEncoder(cfg.TEXT_ENCODER_NAME)
        
        # 2. Projections (về cùng dimension)
        self.img_proj = nn.Linear(self.vit.out_dim, ctx_dim)
        self.txt_proj = nn.Linear(self.text_encoder.hidden_size, ctx_dim)
        
        # 3. Decoder
        self.decoder = FusionDecoder(len(vocab), ctx_dim, vocab.PAD_ID)
        
    def build_memory(self, images, q_input_ids, q_attention_mask):
        # Image Features
        img_tokens = self.vit(images)
        img_feats = torch.tanh(self.img_proj(img_tokens))
        img_ctx = img_feats.mean(dim=1) # (B, C)
        
        # Text Features
        txt_tokens = self.text_encoder(input_ids=q_input_ids, attention_mask=q_attention_mask)
        txt_feats = torch.tanh(self.txt_proj(txt_tokens)) # (B, L, C)
        
        # Create Global Context (for H0 init)
        mask_exp = q_attention_mask.unsqueeze(-1)
        txt_sum = (txt_feats * mask_exp).sum(dim=1)
        lengths = mask_exp.sum(dim=1).clamp(min=1)
        txt_ctx = txt_sum / lengths
        
        global_ctx = torch.tanh(self.decoder.global_fuse(torch.cat([img_ctx, txt_ctx], dim=-1)))
        global_ctx = self.decoder.global_ln(global_ctx)
        global_ctx = self.decoder.global_dropout(global_ctx)
        
        return img_ctx, txt_feats, q_attention_mask, global_ctx

    def forward(self, images, q_input_ids, q_attention_mask, ans_input_ids):
        # 1. Encode & Build Memory
        img_ctx, memory, memory_mask, global_ctx = self.build_memory(images, q_input_ids, q_attention_mask)
        
        # 2. Decode Inputs (Shifted right)
        dec_in_ids = ans_input_ids[:, :-1]
        targets = ans_input_ids[:, 1:]
        
        # 3. Pass through Decoder
        logits = self.decoder(dec_in_ids, img_ctx, memory, memory_mask, global_ctx)
        
        return logits, targets