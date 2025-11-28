import torch
import torch.nn as nn

class FusionDecoder(nn.Module):
    def __init__(self, vocab_size, ctx_dim, pad_id):
        super().__init__()
        self.ctx_dim = ctx_dim
        self.pad_id = pad_id
        
        # Fusion global init state
        self.global_fuse = nn.Linear(ctx_dim * 2, ctx_dim)
        self.global_ln = nn.LayerNorm(ctx_dim)
        self.global_dropout = nn.Dropout(0.3)
        
        # Embedding answer
        self.embed_ans = nn.Embedding(vocab_size, ctx_dim, padding_idx=pad_id)
        self.emb_dropout = nn.Dropout(0.3)
        
        # LSTM Decoder (Input: [Word_Emb, Img_Global_Ctx])
        self.decoder = nn.LSTM(
            input_size=ctx_dim * 2,
            hidden_size=ctx_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Cross Attention (Query: DecoderOut, Key/Val: Text Tokens)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=ctx_dim, num_heads=4, dropout=0.3, batch_first=True
        )
        
        # Gating & Output Fusion
        self.gate_ff = nn.Linear(ctx_dim * 3, ctx_dim)
        self.fusion_ff = nn.Linear(ctx_dim * 3, ctx_dim)
        self.fusion_dropout = nn.Dropout(0.3)
        self.fusion_ln = nn.LayerNorm(ctx_dim)
        
        # Final Projection
        self.out_proj = nn.Linear(ctx_dim, vocab_size, bias=False)
        self.out_proj.weight = self.embed_ans.weight # Weight Tying

    def forward(self, ans_input_ids, img_ctx, memory, memory_mask, global_ctx):
        """
        img_ctx: (B, C) - Mean pooled image features
        memory: (B, L_txt, C) - Text features (PhoBERT tokens)
        global_ctx: (B, C) - Combined Img+Text context for H0
        """
        # Embed inputs
        tok_emb = self.embed_ans(ans_input_ids)   # (B, L, C)
        tok_emb = self.emb_dropout(tok_emb)
        
        # Expand Img Context to match Seq Len
        img_ctx_exp = img_ctx.unsqueeze(1).expand(-1, tok_emb.size(1), -1)
        dec_input = torch.cat([tok_emb, img_ctx_exp], dim=-1) # (B, L, 2C)
        
        # Init Hidden
        h0 = global_ctx.unsqueeze(0) # (1, B, C)
        c0 = torch.zeros_like(h0)
        
        # LSTM
        dec_out, _ = self.decoder(dec_input, (h0, c0)) # (B, L, C)
        
        # Cross Attention
        attn_ctx, _ = self.cross_attn(dec_out, memory, memory, key_padding_mask=(memory_mask == 0))
        
        # Gating mechanism
        img_ctx_time = img_ctx.unsqueeze(1).expand_as(dec_out)
        gate_input = torch.cat([dec_out, attn_ctx, img_ctx_time], dim=-1)
        gate = torch.sigmoid(self.gate_ff(gate_input))
        visual_ctx = gate * img_ctx_time
        
        # Fusion
        fused = torch.cat([dec_out, attn_ctx, visual_ctx], dim=-1)
        fused = torch.tanh(self.fusion_ff(fused))
        fused = self.fusion_dropout(fused)
        fused = self.fusion_ln(fused)
        
        logits = self.out_proj(fused)
        return logits