import torch
import torch.nn as nn
import torch.nn.functional as F
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
        img_ctx = img_feats.mean(dim=1) 
        
        # Text Features
        txt_tokens = self.text_encoder(input_ids=q_input_ids, attention_mask=q_attention_mask)
        txt_feats = torch.tanh(self.txt_proj(txt_tokens)) 
        
        # Global Context
        mask_exp = q_attention_mask.unsqueeze(-1)
        txt_sum = (txt_feats * mask_exp).sum(dim=1)
        lengths = mask_exp.sum(dim=1).clamp(min=1)
        txt_ctx = txt_sum / lengths
        
        global_ctx = torch.tanh(self.decoder.global_fuse(torch.cat([img_ctx, txt_ctx], dim=-1)))
        global_ctx = self.decoder.global_ln(global_ctx)
        global_ctx = self.decoder.global_dropout(global_ctx)
        
        return img_ctx, txt_feats, q_attention_mask, global_ctx

    def forward(self, images, q_input_ids, q_attention_mask, ans_input_ids):
        # Forward dùng cho training (teacher forcing)
        img_ctx, memory, memory_mask, global_ctx = self.build_memory(images, q_input_ids, q_attention_mask)
        dec_in_ids = ans_input_ids[:, :-1]
        targets = ans_input_ids[:, 1:]
        logits = self.decoder(dec_in_ids, img_ctx, memory, memory_mask, global_ctx)
        return logits, targets

    # ================= BEAM SEARCH LOGIC =================
    
    def _decode_step(self, prev_token, h, c, memory, memory_mask, img_ctx):
        """Bước nhảy đơn lẻ cho decoder (dùng chung cho cả Greedy và Beam)"""
        # Embed
        tok_emb = self.decoder.embed_ans(prev_token).unsqueeze(1) # (B, 1, C)
        tok_emb = self.decoder.emb_dropout(tok_emb)
        
        # Expand Image Context
        img_ctx_exp = img_ctx.unsqueeze(1) # (B, 1, C)
        dec_input = torch.cat([tok_emb, img_ctx_exp], dim=-1) # (B, 1, 2C)
        
        # LSTM Step
        dec_out, (h, c) = self.decoder.decoder(dec_input, (h, c))
        
        # Cross Attention
        attn_ctx, _ = self.decoder.cross_attn(dec_out, memory, memory, key_padding_mask=(memory_mask == 0))
        
        # Gating & Fusion
        gate_input = torch.cat([dec_out, attn_ctx, img_ctx_exp], dim=-1)
        gate = torch.sigmoid(self.decoder.gate_ff(gate_input))
        visual_ctx = gate * img_ctx_exp
        
        fused = torch.cat([dec_out, attn_ctx, visual_ctx], dim=-1)
        fused = torch.tanh(self.decoder.fusion_ff(fused))
        fused = self.decoder.fusion_dropout(fused)
        fused = self.decoder.fusion_ln(fused)
        
        # Logits
        logits = self.decoder.out_proj(fused.squeeze(1)) # (B, V)
        return logits, h, c

    def _generate_one_beam(self, img_ctx, memory, memory_mask, global_ctx, max_len, num_beams):
        """Beam search cho 1 sample duy nhất (Logic gốc từ notebook)"""
        device = img_ctx.device
        C = img_ctx.size(-1)
        Lm = memory.size(1)
        beam_size = num_beams

        # Duplicate inputs cho beam size
        img_beam = img_ctx.expand(beam_size, C)            
        mem_beam = memory.expand(beam_size, Lm, C)         
        mask_beam = memory_mask.expand(beam_size, Lm)      
        h = global_ctx.expand(1, beam_size, C).contiguous()
        c = torch.zeros_like(h)

        prev_tokens = torch.full((beam_size,), self.vocab.BOS_ID, dtype=torch.long, device=device)

        # Khởi tạo beams
        beams = [{"tokens": [], "log_prob": 0.0, "finished": False}] + \
                [{"tokens": [], "log_prob": float("-inf"), "finished": True} for _ in range(beam_size - 1)]

        for _ in range(max_len):
            logits, h, c = self._decode_step(prev_tokens, h, c, mem_beam, mask_beam, img_beam)
            log_probs = F.log_softmax(logits, dim=-1) # (beam, V)

            # Xử lý các beam đã kết thúc
            for i, beam in enumerate(beams):
                if beam["finished"]:
                    log_probs[i, :] = float("-inf")
                    log_probs[i, self.vocab.EOS_ID] = 0.0 # Chỉ cho phép chọn EOS tiếp

            # Cộng dồn log_prob hiện tại
            beam_log_probs = torch.tensor([b["log_prob"] for b in beams], device=device).unsqueeze(1)
            total_log_probs = log_probs + beam_log_probs # (beam, V)

            # Lấy top k candidates
            flat = total_log_probs.view(-1)
            topk_log_probs, topk_indices = torch.topk(flat, beam_size)

            new_beams = []
            new_prev_tokens = torch.zeros_like(prev_tokens)
            V = logits.size(-1)

            for new_i, (lp, idx) in enumerate(zip(topk_log_probs, topk_indices)):
                beam_idx = (idx // V).item()
                token_id = (idx % V).item()

                old_beam = beams[beam_idx]
                new_tokens = old_beam["tokens"].copy()
                finished = old_beam["finished"]

                if not finished:
                    if token_id == self.vocab.EOS_ID:
                        finished = True
                    else:
                        new_tokens.append(token_id)

                new_beams.append({
                    "tokens": new_tokens,
                    "log_prob": lp.item(),
                    "finished": finished
                })
                new_prev_tokens[new_i] = token_id

            beams = new_beams
            prev_tokens = new_prev_tokens
            
            # Cập nhật hidden state cho bước tiếp theo dựa trên beam index đã chọn
            # (Bước này quan trọng để align hidden state với beam mới)
            beam_indices = (topk_indices // V) # Indices của beam gốc
            h = h[:, beam_indices, :]
            c = c[:, beam_indices, :]

            if all(b["finished"] for b in beams):
                break

        # Chọn beam tốt nhất đã finish
        finished_beams = [b for b in beams if b["finished"] and len(b["tokens"]) > 0]
        if len(finished_beams) == 0: finished_beams = beams
        best_beam = max(finished_beams, key=lambda b: b["log_prob"])
        
        return best_beam["tokens"]

    def generate_beam(self, images, q_input_ids, q_attention_mask, max_len=15, num_beams=3):
        """Hàm gọi chính để sinh text dùng Beam Search"""
        self.eval()
        decoded_sentences = []
        with torch.no_grad():
            img_ctx, memory, memory_mask, global_ctx = self.build_memory(images, q_input_ids, q_attention_mask)
            B = images.size(0)
            
            for i in range(B):
                # Slice từng sample trong batch
                img_i = img_ctx[i:i+1]
                mem_i = memory[i:i+1]
                mask_i = memory_mask[i:i+1]
                ctx_i = global_ctx[i:i+1]
                
                token_ids = self._generate_one_beam(
                    img_i, mem_i, mask_i, ctx_i,
                    max_len=max_len,
                    num_beams=num_beams
                )
                decoded_sentences.append(self.vocab.decode(token_ids))
                
        return decoded_sentences