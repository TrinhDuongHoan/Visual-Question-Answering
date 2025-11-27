import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from src.models.image_encoder import ImageEncoder
from src.models.text_encoder import TextEncoder

class VQANet(nn.Module):
    def __init__(self, 
                 vit_name="google/vit-base-patch16-224", 
                 phobert_name="vinai/phobert-base", 
                 gpt_name="minhtoan/vietnamese-gpt2-finetune"):
        super(VQANet, self).__init__()
        
        # 1. Encoders (Quan trọng: Freeze=False để học tinh chỉnh)
        self.image_encoder = ImageEncoder(vit_name, freeze=False)
        self.text_encoder = TextEncoder(phobert_name, freeze=False)
        
        # 2. Decoder
        self.decoder = GPT2LMHeadModel.from_pretrained(gpt_name)
        
        # 3. Projectors (Cầu nối)
        # Vì ViT và PhoBERT có thể khác chiều với GPT, ta cần lớp Linear để chiếu về cùng chiều
        self.img_hidden = self.image_encoder.model.config.hidden_size
        self.txt_hidden = self.text_encoder.model.config.hidden_size
        self.gpt_hidden = self.decoder.config.n_embd
        
        # Chiếu đặc trưng ảnh sang không gian GPT
        self.img_projector = nn.Linear(self.img_hidden, self.gpt_hidden)
        
        # Chiếu đặc trưng text sang không gian GPT
        self.txt_projector = nn.Linear(self.txt_hidden, self.gpt_hidden)
        
        # Layer Norm để ổn định training (Rất quan trọng cho Generative)
        self.ln_visual = nn.LayerNorm(self.gpt_hidden)

    def forward(self, pixel_values, question_ids, question_mask, labels=None):
        # --- BƯỚC 1: ENCODE ---
        # Ảnh: (Batch, 197, 768) - Giữ nguyên không gian
        img_feat = self.image_encoder(pixel_values)
        
        # Câu hỏi: (Batch, Seq_Len, 768) - Lấy sequence, không lấy pooler
        # Lưu ý: TextEncoder cũ của bạn trả về pooler, cần sửa lại TextEncoder
        # Hoặc dùng luôn output của model bên dưới:
        txt_outputs = self.text_encoder.model(input_ids=question_ids, attention_mask=question_mask)
        txt_feat = txt_outputs.last_hidden_state # (Batch, Q_Len, 768)
        
        # --- BƯỚC 2: PROJECTION (Mapping) ---
        # Đưa tất cả về không gian vector của GPT
        img_embeds = self.img_projector(img_feat) # (Batch, 197, GPT_Hidden)
        img_embeds = self.ln_visual(img_embeds)   # Chuẩn hóa
        
        txt_embeds = self.txt_projector(txt_feat) # (Batch, Q_Len, GPT_Hidden)
        
        # --- BƯỚC 3: CONCATENATE (Nối chuỗi) ---
        # Input cho GPT sẽ là: [ẢNH] [CÂU HỎI] [CÂU TRẢ LỜI]
        # GPT sẽ nhìn thấy toàn bộ ảnh trước, rồi đến câu hỏi
        inputs_embeds = torch.cat((img_embeds, txt_embeds), dim=1) # (Batch, 197 + Q_Len, GPT_Hidden)

        img_mask = torch.ones((pixel_values.shape[0], 197), device=pixel_values.device)
        
        # Mask của câu hỏi: Lấy từ question_mask truyền vào (có số 0 ở chỗ padding)
        # Shape: (Batch, Seq_Len)
        
        # Nối Mask: [Mask Ảnh] + [Mask Câu hỏi]
        gpt_attention_mask = torch.cat((img_mask, question_mask), dim=1)
        
        # --- BƯỚC 4: DECODE (Training) ---
        if labels is not None:
            # Lấy embedding của câu trả lời thật
            # Lưu ý xử lý labels -100 như đã bàn trước đó
            decoder_input_ids = labels.clone()
            pad_id = self.decoder.config.pad_token_id if self.decoder.config.pad_token_id else 0
            decoder_input_ids[decoder_input_ids == -100] = pad_id
            
            ans_embeds = self.decoder.transformer.wte(decoder_input_ids)
            
            # Nối toàn bộ: [ẢNH + CÂU HỎI] + [CÂU TRẢ LỜI]
            full_inputs_embeds = torch.cat((inputs_embeds, ans_embeds), dim=1)
            
            # Tạo labels giả cho phần [ẢNH + CÂU HỎI] là -100 (không tính loss)
            context_len = inputs_embeds.shape[1] # 197 + Q_Len
            context_labels = torch.full((labels.shape[0], context_len), -100).to(labels.device)
            
            full_labels = torch.cat((context_labels, labels), dim=1)
            
            return self.decoder(inputs_embeds=full_inputs_embeds, attention_mask=gpt_attention_mask, labels=full_labels)
            
        else:
            return inputs_embeds # Trả về context để dùng cho hàm generate

    def generate_answer(self, pixel_values, question_ids, question_mask, max_length=20):
        # 1. Tạo Context (Ảnh + Câu hỏi)
        inputs_embeds = self.forward(pixel_values, question_ids, question_mask)
        
        # 2. Sinh từ
        output_ids = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_length,
            bos_token_id=self.decoder.config.bos_token_id,
            pad_token_id=self.decoder.config.pad_token_id,
            eos_token_id=self.decoder.config.eos_token_id,
            num_beams=3,
            repetition_penalty=1.2, # Mức phạt nhẹ nhàng
            no_repeat_ngram_size=0, # Tắt cái này đi nếu hay bị lỗi từ lạ
            early_stopping=True
        )
        return output_ids