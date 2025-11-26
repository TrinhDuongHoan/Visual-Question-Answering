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

        self.image_encoder = ImageEncoder(vit_name)
        self.text_encoder = TextEncoder(phobert_name)
        
        print(f"Loading Decoder: {gpt_name}...")
        self.decoder = GPT2LMHeadModel.from_pretrained(gpt_name)
        
        self.img_hidden = self.image_encoder.model.config.hidden_size # 768
        self.txt_hidden = self.text_encoder.model.config.hidden_size  # 768
        self.gpt_hidden = self.decoder.config.n_embd                  # 768 hoặc 1024 tùy model
        
        self.fusion = nn.Sequential(
            nn.Linear(self.img_hidden + self.txt_hidden, self.gpt_hidden),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, pixel_values, question_ids, question_mask, labels=None):
        """
        pixel_values: Ảnh
        question_ids: Input IDs câu hỏi
        labels: Input IDs câu trả lời (Target)
        """
        
        img_feat = self.image_encoder(pixel_values)           # (Batch, 768)
        txt_feat = self.text_encoder(question_ids, question_mask) # (Batch, 768)
  
        # Nối 2 vector lại thành (Batch, 1536)
        concat_feat = torch.cat((img_feat, txt_feat), dim=1)
        
        # Chiếu về không gian GPT (Batch, 768) -> Biến đổi thành (Batch, 1, 768) để giả lập 1 token
        fused_embeds = self.fusion(concat_feat).unsqueeze(1) 
        
        # --- BƯỚC 3: DECODE (GPT) ---
        if labels is not None:
            # Training Mode
            
            decoder_input_ids = labels.clone()

            pad_token_id = self.decoder.config.pad_token_id if self.decoder.config.pad_token_id is not None else 0

            decoder_input_ids[decoder_input_ids == -100] = pad_token_id

            # Lấy embedding của câu trả lời thật từ GPT
            # inputs_embeds của GPT nhận vào vector chứ không nhận ID
            answer_embeds = self.decoder.transformer.wte(decoder_input_ids) # (Batch, Seq_Len, 768)
            
            # Nối vector Fused vào TRƯỚC vector câu trả lời
            # Tưởng tượng: [FUSED_INFO] + [Câu trả lời]
            full_inputs_embeds = torch.cat((fused_embeds, answer_embeds), dim=1)
            
            # Gọi GPT. Lưu ý: labels cần dịch chuyển hoặc xử lý padding nếu cần kỹ hơn.
            # Nhưng GPT2LMHeadModel tự động shift labels bên trong để tính loss.
            # Ta cần tạo labels giả cho phần fused_embeds (là -100 để không tính loss cho phần này)
            
            fused_labels = torch.full((labels.shape[0], 1), -100).to(labels.device)
            full_labels = torch.cat((fused_labels, labels), dim=1)

            outputs = self.decoder(inputs_embeds=full_inputs_embeds, labels=full_labels)
            return outputs # Chứa loss và logits
            
        else:
            # Inference Mode (Sinh câu trả lời) sẽ xử lý sau
            return fused_embeds

    def generate_answer(self, pixel_values, question_ids, question_mask, max_length=20):
        """
        Hàm dùng để sinh câu trả lời từ ảnh và câu hỏi (Inference).
        """

        img_feat = self.image_encoder(pixel_values)
        txt_feat = self.text_encoder(question_ids, question_mask)
        concat_feat = torch.cat((img_feat, txt_feat), dim=1)
        
        fused_embeds = self.fusion(concat_feat).unsqueeze(1) 
        
        output_ids = self.decoder.generate(
            inputs_embeds=fused_embeds,
            max_new_tokens=max_length, 
            bos_token_id=self.decoder.config.bos_token_id,
            pad_token_id=self.decoder.config.pad_token_id,
            eos_token_id=self.decoder.config.eos_token_id,
            num_beams=3, 
            repetition_penalty=1.1,
            no_repeat_ngram_size=0,
            early_stopping=True
        )
        
        return output_ids