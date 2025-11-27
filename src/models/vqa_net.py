import torch
import torch.nn as nn
from src.models.image_encoder import ImageEncoder
from src.models.text_encoder import TextEncoder
from src.models.decoder import LSTMDecoder

class VQANet(nn.Module):
    def __init__(self, vocab_size, 
                 vit_name="google/vit-base-patch16-224", 
                 phobert_name="vinai/phobert-base",
                 embed_dim=256,
                 hidden_dim=512):
        super(VQANet, self).__init__()
        
        # 1. Khởi tạo các Module con
        self.image_encoder = ImageEncoder(vit_name, freeze=False)
        self.text_encoder = TextEncoder(phobert_name, freeze=False)
        self.decoder = LSTMDecoder(vocab_size, embed_dim, hidden_dim)
        
        # Lấy kích thước vector encoder
        img_dim = self.image_encoder.model.config.hidden_size # 768
        txt_dim = self.text_encoder.model.config.hidden_size  # 768
        
        # 2. Fusion Layer (Bộ trộn)
        # Chuyển đổi [Ảnh + Hỏi] thành trạng thái khởi đầu (h0, c0) cho LSTM
        self.project_h = nn.Linear(img_dim + txt_dim, hidden_dim)
        self.project_c = nn.Linear(img_dim + txt_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, pixel_values, question_ids, question_mask, labels=None):
        # 1. Encode
        img_feat = self.image_encoder(pixel_values)
        txt_feat = self.text_encoder(question_ids, question_mask)
        
        # 2. Fuse (Tạo trí nhớ ban đầu)
        combined = torch.cat((img_feat, txt_feat), dim=1)
        combined = self.dropout(combined)
        
        # Tạo h0, c0 (Kích thước: 1, Batch, Hidden) cho LSTM
        h0 = torch.tanh(self.project_h(combined)).unsqueeze(0)
        c0 = torch.tanh(self.project_c(combined)).unsqueeze(0)
        init_states = (h0, c0)
        
        # 3. Decode
        if labels is not None:
            # Xử lý labels cho embedding (thay -100 bằng 0)
            captions_in = labels.clone()
            captions_in[captions_in == -100] = 0
            
            # Truyền vào Decoder
            logits = self.decoder(captions_in, init_states)
            
            # Tính Loss
            loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            return logits, loss
        else:
            # Trả về init_states để dùng cho hàm generate
            return init_states

    def generate_answer(self, pixel_values, question_ids, question_mask, start_token_id, max_length=20):
        # 1. Lấy trạng thái khởi đầu
        init_states = self.forward(pixel_values, question_ids, question_mask)
        
        # 2. Gọi Decoder sinh từ
        output_ids = self.decoder.generate(init_states, start_token_id, max_length)
        
        return output_ids