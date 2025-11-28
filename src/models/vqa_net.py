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
                 hidden_dim=512,
                 freeze_encoder=True):
        super(VQANet, self).__init__()
        
        self.image_encoder = ImageEncoder(vit_name, freeze=freeze_encoder)
        self.text_encoder = TextEncoder(phobert_name, freeze=freeze_encoder)
        self.decoder = LSTMDecoder(vocab_size, embed_dim, hidden_dim)
        
        img_dim = self.image_encoder.model.config.hidden_size
        txt_dim = self.text_encoder.model.config.hidden_size
        
        self.project_h = nn.Linear(img_dim + txt_dim, hidden_dim)
        self.project_c = nn.Linear(img_dim + txt_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, pixel_values, question_ids, question_mask, labels=None):
        img_feat = self.image_encoder(pixel_values)
        txt_feat = self.text_encoder(question_ids, question_mask)
        
        combined = torch.cat((img_feat, txt_feat), dim=1)
        combined = self.dropout(combined)
        
        h0 = torch.tanh(self.project_h(combined)).unsqueeze(0)
        c0 = torch.tanh(self.project_c(combined)).unsqueeze(0)
        init_states = (h0, c0)
        
        if labels is not None:
            decoder_input_ids = labels[:, :-1].clone()
            decoder_input_ids[decoder_input_ids == -100] = 0
            
            logits = self.decoder(decoder_input_ids, init_states)
            return logits
        else:
            return init_states

    def generate_answer(self, pixel_values, question_ids, question_mask, start_token_id, max_length=20):
        init_states = self.forward(pixel_values, question_ids, question_mask)
        output_ids = self.decoder.generate(init_states, start_token_id, max_length)
        return output_ids