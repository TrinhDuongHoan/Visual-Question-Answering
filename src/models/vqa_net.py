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
        
        self.image_encoder = ImageEncoder(vit_name, freeze=False)
        self.text_encoder = TextEncoder(phobert_name, freeze=False)
        self.decoder = GPT2LMHeadModel.from_pretrained(gpt_name)
        
        self.img_hidden = self.image_encoder.model.config.hidden_size
        self.txt_hidden = self.text_encoder.model.config.hidden_size
        self.gpt_hidden = self.decoder.config.n_embd
        
        self.img_projector = nn.Linear(self.img_hidden, self.gpt_hidden)
        self.txt_projector = nn.Linear(self.txt_hidden, self.gpt_hidden)
        self.ln_visual = nn.LayerNorm(self.gpt_hidden)

    def forward(self, pixel_values, question_ids, question_mask, labels=None):
        img_feat = self.image_encoder(pixel_values)
        
        txt_outputs = self.text_encoder.model(input_ids=question_ids, attention_mask=question_mask)
        txt_feat = txt_outputs.last_hidden_state
        
        img_embeds = self.img_projector(img_feat)
        img_embeds = self.ln_visual(img_embeds)
        txt_embeds = self.txt_projector(txt_feat)
        
        if labels is not None:
            decoder_input_ids = labels.clone()
            pad_id = self.decoder.config.pad_token_id if self.decoder.config.pad_token_id is not None else 0
            decoder_input_ids[decoder_input_ids == -100] = pad_id
            
            ans_embeds = self.decoder.transformer.wte(decoder_input_ids)
            
            full_inputs_embeds = torch.cat((img_embeds, txt_embeds, ans_embeds), dim=1)
            
            img_mask = torch.ones((pixel_values.shape[0], img_embeds.shape[1]), device=pixel_values.device)
            ans_mask = torch.ones((labels.shape[0], ans_embeds.shape[1]), device=labels.device)
            full_attention_mask = torch.cat((img_mask, question_mask, ans_mask), dim=1)
            
            context_len = img_embeds.shape[1] + txt_embeds.shape[1]
            context_labels = torch.full((labels.shape[0], context_len), -100).to(labels.device)
            full_labels = torch.cat((context_labels, labels), dim=1)
            
            return self.decoder(
                inputs_embeds=full_inputs_embeds, 
                attention_mask=full_attention_mask,
                labels=full_labels
            )
            
        else:
            inputs_embeds = torch.cat((img_embeds, txt_embeds), dim=1)
            
            img_mask = torch.ones((pixel_values.shape[0], img_embeds.shape[1]), device=pixel_values.device)
            gpt_attention_mask = torch.cat((img_mask, question_mask), dim=1)
            
            return inputs_embeds, gpt_attention_mask

    def generate_answer(self, pixel_values, question_ids, question_mask, max_length=20):
        inputs_embeds, attention_mask = self.forward(pixel_values, question_ids, question_mask)
        
        output_ids = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            bos_token_id=self.decoder.config.bos_token_id,
            pad_token_id=self.decoder.config.pad_token_id,
            eos_token_id=self.decoder.config.eos_token_id,
            num_beams=3,
            repetition_penalty=1.2,
            no_repeat_ngram_size=0,
            early_stopping=True
        )
        return output_ids