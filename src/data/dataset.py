import json
import torch
import os
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from src.data.tokenizer import text_normalize_simple

class VQADataset(Dataset):
    def __init__(self, json_flat_path, tokenizer, transform, vocab=None, has_answer=True, max_q_len=64):
        with open(json_flat_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer # PhoBERT tokenizer
        self.transform = transform
        self.vocab = vocab # AnswerVocab
        self.has_answer = has_answer
        self.max_q_len = max_q_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item["image_path"]
        question = item["question"]
        
        try:
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
        except Exception as e:
            # Fallback hình đen nếu lỗi load ảnh
            img = torch.zeros((3, 224, 224))

        # Encode question (PhoBERT)
        encoded = self.tokenizer(
            text_normalize_simple(question),
            max_length=self.max_q_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        q_input_ids = encoded["input_ids"].squeeze(0)
        q_attention = encoded["attention_mask"].squeeze(0)
        
        sample = {
            "image": img,
            "question": question,
            "q_input_ids": q_input_ids,
            "q_attention_mask": q_attention
        }
        
        if self.has_answer and self.vocab:
            answer = item["answer"]
            ans_ids = self.vocab.encode(answer)
            sample["answer"] = answer
            sample["answer_ids"] = ans_ids
        
        return sample

def vqa_collate_fn(batch, pad_id):
    images = torch.stack([b["image"] for b in batch], dim=0)
    q_input_ids = torch.stack([b["q_input_ids"] for b in batch], dim=0)
    q_attention = torch.stack([b["q_attention_mask"] for b in batch], dim=0)
    
    out = {
        "images": images,
        "q_input_ids": q_input_ids,
        "q_attention_mask": q_attention,
        "questions": [b["question"] for b in batch],
    }
    
    if "answer_ids" in batch[0]:
        ans_seqs = [b["answer_ids"] for b in batch]
        ans_padded = pad_sequence(ans_seqs, batch_first=True, padding_value=pad_id)
        out["answer_ids"] = ans_padded
        out["answers"] = [b["answer"] for b in batch]
    
    return out