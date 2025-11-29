import json
import torch
import os
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from src.data.tokenizer import text_normalize_simple

class VQADataset(Dataset):
    def __init__(self, json_flat_path, tokenizer, transform, vocab=None, has_answer=True, max_q_len=128, ocr_cache_path=None):
        with open(json_flat_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
            
        # [NEW] Load OCR Cache
        self.ocr_data = {}
        if ocr_cache_path and os.path.exists(ocr_cache_path):
            print(f"Loading OCR cache from {ocr_cache_path}...")
            with open(ocr_cache_path, "r", encoding="utf-8") as f:
                self.ocr_data = json.load(f)
        
        self.tokenizer = tokenizer
        self.transform = transform
        self.vocab = vocab
        self.has_answer = has_answer
        self.max_q_len = max_q_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item["image_path"]
        question = item["question"]
        
        # [NEW] Lấy nội dung OCR
        ocr_text = self.ocr_data.get(img_path, "")
        
        # [NEW] Cải tiến prompt cho PhoBERT
        # Input sẽ là: "câu hỏi ? [SEP] ngữ cảnh: nội dung ocr"
        # Việc này giúp PhoBERT hiểu rằng phần sau là thông tin bổ trợ
        final_input_text = f"{text_normalize_simple(question)} {self.tokenizer.sep_token} ngữ cảnh: {text_normalize_simple(ocr_text)}"
        
        try:
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
        except Exception as e:
            img = torch.zeros((3, 224, 224))

        # Encode question + OCR
        encoded = self.tokenizer(
            final_input_text, # Dùng text đã ghép
            max_length=self.max_q_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        q_input_ids = encoded["input_ids"].squeeze(0)
        q_attention = encoded["attention_mask"].squeeze(0)
        
        sample = {
            "image": img,
            "question": question, # Giữ nguyên câu hỏi gốc để hiển thị
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