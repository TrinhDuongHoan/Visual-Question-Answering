import re
import json
from collections import Counter
import torch
from underthesea import word_tokenize

def text_normalize_simple(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def vi_seg(s: str) -> list:
    s = text_normalize_simple(s)
    return word_tokenize(s, format="text").split()

class AnswerVocab:
    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.stoi = {}
        self.itos = []
        self.specials = ["<pad>", "<bos>", "<eos>", "<unk>"]
        
    def build(self, json_flat_path):
        with open(json_flat_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        counter = Counter()
        for item in data:
            if "answer" in item:
                ans = item["answer"]
                tokens = vi_seg(ans)
                counter.update(tokens)
            
        for sp in self.specials:
            self.stoi[sp] = len(self.itos)
            self.itos.append(sp)
            
        for tok, freq in counter.items():
            if freq >= self.min_freq and tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)
                
        print(f"Vocab size (answer): {len(self.itos)} (min_freq={self.min_freq})")
        
        self.PAD_ID = self.stoi["<pad>"]
        self.BOS_ID = self.stoi["<bos>"]
        self.EOS_ID = self.stoi["<eos>"]
        self.UNK_ID = self.stoi["<unk>"]
        
    def encode(self, text):
        tokens = vi_seg(text)
        ids = [self.BOS_ID] + [self.stoi.get(t, self.UNK_ID) for t in tokens] + [self.EOS_ID]
        return torch.tensor(ids, dtype=torch.long)
    
    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        tokens = []
        for i in ids:
            if i == self.EOS_ID:
                break
            if i in [self.PAD_ID, self.BOS_ID]:
                continue
            if 0 <= i < len(self.itos):
                tokens.append(self.itos[i])
        return " ".join(tokens)
        
    def __len__(self):
        return len(self.itos)