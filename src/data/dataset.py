import os
import json
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class VQADataset(Dataset):
    def __init__(
        self, 
        dataframe,  
        question_tokenizer, 
        answer_tokenizer, 
        transform=None, 
        max_question_len=64, 
        max_answer_len=32
    ):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame đã chứa cột 'image_path'.
            question_tokenizer: Tokenizer của PhoBERT (cho câu hỏi).
            answer_tokenizer: Tokenizer của GPT (cho câu trả lời).
            transform: Các phép biến đổi ảnh (Resize, Normalize...).
            max_question_len: Độ dài tối đa của câu hỏi.
            max_answer_len: Độ dài tối đa của câu trả lời.
        """
        self.data = dataframe
        self.q_tokenizer = question_tokenizer
        self.a_tokenizer = answer_tokenizer
        self.transform = transform
        self.max_question_len = max_question_len
        self.max_answer_len = max_answer_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        image_path = row['image_path'] 
        
        try:
            image = Image.open(image_path).convert("RGB")
        except (OSError, FileNotFoundError):
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            pixel_values = self.transform(image)
        else:
            pixel_values = image

        question_text = str(row['question'])
        q_encoding = self.q_tokenizer(
            question_text,
            max_length=self.max_question_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        answer_text = str(row['answer']) + " " + self.a_tokenizer.eos_token
        a_encoding = self.a_tokenizer(
            answer_text,
            max_length=self.max_answer_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        labels = a_encoding['input_ids'].squeeze()
        labels[labels == self.a_tokenizer.pad_token_id] = -100
        
        return {
            'pixel_values': pixel_values,
            'question_input_ids': q_encoding['input_ids'].squeeze(),
            'question_attention_mask': q_encoding['attention_mask'].squeeze(),
            'labels': labels
        }
    
class OpenViVQADataset(Dataset):
    def __init__(
        self, 
        json_path, 
        image_dir, 
        q_tokenizer, 
        a_tokenizer, 
        transform=None, 
        max_question_len=64, 
        max_answer_len=32
    ):

        print(f"Loading Kaggle OpenViVQA data from {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if isinstance(data, dict):
            # Thường dataset này gom hết vào key 'annotations' hoặc data nằm ngay root
            if 'annotations' in data:
                self.data = data['annotations']
            else:
                # Trường hợp khác, có thể phải duyệt keys để tìm list
                # Nhưng với dataset windyy261203, cấu trúc train.json là dict chứa keys image_id
                # Hãy cẩn thận: Dataset này format JSON hơi lạ so với gốc.
                pass 
                # Lát nữa ta sẽ check cấu trúc json thực tế ở cell bên dưới để chắc chắn
                # Tạm thời giả định nó là list chuẩn hoặc dict['annotations']
                self.data = list(data.values()) if not 'annotations' in data else data['annotations']
        else:
            self.data = data # Nếu là list thì dùng luôn
            
        self.image_dir = image_dir
        self.q_tokenizer = q_tokenizer
        self.a_tokenizer = a_tokenizer
        self.transform = transform
        self.max_question_len = max_question_len
        self.max_answer_len = max_answer_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]
        
        image_id = item.get('image_id')

        image_filename = f"{image_id}.jpg" 
        image_path = os.path.join(self.image_dir, image_filename)
        
        # 2. Load Ảnh
        try:
            image = Image.open(image_path).convert("RGB")
        except (OSError, FileNotFoundError):
            # print(f"Warning: Missing image {image_path}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            pixel_values = self.transform(image)
        else:
            pixel_values = image

        # 3. Tokenize Câu hỏi
        question_text = str(item['question'])
        q_encoding = self.q_tokenizer(
            question_text,
            max_length=self.max_question_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        # 4. Tokenize Câu trả lời
        answer_text = str(item['answer']) + " " + self.a_tokenizer.eos_token
        a_encoding = self.a_tokenizer(
            answer_text,
            max_length=self.max_answer_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        labels = a_encoding['input_ids'].squeeze()
        if self.a_tokenizer.pad_token_id is not None:
            labels[labels == self.a_tokenizer.pad_token_id] = -100
        
        return {
            'pixel_values': pixel_values,
            'question_input_ids': q_encoding['input_ids'].squeeze(),
            'question_attention_mask': q_encoding['attention_mask'].squeeze(),
            'labels': labels
        }