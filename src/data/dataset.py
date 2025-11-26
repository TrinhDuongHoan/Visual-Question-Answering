import os
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
        
        # 1. XỬ LÝ ẢNH (Dùng thẳng cột image_path)
        image_path = row['image_path'] 
        
        try:
            image = Image.open(image_path).convert("RGB")
        except (OSError, FileNotFoundError):
            # Fallback: Tạo ảnh đen nếu đường dẫn sai hoặc ảnh lỗi
            # print(f"Warning: Could not read image at {image_path}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            pixel_values = self.transform(image)
        else:
            pixel_values = image

        # 2. XỬ LÝ CÂU HỎI
        question_text = str(row['question'])
        q_encoding = self.q_tokenizer(
            question_text,
            max_length=self.max_question_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        # 3. XỬ LÝ CÂU TRẢ LỜI
        answer_text = str(row['answer'])
        a_encoding = self.a_tokenizer(
            answer_text,
            max_length=self.max_answer_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'pixel_values': pixel_values,
            'question_input_ids': q_encoding['input_ids'].squeeze(),
            'question_attention_mask': q_encoding['attention_mask'].squeeze(),
            'labels': a_encoding['input_ids'].squeeze()
        }