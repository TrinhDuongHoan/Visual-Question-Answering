import os
from transformers import AutoTokenizer

def load_tokenizers(encoder_model_name="vinai/phobert-base", decoder_model_name="minhtoan/vietnamese-gpt2-finetune"):
    print(f"Loading tokenizers: {encoder_model_name} & {decoder_model_name}...")

    q_tokenizer = AutoTokenizer.from_pretrained(encoder_model_name)
    a_tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)

    if a_tokenizer.pad_token is None:
        a_tokenizer.pad_token = a_tokenizer.eos_token
        a_tokenizer.pad_token_id = a_tokenizer.eos_token_id

    print("Tokenizers loaded successfully!")
    return q_tokenizer, a_tokenizer