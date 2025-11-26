import torch
import evaluate
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def compute_model_size(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params / 1e6:.2f}M")
    print(f"Total params:     {total_params / 1e6:.2f}M")
    
    return trainable_params, total_params

def compute_bleu(predictions, references):

    preds_tokenized = [p.split() for p in predictions]
    refs_tokenized = [[r.split()] for r in references] 

    chencherry = SmoothingFunction()

    bleu1 = corpus_bleu(refs_tokenized, preds_tokenized, weights=(1.0, 0, 0, 0), smoothing_function=chencherry.method1)
    bleu2 = corpus_bleu(refs_tokenized, preds_tokenized, weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry.method1)
    bleu3 = corpus_bleu(refs_tokenized, preds_tokenized, weights=(0.33, 0.33, 0.33, 0), smoothing_function=chencherry.method1)
    bleu4 = corpus_bleu(refs_tokenized, preds_tokenized, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)

    return {
        "bleu1": round(bleu1, 4),
        "bleu2": round(bleu2, 4),
        "bleu3": round(bleu3, 4),
        "bleu4": round(bleu4, 4)
    }

def compute_meteor(predictions, references):
    meteor = evaluate.load("meteor")
    results = meteor.compute(predictions=predictions, references=references)
    return round(results["meteor"], 4)

def compute_rouge(predictions, references):
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=predictions, references=references)
    return {
        "rouge1": round(results["rouge1"], 4),
        "rougeL": round(results["rougeL"], 4)
    }