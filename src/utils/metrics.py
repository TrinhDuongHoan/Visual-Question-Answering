import torch
import pandas as pd
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from src.data.tokenizer import vi_seg

def compute_model_size(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params / 1e6:.2f}M")
    print(f"Total params:     {total_params / 1e6:.2f}M")
    
    return trainable_params, total_params


def compute_loss(logits, targets, pad_id, smoothing=0.1):
    B, L, V = logits.size()
    logits = logits.reshape(-1, V)
    targets = targets.reshape(-1)
    
    mask = (targets != pad_id)
    if mask.sum() == 0: 
        return torch.tensor(0.0, device=logits.device)
    
    logits = logits[mask]
    targets = targets[mask]
    
    with torch.no_grad():
        true_dist = torch.zeros_like(logits)
        true_dist.fill_(smoothing / (V - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
        
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(true_dist * log_probs).sum(dim=-1).mean()
    return loss

def evaluate_metrics(refs, hyps):
    smooth_fn = SmoothingFunction().method1
    rouge_scorer_obj = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    
    bleu1, bleu2, bleu3, bleu4, meteor, rougeL = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    n = len(refs)
    
    for r, h in zip(refs, hyps):
        ref_tokens = vi_seg(r)
        hyp_tokens = vi_seg(h)
        if not hyp_tokens: continue
            
        bleu1 += sentence_bleu([ref_tokens], hyp_tokens, weights=(1,0,0,0), smoothing_function=smooth_fn)
        bleu2 += sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5,0.5,0,0), smoothing_function=smooth_fn)
        bleu3 += sentence_bleu([ref_tokens], hyp_tokens, weights=(0.33,0.33,0.33,0), smoothing_function=smooth_fn)
        bleu4 += sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth_fn)
        meteor += meteor_score([ref_tokens], hyp_tokens) 
        rougeL += rouge_scorer_obj.score(" ".join(ref_tokens), " ".join(hyp_tokens))["rougeL"].fmeasure
        
    return pd.DataFrame.from_dict({
        "BLEU-1": bleu1 / n,
        "BLEU-2": bleu2 / n,
        "BLEU-3": bleu3 / n,
        "BLEU-4": bleu4 / n,
        "METEOR": meteor / n,
        "ROUGE-L": rougeL / n
    })