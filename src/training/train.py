import os
import torch
import pandas as pd
from tqdm.auto import tqdm
from src.utils.metrics import compute_loss

def _move_to_device(batch, device):
    images = batch["images"].to(device)
    q_ids = batch["q_input_ids"].to(device)
    q_mask = batch["q_attention_mask"].to(device)
    ans_ids = batch["answer_ids"].to(device)
    return images, q_ids, q_mask, ans_ids

def _train_batch(model, batch, optimizer, cfg):
    images, q_ids, q_mask, ans_ids = _move_to_device(batch, cfg.DEVICE)
    optimizer.zero_grad()
    logits, targets = model(images, q_ids, q_mask, ans_ids)
    loss = compute_loss(logits, targets, model.vocab.PAD_ID)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()

def _eval_batch(model, batch, cfg):
    images, q_ids, q_mask, ans_ids = _move_to_device(batch, cfg.DEVICE)
    logits, targets = model(images, q_ids, q_mask, ans_ids)
    loss = compute_loss(logits, targets, model.vocab.PAD_ID)
    return loss.item()

def train_epoch(model, train_loader, optimizer, cfg, epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Train")
    for batch in pbar:
        loss = _train_batch(model, batch, optimizer, cfg)
        total_loss += loss
        pbar.set_postfix({"loss": loss})
    return total_loss / len(train_loader)

def eval_epoch(model, val_loader, cfg, epoch):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
            loss = _eval_batch(model, batch, cfg)
            total_loss += loss
    return total_loss / len(val_loader)

def _save_best_checkpoint(model, cfg):
    save_path = os.path.join(cfg.CHECKPOINT_DIR, "best_model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Saved best model to {save_path}")

def _should_stop(no_imp, cfg):
    return no_imp >= cfg.EARLY_STOP_PATIENCE

def train_model(model, train_loader, val_loader, optimizer, scheduler, cfg):
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    history = {"train_loss": [], "val_loss": []}

    best_loss = float('inf')

    no_imp = 0
    for epoch in range(cfg.NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, cfg, epoch)
        val_loss = eval_epoch(model, val_loader, cfg, epoch)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        print(f"Epoch {epoch+1}/{cfg.NUM_EPOCHS}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            _save_best_checkpoint(model, cfg)
            no_imp = 0
        else:
            no_imp += 1
            if _should_stop(no_imp, cfg):
                print("Early stopping triggered.")
                break
    return pd.DataFrame(history)