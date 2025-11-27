import torch
from tqdm.auto import tqdm
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim
import pandas as pd

def get_optimizer_scheduler(model, dataloader, epochs, lr=1e-4):

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    num_training_steps = len(dataloader) * epochs
    num_warmup_steps = int(num_training_steps * 0.1) 

    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )
    
    return optimizer, scheduler


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device):

    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch in progress_bar:
        pixel_values = batch['pixel_values'].to(device)
        q_ids = batch['question_input_ids'].to(device)
        q_mask = batch['question_attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(pixel_values, q_ids, q_mask, labels=labels)
        
        targets = labels[:, 1:]
        
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        loss.backward()
        optimizer.step()

        scheduler.step()
            
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
        
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):

    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validating", leave=False)
        
        for batch in progress_bar:
            pixel_values = batch['pixel_values'].to(device)
            q_ids = batch['question_input_ids'].to(device)
            q_mask = batch['question_attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(pixel_values, q_ids, q_mask, labels=labels)
            
            targets = labels[:, 1:]
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, checkpoint_path):

    patience = 3
    bad_epoch = 0

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bad_epoch = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, checkpoint_path)
            print("🔥 New best model saved!")
        else:
            bad_epoch += 1
            if bad_epoch >= patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break
            
    return pd.DataFrame(history)


def load_checkpoint(model, optimizer, filename, device):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint: {filename}")
    return model

