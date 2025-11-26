import torch
from tqdm.auto import tqdm
import torch.nn as nn

def train_epoch(model, dataloader, optimizer, device, scheduler=None):
   
    model.train() 
    total_loss = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch in progress_bar:
        pixel_values = batch['pixel_values'].to(device)
        q_ids = batch['question_input_ids'].to(device)
        q_mask = batch['question_attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
    
        outputs = model(pixel_values, q_ids, q_mask, labels=labels)

        loss = outputs.loss
        loss.backward()
        
        optimizer.step()

        if scheduler:
            scheduler.step()
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
        
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def validate(model, dataloader, device):
    model.eval() 
    total_loss = 0
    
    with torch.no_grad(): 
        progress_bar = tqdm(dataloader, desc="Validating", leave=False)
        
        for batch in progress_bar:
            pixel_values = batch['pixel_values'].to(device)
            q_ids = batch['question_input_ids'].to(device)
            q_mask = batch['question_attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(pixel_values, q_ids, q_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def save_checkpoint(model, optimizer, epoch, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    print(f"Saved checkpoint: {filename}")