import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
from tqdm import tqdm
from math import isclose


@torch.inference_mode
def evaluate(model: nn.Module, 
             criterion: callable,
             dataloader: data.DataLoader,
             device: torch.device
            ) -> tuple[float, float]:
    
    model.eval()

    running_loss = 0
    running_corrects = 0 
    amount = 0

    for inputs, targets in (pbar := tqdm(dataloader, desc='Validation step')):
        
        inputs, targets = inputs.to(device), targets.to(device)
    
        logits = model(inputs)
        
        preds = torch.max(logits, dim=1)[1]
        loss = criterion(logits, targets)
            
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == targets)

        amount += inputs.size(0)

        val_loss = running_loss / amount
        val_accuracy = running_corrects.item() / amount

        pbar.set_description(f'Val Loss: {val_loss:.06f} | Val acc: {val_accuracy:.06f}')

    model.train()

    return val_loss, val_accuracy


def trainer(model: nn.Module,
            dataloaders: dict[str, data.DataLoader],
            device: torch.device,
            save_best: bool = False,
            epoch: int = 100,
            lr: float = 2e-3,
            grad_clip: int = 1
           ) -> tuple[list[int], list[int], list[int], list[int]]:
    
    train_acc_hist, train_loss_hist = [], []
    val_acc_hist, val_loss_hist = [], []

    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.AdamW(model.parameters(), lr = lr)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=3, min_lr=8e-5)

    model.train()
    model.to(device)

    last_lr = lr
    best_acc = 0

    for e in range(epoch):
        
        running_loss = 0
        running_corrects = 0 
        amount = 0

        for inputs, targets in (pbar := tqdm(dataloaders['train'], desc=f'Epoch {e+1}')):

            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()

            logits = model(inputs)
            
            preds = torch.max(logits, dim=1)[1]
            loss = criterion(logits, targets)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == targets)

            amount += inputs.size(0)

            train_loss = running_loss / amount
            train_accuracy = running_corrects.item() / amount

            pbar.set_description(f'Epoch {e+1} | Loss: {train_loss:.06f} | Acc: {train_accuracy:.06f}')

        val_loss, val_acc = evaluate(model, criterion, dataloaders['val'], device)
        
        scheduler.step(-val_acc)
        
        if not isclose(last_lr, scheduler.get_last_lr()[0]):
            last_lr = scheduler.get_last_lr()[0]
            tqdm.write(f'Epoch {e} | A Plateau has been reached. Reducing lr to {last_lr:.3e}')

        if val_acc > best_acc and save_best:
            best_acc = val_acc
            torch.save(model.state_dict(), f'params/params.pt')
            
        train_acc_hist.append(train_accuracy)
        train_loss_hist.append(train_loss)
        
        val_acc_hist.append(val_acc)
        val_loss_hist.append(val_loss)

        return train_acc_hist, train_loss_hist, val_acc_hist, val_loss_hist