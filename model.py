import torch
import torch.nn as nn
import callbacks
import utility
from tqdm import tqdm
from torch.cuda import amp

class Tesseract(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = None
        self.scheduler = None
        self.flag = False
        self.fp16 = False
        self.scaler = None
     
    def forward(self, *args, **kwargs):
        super().forward(*args, **kwargs)
        
    def fetch_optimizer(self, *args, **kwargs):
        return
        
    def train_one_epoch(self, dataloader):
        self.train()
        losses = utility.AverageMeter()
        acc = utility.AverageMeter()
        tk0 = tqdm(dataloader, total=len(dataloader))
        for idx, data in enumerate(tk0):
            logit, loss, accuracy = self.train_one_step(data)
            losses.update(loss.item(), dataloader.batch_size)
            acc.update(accuracy.item())
            tk0.set_postfix(loss=losses.avg, metric=acc.avg, stage='train')
        tk0.close()
        return losses.avg
    
    def train_one_step(self, data):
        self.optimizer.zero_grad()
        for k, v in data.items():
            data[k] = v.to('cuda')
        with torch.set_grad_enabled(True):
            if self.fp16:
                with amp.autocast():    
                    logit, loss, accuracy = self(**data)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                logit, loss, accuracy = self(**data)
                loss.backward()
                self.optimizer.step()
        return logit, loss, accuracy
    
    def validate_one_epoch(self, dataloader):
        self.eval()
        losses = utility.AverageMeter()
        acc = utility.AverageMeter()
        tk0 = tqdm(dataloader, total=len(dataloader))
        for idx, data in enumerate(tk0):
            logit, loss, accuracy = self.validate_one_step(data)
            losses.update(loss.item(), dataloader.batch_size)
            acc.update(accuracy.item())
            tk0.set_postfix(loss=losses.avg, metric=acc.avg, stage='valid')
        tk0.close()
        return losses.avg
    
    def validate_one_step(self, data):
        for k, v in data.items():
            data[k] = v.to('cuda')
        logit, loss, accuracy = self(**data)
        return logit, loss, accuracy
    
    def predict(self, dataset, batch_size, device):
        self.eval()
        if next(self.parameters()).device!=device:
            self.to(device)
        predictions = []
        dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4
        )
        
        tk0 = tqdm(dataloader, total=len(dataloader))
        for idx, data in enumerate(tk0):
            logit = self.predict_one_step(data)
            predicted = logit.cpu().numpy()
            predictions.append(predicted)
        tk0.close()
        return np.concatenate(predictions)
    
    def predict_one_step(self, data):
        for k, v in data.items():
            data[k] = v.to('cuda')
        with torch.no_grad():        
            logit, _, _ = self(**data)
        return logit
    
    
    def fit(
        self, 
        train_dataset,
        valid_dataset,
        train_bs=8,
        valid_bs=8,
        epochs=10,
        callback=None,
        fp16=False,
        device='cpu',
        workers = 1
    ):
        train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=True,
        num_workers=workers,
        drop_last=True
        )
        valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=valid_bs,
        shuffle=False,
        num_workers=workers,
        drop_last=True
        )
        self.optimizer = self.fetch_optimizer()
        
        if self.scheduler is None:
            self.scheduler = self.fetch_scheduler()
        
        if next(self.parameters()).device!=device:
            self.to(device)
            
        caller = callbacks.CallbackRunner(self, callback) 
        
        self.fp16 = fp16
        
        if self.fp16:
            self.scaler = amp.GradScaler()
        
        for epoch in range(epochs):
            train_loss = self.train_one_epoch(train_loader)
            valid_loss = self.validate_one_epoch(valid_loader)
            
            if self.scheduler is None:
                self.scheduler.step()
            caller('on_epoch_end',  valid_loss)
            if self.flag:
                break