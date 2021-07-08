import os
import random
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import DogDataset
import model
import torchvision
from torch.nn import functional as F
from torch.cuda import amp
from efficientnet_pytorch import EfficientNet
import albumentations as A
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=42)
     
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
p=0.5

class EarlyStopping:
    def __init__(self, early_stop=False, patience=0, model_path='model.pth'):
        self.best_loss = np.inf
        self.early_stop = early_stop
        self.counter = 0
        self.patience = patience
        self.model_path = model_path
        
    def on_epoch_end(self, valid_loss, model):
        if valid_loss < self.best_loss:
            print(
                "Validation score improved ({} --> {}). Saving model!".format(
                    self.best_loss, valid_loss
                        )
                )
            self.best_loss = valid_loss
            torch.save(model.state_dict(), self.model_path)
        elif(self.early_stop == True):
            self.counter += 1
            print(
                "EarlyStopping counter: {} out of {}".format(
                    self.counter, self.patience
                )
            )
        if(self.counter >= self.patience):
            model.flag = True
    
class MyModel(model.Model):
     def __init__(self, num_class, pretrained=False):
         super().__init__() 
         self.backbone = torchvision.models.resnet18(pretrained=pretrained)    
         in_features = self.backbone.fc.in_features
         self.out = nn.Linear(in_features, num_class)
         
     def loss_fn(self, outputs, targets):
         loss = nn.CrossEntropyLoss()(outputs, targets)
         return loss
     
     def metrics_fn(self, outputs, targets):
         outputs = torch.argmax(outputs, axis=1).cpu().detach().numpy()
         targets = targets.cpu().detach().numpy()
         acc = accuracy_score(targets, outputs)
         return acc
     
     def fetch_optimizer(self):
         opt = torch.optim.Adam(self.parameters(), lr=1e-4)
         return opt
     
     def fetch_scheduler(self):
         sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
             self.optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
         )
         return sch
         
     def forward(self, image, targets=None):
         batch_size, C, H, W = image.shape
         x = self.backbone.conv1(image)
         x = self.backbone.bn1(x)
         x = self.backbone.relu(x)
         x = self.backbone.maxpool(x)
  
         x = self.backbone.layer1(x)
         x = self.backbone.layer2(x)
         x = self.backbone.layer3(x)
         x = self.backbone.layer4(x)
         
         x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
         x = self.out(x)
         
         loss=None
         
         if targets is not None:
             loss = self.loss_fn(x, targets)
             accuracy = self.metrics_fn(x, targets)
             return x, loss, accuracy
         return x, None, None
     


if __name__=='__main__':
    df = pd.read_csv("D:\\Dataset\\DOGvsCAT\\train_folds.csv")
    df_train=df.loc[df.kfold!=0].reset_index(drop=True)
    df_valid=df.loc[df.kfold==0].reset_index(drop=True)
    
    train_images = df_train.images.values.tolist()
    train_images = [i for i in train_images]
    train_targets = df_train.targets.values
    
    valid_images = df_valid.images.values.tolist()
    valid_images = [i for i in valid_images]
    valid_targets = df_valid.targets.values
    
    train_aug = A.Compose(
            [
            A.CenterCrop(224,224),
            A.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),  
            ]
            
        )
    
    valid_aug = A.Compose(
            [
                A.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
            ]
        )
    
    
    train_dataset = DogDataset.ClassificationDataset(
            image_paths=train_images,
            targets=train_targets,
            resize=(256,256),
            augmentations=train_aug
        )
    
    valid_dataset = DogDataset.ClassificationDataset(
            image_paths=valid_images,
            targets=valid_targets,
            resize=(256,256),
            augmentations=train_aug
        )
    
    es = EarlyStopping(early_stop=True, patience=5, model_path=f'model.bin')
    NUM_CLASS=2
    modl = MyModel(NUM_CLASS, pretrained=True)
    modl.fit(train_dataset, valid_dataset, train_bs=16, valid_bs=16, epochs=10, callback=[es], fp16=True, device='cuda')