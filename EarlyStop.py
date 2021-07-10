import numpy as np
import torch
from callbacks import Callback

class EarlyStopping(Callback):
    def __init__(self, monitor, model_path, patience=5, mode="min", delta=0.001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.model_path = model_path
        self.monitor = monitor
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

        if self.monitor.startswith('train_'):
            self.model_state = 'train'
            self.monitor_value = self.monitor[len('train_'):]

        if self.monitor.startswith('valid_'):
            self.model_state = 'valid'
            self.monitor_value = self.monitor[len('valid_'):]

    def on_epoch_end(self, model):
        valid_loss = model.metrics[self.model_state][self.monitor_value]
        if self.mode == "min":
            score = -1.0 * valid_loss
        else:
            score = np.copy(valid_loss)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(valid_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                "EarlyStopping counter: {} out of {}".format(
                    self.counter, self.patience
                )
            )
            if self.counter >= self.patience:
                model.flag = True
        else:
            self.best_score = score
            self.save_checkpoint(valid_loss, model)
            self.counter = 0

    def save_checkpoint(self, valid_loss, model):
        if valid_loss not in [-np.inf, np.inf, -np.nan, np.nan]:
            print(
                "Validation score improved ({} --> {}). Saving model!".format(
                    self.val_score, valid_loss
                )
            )
            torch.save(model.state_dict(), self.model_path)
        self.val_score = valid_loss