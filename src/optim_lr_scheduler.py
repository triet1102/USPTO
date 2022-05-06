import torch
 
class WarmupLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            num_warmups,
            num_decreases
    ) -> None:
        self.min_lr = 1e-6
        self.peak_lr = optimizer.param_groups[0]["lr"]
        if num_warmups != 0:
            self.warmup_rate = (self.peak_lr - self.min_lr) / num_warmups
        else:
            self.warmup_rate = 0
        
        self.decrease_rate = (self.peak_lr - self.min_lr) / num_decreases
        
        self.num_warmups = num_warmups
        self.num_decreases = num_decreases
        self.update_warmup_step = 1
        self.update_decrease_step = 1
        self.lr = self.min_lr
        super().__init__(optimizer)

    def set_lr(self, optimizer, lr):
        optimizer.param_groups[0]["lr"]=lr

    def step(self, val_loss = None):
        print(self.lr)
        if self.update_warmup_step <= self.num_warmups:
            lr = self.min_lr + self.warmup_rate * self.update_warmup_step
            self.set_lr(self.optimizer, lr)
            self.lr = lr
            self.update_warmup_step += 1
        else:
            lr = self.min_lr + self.decrease_rate * (self.num_decreases - self.update_decrease_step)
            self.set_lr(self.optimizer, lr)
            self.lr = lr
            self.update_decrease_step += 1
        
        return self.lr