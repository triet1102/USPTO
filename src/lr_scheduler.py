import torch
 
class WarmupLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            num_warmups,
            num_decrease
    ) -> None:
        super().__init__(optimizer)
        self.min_lr = 1e-6
        self.peak_lr = optimizer.param_groups[0]["lr"]
        if num_warmups != 0:
            self.warmup_rate = (self.peak_lr - self.min_lr) / num_warmups
        else:
            self.warmup_rate = 0
        
        self.decrease_rate = (self.peak_lr - self.min_lr) / num_decrease

        self.update_lr_step = 0

        self.lr = self.min_lr

    def set_lr(self, optimizer, lr):
        optimizer.param_groups[0]["lr"]=lr

    def step(self, val_loss = None):
        print(self.lr)
        if self.update_lr_step < self.warmup_steps:
            lr = self.init_lr + self.warmup_rate * (self.update_steps + 1)
            self.set_lr(self.optimizer, lr)
            self.lr = lr
            self.update_lr_step += 1
        else:
            lr = self.
        
        return self.lr