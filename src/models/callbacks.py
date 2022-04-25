import wandb
from tensorflow.python.keras.callbacks import Callback


class LRLogger(Callback):
    """
    Callback for learn rate logging
    """

    def __init__(self, optimizer):
        super(LRLogger, self).__init__()
        self.optimizer = optimizer

    def on_epoch_end(self, epoch, logs):
        lr = self.optimizer.learning_rate(self.optimizer.iterations)
        wandb.log({"lr": lr}, commit=False)
