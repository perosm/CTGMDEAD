class EarlyStopping:
    """Implements early stopping to terminate training when validation loss stops improving.

    Monitors validation loss and stops training if no improvement is observed for a given
    number of consecutive epochs (patience). Improvement is defined as a reduction in loss
    greater than the specified minimum delta threshold.

    Attributes:
        patience (int): Number of epochs to wait before stopping after no improvement.
        min_delta (float): Minimum change in monitored quantity to qualify as improvement.
        best_loss (float): Best validation loss observed during training.
        counter (int): Current count of epochs without improvement.
        early_stop (bool): Flag indicating whether to stop training.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.01):
        """Initialize early stopping tracker.

        Args:
            patience: Number of epochs to wait after last improvement before stopping.
            min_delta: Minimum absolute change in loss to qualify as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss: float) -> None:
        """
        Check if training should be stopped based on validation loss.

        Updates internal state including:
        - counter of epochs without improvement
        - early_stop flag if patience is exceeded
        Resets counter if improvement is detected.

        Args:
            val_loss: Current epoch's validation loss.
        """

        diff = val_loss - self.best_val_loss * (1 - self.min_delta)
        if diff > 0:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0
            self.best_val_loss = val_loss
