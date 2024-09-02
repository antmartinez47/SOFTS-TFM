import numpy as np
import torch

class EarlyStoppingCallback:
    """
    Implements early stopping mechanism for model training.

    This callback monitors the validation loss and stops training when the loss
    doesn't improve for a specified number of epochs.

    Attributes:
        patience (int): Number of epochs to wait for improvement before stopping.
        verbose (bool): If True, prints messages about early stopping status.
        counter (int): Counts the number of epochs without improvement.
        best_score (float): The best validation loss observed.
        early_stop (bool): Flag indicating whether to stop training.
        delta (float): Minimum change in monitored quantity to qualify as improvement.

    Note:
        This callback is crucial for preventing overfitting and is part of the
        training procedure described in Section 4.2 of the paper.
    """

    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = np.inf
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        """
        Determines whether to stop training based on validation loss.

        Args:
            val_loss (float): The current validation loss.

        This method updates the early stopping state and potentially sets
        the early_stop flag to True if the stopping criterion is met.
        """
        if val_loss >= self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

class ModelCheckpointCallback:
    """
    Saves the best model based on validation loss.

    This callback saves the model weights when the validation loss improves.

    Attributes:
        verbose (bool): If True, prints messages about saving the model.
        best_score (float): The best validation loss observed.
        delta (float): Minimum change in monitored quantity to qualify as improvement.
        path (str): Path to save the model weights.
        tune (bool): If True, enables additional functionality for hyperparameter tuning.

    Note:
        This callback is essential for preserving the best model during training,
        as mentioned in the experimental setup in Section 4.2 of the paper.
    """

    def __init__(self, path, delta=0, verbose=False, tune=False):

        self.verbose = verbose
        self.best_score = np.inf
        self.delta = delta
        self.path = path
        self.tune = tune

    def __call__(self, val_loss, model, path=None):
        """
        Checks if the model should be saved based on the validation loss.

        Args:
            val_loss (float): The current validation loss.
            model (torch.nn.Module): The model to be saved.
            path (str, optional): Alternative path to save the model.

        This method saves the model if the validation loss has improved.
        """
        
        if val_loss < self.best_score - self.delta:
            if self.tune:
                self.save_weights(self.best_score, val_loss, model, path)
            else:
                self.save_weights(self.best_score, val_loss, model)
            self.best_score = val_loss

    def save_weights(self, old_score, new_score, model, path=None):
        """
        Saves the model weights when validation loss improves.

        Args:
            old_score (float): The previous best validation loss.
            new_score (float): The new validation loss.
            model (torch.nn.Module): The model to be saved.
            path (str, optional): Alternative path to save the model.

        This method handles the actual saving of the model weights and
        prints a message if verbose is True.
        """

        if self.verbose:
            print(f'Validation loss decreased ({old_score:.4f} --> {new_score:.4f}).  Saving model state dict ...')
        if path is not None:
            torch.save(model.state_dict(), path)
        else:
            torch.save(model.state_dict(), self.path)