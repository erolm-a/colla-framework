"""
A set of training helpers
"""

from typing import Callable, Optional


import torch
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from tqdm import tqdm

from .device import DEVICE

MAX_GRAD_NORM = 1.0

# pylint: disable(too-many-arguments)
def train_model(
    model: Module,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    load_from_dataloader: Callable,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    epochs: int,
    metric: Optional[Callable]):
    """
    Train a model.

    :param model a Model whose forward returns a tuple with AT LEAST 2 elements.
    """
    
    train_losses = []
    validation_losses = []

    for epoch in range(epochs):
        model.train()

        total_loss = 0

        for batch in tqdm(train_dataloader):
            inputs = [elem.to(DEVICE) for elem in load_from_dataloader(batch)]

            model.zero_grad()
            loss, *_ = model(*inputs)
            loss.backward()
            total_loss += loss.item()

            clip_grad_norm_(parameters=model.parameters(),
                                max_norm=MAX_GRAD_NORM)

            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        tqdm.write(f"Average train loss at epoch {epoch}: {avg_train_loss}")

        train_losses.append(avg_train_loss)

        model.eval()

        total_loss = 0

        for batch in tqdm(validation_dataloader):
            inputs = [elem.to(DEVICE) for elem in load_from_dataloader(batch)]


            with torch.no_grad():
                loss, *outputs = model(*inputs)
                total_loss += loss.item()

                if metric:
                    metric(inputs, outputs)

        avg_validation_loss = total_loss / len(validation_dataloader)
        validation_losses.append(avg_validation_loss)
        tqdm.write(f"Average eval loss at epoch {epoch}: {avg_validation_loss}")

    return avg_train_loss, avg_validation_loss