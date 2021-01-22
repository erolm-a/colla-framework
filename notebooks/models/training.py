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
from transformers import AdamW, get_linear_schedule_with_warmup

from tqdm import tqdm

from .device import DEVICE

from tools.dumps import get_filename_path

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
        tqdm.write(
            f"Average eval loss at epoch {epoch}: {avg_validation_loss}")

    return avg_train_loss, avg_validation_loss


def get_optimizer(model: Module, full_finetuning=False):
    """
    Get an optimizer

    :param model a PyTorch model
    :param full_finetuning if True perform full finetuning
    """
    param_optimizer = list(model.named_parameters())

    if full_finetuning:
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer]}]

    return AdamW(
        optimizer_grouped_parameters,
        lr=1e-4,
        eps=1e-8
    )


def get_schedule(epochs, optimizer, train_dataloader):
    """
    Get a schedule
    """
    total_steps = len(train_dataloader) * epochs

    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05*total_steps),
        num_training_steps=total_steps
    )


def save_models(**kwargs):
    """
    Save the models

    >>> save_models(common_model=common_model, bioclassifier= bioclassifier)

    All the kwarg parameters must be `Module`s.
    """

    for key, value in kwargs.items():
        path = get_filename_path(f"eae/{key}.pt")
        torch.save(value.state_dict(), path)


def load_model(model: type, saved_model_name: str, *args, **kwargs):
    """
    Load a given model.

    :param model a class that inherits from Pytorch's Module
    :param path_name the saved name of the model
    """
    model = model(*args, **kwargs)
    model.load_state_dict(torch.load(
        get_filename_path(f"eae/{saved_model_name}.pt")))

    return model
