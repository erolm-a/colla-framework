"""
A set of training helpers
"""

from typing import Callable, Optional, List, Tuple, Any

import deprecated
import torch
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

from torch.cuda.amp import GradScaler, autocast
import wandb

from tqdm import tqdm

from tools.dumps import get_filename_path

from .device import DEVICE

def login():
    wandb.login() # this causes stdin read for the first time only
    wandb.init(project="EntitiesAsExperts")

MAX_GRAD_NORM = 1.0

class MetricWrapper:
    """
    A mere wrapper for a HuggingFace's Datasets Metric.

    The calling order is:

    for batch in validation:
        1. reset()
        2. add_batch()
        3. compute()
    """

    def add_batch(inputs, outputs):
        """Add a batch"""
        raise NotImplementedError("")
    
    def compute(self, loss: float):
        """
        Compute the metric after the batches have been added.
        This call may call wandb to perform logging.
        """
        raise NotImplementedError("")

    def reset(self):
        """
        Reset the internal metric counter.
        """
        raise NotImplementedError("")


def train_log(
    loss: float,
    example_ct: int,
    epoch: int,
    verbose=False
):
    loss = float(loss)

    wandb.log({"train_loss": loss, "epoch": epoch}, step=example_ct)
    if verbose:
        print(f"Loss after " + str(example_ct).zfill(5) +
              f" examples: {loss:.3f}")

# pylint: disable(too-many-arguments)


def train_model(
        model: Module,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        load_from_dataloader: Callable,
        optimizer: Optimizer,
        scheduler: LambdaLR,
        epochs: int,
        metric: Optional[MetricWrapper] = None,
        gradient_accumulation_factor: int = 4,
        automatic_mixed_precision: bool = False):
    """
    Train a model.

    :param model a Model whose forward returns a tuple with AT LEAST 2 elements.
    :param train_dataloader a dataloader
    :param validation_dataloader a dataloader for validation
    :param load_from_dataloader a callable that returns a pair (model_params, metric_params).
        The idea is that model_params is entirely made of pytorch tensors that can be moved to
        the gpu, while metric_params contains the parameters for the metric.

    :param optimizer
    :param scheduler
    :param epochs
    :param metric if provided, a callable that invokes a metric (typically a Dataset.Metric).
           It will be called with params (model_params + metric_params, outputs) where outputs
           are the outputs of the model.
    :param gradient_accumulation_factor if provided, accumulate the gradient for the given number
           of steps. This can be useful in order to avoid OOM issues with CUDA.
    """

    train_losses = []

    scaler = GradScaler()

    for epoch in range(epochs):
        model.train()

        total_loss = 0
        example_ct = 0

        pending_updates = False

        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            model_input, _ = load_from_dataloader(batch)
            inputs = [elem.to(DEVICE) for elem in model_input]

            with autocast():
                loss, *_ = model(*inputs)
                scaler.scale(loss).backward()
                pending_updates = True
                total_loss += loss.item()

                clip_grad_norm_(parameters=model.parameters(),
                                max_norm=MAX_GRAD_NORM)

                if (batch_idx + 1) % gradient_accumulation_factor == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    model.zero_grad()
                    pending_updates = False

                example_ct += len(model_input)

                if (batch_idx + 1) % 25 == 0:
                    train_log(loss, example_ct, epoch)
                
                
        if pending_updates:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            model.zero_grad()
            pending_updates = False

        avg_train_loss = total_loss / len(train_dataloader)
        tqdm.write(f"Average train loss at epoch {epoch}: {avg_train_loss}")
        
        train_losses.append(avg_train_loss)
        model.eval()
        total_loss = 0
        metric.reset()

        # make sure the model has zero gradient before evaluating
        model.zero_grad()

        with torch.no_grad():
            for batch in tqdm(validation_dataloader):
                model_inputs, metric_inputs = load_from_dataloader(batch)
                inputs = [elem.to(DEVICE) for elem in model_inputs]

                loss, *outputs = model(*inputs)
                total_loss += loss.item()

                if metric:
                    metric.add_batch(model_inputs + metric_inputs, outputs, loss)

        avg_validation_loss = total_loss / len(validation_dataloader)
        tqdm.write(
            f"Average eval loss at epoch {epoch}: {avg_validation_loss}")
        metric.compute(epoch)

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

@deprecated.deprecated("TODO: Migrate to W&B")
def save_models(**kwargs):
    """
    Save the models

    >>> save_models(common_model=common_model, bioclassifier= bioclassifier)

    All the kwarg parameters must be `Module`s.
    """

    for key, value in kwargs.items():
        path = get_filename_path(f"eae/{key}.pt")
        torch.save(value.state_dict(), path)

@deprecated.deprecated("TODO: Migrate to W&B")
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
