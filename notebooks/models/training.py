"""
A set of training helpers
"""

import json
import os
from typing import Callable, Optional, List, Tuple, Any

import deprecated
import torch
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

import wandb

from tqdm import tqdm
from tools.dumps import get_filename_path
from .device import get_available_device

DEVICE = get_available_device()

MAX_GRAD_NORM = 1.0

class MetricWrapper:
    """
    A mere wrapper for a HuggingFace's Datasets Metric.

    The calling order is:

    for batch in validation:
        1. reset()
        2. add_batch()
        3. compute()
    
    This class provides a default implementation of a MetricWrapper which
    sends evaluation loss to W&B.
    """

    def __init__(self, dataloader: DataLoader):
        self.reset()
        self.dataloader_length = len(dataloader)

    def add_batch(self, _inputs, _outputs, loss: torch.tensor):
        """Add a batch

        This step must update wandb with the number of 

        :param _inputs
        :param _outputs
        :param loss the loss of the model (must be moved to cpu)
        """
        self.loss += float(loss)
    
    def compute(self, epoch: int) -> float:
        """
        Compute the metric after the batches have been added.
        This call may call wandb to perform logging.
        """
        avg_loss = self.loss / self.dataloader_length
        wandb.log({'val_loss': avg_loss, "epoch": epoch})

    def reset(self):
        """
        Reset the internal metric counter.
        """
        self.loss = 0.0


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
    
    # TODO: log mismatching examples...

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
           are the outputs of the model. If a metric is not provided a default one will be provided.
    :param gradient_accumulation_factor if provided, accumulate the gradient for the given number
           of steps. This can be useful in order to avoid OOM issues with CUDA.
    """


    if metric is None:
        metric = MetricWrapper(validation_dataloader)

    example_ct = 0

    for epoch in range(epochs):
        model.train()


        pending_updates = False

        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            model_input, _ = load_from_dataloader(batch)
            inputs = [elem.to(DEVICE) for elem in model_input]

            loss, *_ = model(*inputs)
            loss.backward()
            pending_updates = True

            clip_grad_norm_(parameters=model.parameters(),
                            max_norm=MAX_GRAD_NORM)

            if (batch_idx + 1) % gradient_accumulation_factor == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                pending_updates = False

            example_ct += len(model_input)

            if (batch_idx + 1) % 25 == 0:
                train_log(loss, example_ct, epoch)
                
                
        if pending_updates:
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            pending_updates = False

        model.eval()

        metric.reset()
        model.zero_grad()

        with torch.no_grad():
            for batch in tqdm(validation_dataloader):
                model_input, metric_input = load_from_dataloader(batch)
                example_ct += len(model_input)

                inputs = [elem.to(DEVICE) for elem in model_input]
    
                loss, *outputs = model(*inputs)

                metric.add_batch(model_input + metric_input, outputs, loss)

        metric.compute(epoch)



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

def save_models(format='h5', **kwargs):
    """
    Save the models

    >>> save_models(common_model=common_model, bioclassifier= bioclassifier)

    All the kwarg parameters must be `Module`s.
    If the models have a config attribute then it will be saved as well.
    """

    for key, value in kwargs.items():
        model_path = os.path.join(f"{wandb.run.dir}/{key}.{format}")
        torch.save(value.state_dict(), model_path)

        config = getattr(value, "config", None)
        if config:
            config_path = os.path.join(f"{wandb.run.dir}/{key}.json")
            with open(config_path, "w") as f:
                json.dump(config, f)