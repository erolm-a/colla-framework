"""
A set of training helpers
"""

from abc import ABC, abstractmethod
import json
import os
from typing import Callable, Optional, List, Tuple, Any, Union

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

    def add_batch(self, _inputs, _outputs: List, loss: float):
        """Add a batch

        :param _inputs
        :param _outputs: a list of outputs from the model.
               The training loop is type-agnostic, but is known to return a list of return values
               (or just a singleton). 
        :param loss the loss of the model
        """
        self.loss += loss
    
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


class ModelTrainer(ABC):
    """
    A wrapper on trainers that performs logging, example collection and other goodies.
    
    Default methods are implemented, but callers should override according to the needs.

    At least, the caller must define load_from_dataloader.

    Yes. This is a clone of Pytorch's Lighting. I am so ashamed of this code.
    """

    def __init__(
        self,
        model: Module,
        enable_wandb=True,
        watch_wandb=True):
        """
        :param model a module to track.
               The module must have a forward that returns a pair (loss, something).
        :param optimizer the model optimizer.
        :param scheduler the module scheduler
        :param enable_wandb whether to enable wandb logging and example reporting
        :param watch_wandb whether to track the model with wandb.
        """
        self.enable_wandb = enable_wandb
        self.model = model.to(DEVICE)
        self.model.train()

        if watch_wandb:
            wandb.watch(model)
    
    @property
    def training(self) -> bool:
        """
        Wrapper over model.training()
        """
        return self.training()

    @training.setter
    def training(self, new_value: bool):
        """
        Wrapper over model.train()
        """
        self.model.train(new_value)

    @staticmethod
    @abstractmethod
    def load_from_dataloader(batch) -> Union[List[torch.tensor], Tuple[List[torch.tensor], Any]]:
        """
        :param batch a batch loaded from a given data loader

        If training this method should return a list of tensors for training.
        Otherwise, this should return a pair of such tensors and other useful metric data.
        """
        pass

    def step(self, batch, metric: MetricWrapper) -> Optional[torch.tensor]:
        """
        :param metric if evaluating, metric.add_batch() will be called. Otherwise it will be ignored
        """
        if self.training:
            model_input = self.load_from_dataloader(batch)
            inputs = [elem.to(DEVICE) for elem in model_input]
            loss, _ = self.model(*inputs)

            return loss
        else:
            model_input, metric_input = self.load_from_dataloader(batch)
            inputs = [elem.to(DEVICE) for elem in model_input]
            loss, outputs = self.model(*inputs)
            loss = float(loss)
            metric.add_batch(model_input + metric_input, outputs, loss)
        

    def train_log(self, loss: float, example_ct: int, epoch: int, verbose = False):
        """
        Log train step. For the evaluation we rely on MetricWrapper
        """
        log_payload = {"train_loss": loss, "epoch": epoch}

        if DEVICE == "cuda":
            log_payload["gpu_mem_allocated"] = torch.cuda.memory_allocated() 
        
        if self.enable_wandb:
            wandb.log(log_payload, step=example_ct)

    def zero_grad(self):
        self.model.zero_grad()


def train_model(
        model_trainer: ModelTrainer,
        train_dataloader: DataLoader,
        validation_dataloader: Optional[DataLoader],
        testing_dataloader: Optional[DataLoader],
        optimizer: Optimizer,
        scheduler: LambdaLR,
        epochs: int,
        validation_frequency: int = 100,
        metric: Optional[MetricWrapper] = None,
        gradient_accumulation_factor: int = 4,
        seed = 123456
    ):
    """
    Train a model.

    :param model_trainer an instance of ModelTrainer.
    :param train_dataloader a dataloader
    :param validation_dataloader a dataloader for validation. This gets called after every `validation_frequency` steps.
           If None no validation step is performed
    :param testing_dataloader a dataloader for testing.
    :param epochs
    :param validation_frequency how often to perform validation.
    :param gradient_accumulation_factor if provided, accumulate the gradient for the given number
           of steps. This can be useful in order to avoid OOM issues with CUDA.
    :param seed if provided, set up the seed.
    """

    if metric is None:
        metric = MetricWrapper(validation_dataloader)

    train_example_ct = 0

    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
    
    for epoch in range(epochs):
        model_trainer.training = True
        model_trainer.zero_grad()

        pending_updates = False

        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            if (batch_idx + 1) % validation_frequency != 0:
                loss = model_trainer.model_step(batch, metric) / gradient_accumulation_factor

                loss.backward()

                loss = loss.detach().cpu()
                loss = float(loss)

                pending_updates = True

                clip_grad_norm_(parameters=model_trainer.parameters(),
                                max_norm=MAX_GRAD_NORM)

                if (batch_idx + 1) % gradient_accumulation_factor == 0:
                    optimizer.step()
                    scheduler.step()
                    model_trainer.zero_grad()
                    pending_updates = False

                train_example_ct += len(batch[0])

                if (batch_idx + 1) % 25 == 0:
                    model_trainer.train_log(loss, train_example_ct, epoch)
            else:
                model_trainer.training = False
                model_trainer.zero_grad()
                metric.reset()

                with torch.no_grad():
                    for validation_batch in tqdm(validation_dataloader):
                        model_trainer.step(validation_batch, metric)

                metric.compute(epoch)

                model_trainer.training = True

        if pending_updates:
            model_trainer.training = True
            optimizer.step()
            scheduler.step()
            model_trainer.zero_grad()
            pending_updates = False
    
    model_trainer.training = False
        

def get_optimizer(model: Module, learning_rate: float, full_finetuning=False):
    """
    Initialize an AdamW optimizer.

    :param model a PyTorch model
    :param learning_rate the learning rate for AdamW
    :param full_finetuning if True explore weight decay
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
        lr=learning_rate,
        eps=1e-8
    )


def get_schedule(epochs: int, optimizer: Optimizer, train_dataloader: DataLoader):
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
