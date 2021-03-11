"""
A set of training helpers
"""

from abc import ABC, abstractmethod
import json
import os
from typing import cast, Callable, Optional, List, Tuple, Any, Union, Sequence, Dict

import numpy as np
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


class MetricWrapper(ABC):
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

    def __init__(self, dataloader: DataLoader, enable_wandb=True):
        """
        :warning All the children with a custom __init__ must redefine this constructor
        """
        self.reset(True)
        self.dataloader = dataloader
        self.dataloader_length = len(dataloader)
        self.enable_wandb = enable_wandb

    @abstractmethod
    def add_batch(self, _inputs, _outputs, loss: float):
        """Add a batch

        :param _inputs
        :param _outputs: a list of outputs from the model.
               The training loop is type-agnostic, but is known to return a list of return values
               (or just a singleton).
        :param loss the loss of the model
        """
        pass

    @abstractmethod
    def compute(self, epoch: int) -> float:
        """
        Compute the metric after the batches have been added.
        This call may call wandb to perform logging.
        """
        pass

    @abstractmethod
    def reset(self, is_validation: bool):
        """
        Reset the internal metric counter.
        """
        pass


class TorchScriptDumpable(ABC):
    @abstractmethod
    def generate_dummy_input(self):
        """
        Generate dummy input. Useful for getting a Torchscript trace.
        """
        pass

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
            run_name: str,
            enable_wandb=True,
            watch_wandb=True):
        """
        :param model a module to track.
               The module must have a forward that returns a pair (loss, something).
        :param run_name the name of the run. Needed for checkpointing.
        :param enable_wandb whether to enable wandb logging and example reporting
        :param watch_wandb whether to track the model with wandb.
        """
        self.enable_wandb = enable_wandb
        self.run_name = run_name
        self.model = model.to(DEVICE)
        self.model.train()


        if enable_wandb:
            # save everything we are going to write
            wandb.save("*.json")
            wandb.save("*.h5")
            wandb.save("*.pt")
            wandb.save("*.pth")

        if watch_wandb:
            wandb.watch(model)

    @property
    def training(self) -> bool:
        """
        Wrapper over model.training
        """
        return self.model.training

    @training.setter
    def training(self, new_value: bool):
        """
        Wrapper over model.train()
        """
        self.model.train(new_value)

    @staticmethod
    @abstractmethod
    def load_from_dataloader(batch) -> Tuple[List[torch.Tensor], List[Any]]:
        """
        :param batch a batch loaded from a given data loader

        If training this method should return a list of tensors for training.
        Otherwise, this should return a pair of such tensors and other useful metric data.
        """


    def step(self, batch, metric: MetricWrapper, is_validation=True) -> Optional[torch.Tensor]:
        """
        :param batch
        :param metric if evaluating, metric.add_batch() will be called. Otherwise it will be
               ignored
        :param is_validation if evaluating, metric will assume that the current batch is taken from
               a validation set, otherwise from a test set. As a consequence example logging may be
               enabled or disabled.
        """
        model_input, metric_input = self.load_from_dataloader(batch)
        inputs = tuple(elem.to(DEVICE) for elem in model_input)
        loss, outputs = self.model(*inputs)

        if self.training:
            return loss

        loss = float(loss)
        metric.is_validation = is_validation
        metric.add_batch(model_input + metric_input, outputs, loss)

        # make mypy happy
        return None

    def train_log(self, loss: float, epoch: int):
        """
        Log train step. For the evaluation we rely on MetricWrapper
        """
        log_payload = {"train_loss": loss, "epoch": epoch}

        if DEVICE == "cuda":
            log_payload["gpu_mem_allocated"] = torch.cuda.memory_allocated()

        if self.enable_wandb:
            wandb.log(log_payload)

    def zero_grad(self):
        self.model.zero_grad()

    def save_models(
        self,
        model_name: str,
        network_format = 'pytorch',
        export: Optional[str] = None
    ):
        """
        Save the model as a checkpoint.

        :param network_format can be "torchscript" or "pytorch" (default)
            When `network_format` = "torchscript" we expect the model to
            implement "generate_dummy_input". The generated dummy values should
            only be used for evaluation and/or visualization tasks (e.g. netron),
            NOT for finetuning and/or transfer learning. Extension format: "pt"
            When `network_format`= "pytorch" the model state will be saved.

        :param model_name
        :param export if provided save the current file as a W&B artifact.
        """

        extension_format = {
            "pytorch": "pt",
            "torchscript": "pt",
            # ...
        }

        extension = extension_format[network_format]

        path_dir = wandb.run.dir if self.enable_wandb else get_filename_path("eae")

        full_model_name = f"{model_name}.{extension}"
        model_path = os.path.join(path_dir, full_model_name)

        if network_format == "pytorch":
            torch.save(self.model.state_dict(), model_path)

        elif network_format == "torchscript":
            assert isinstance(self.model, TorchScriptDumpable), \
                "Torchscript exporting is only available when implementing TorchScriptDumpable"
            traced = torch.jit.trace(self.model, self.model.generate_dummy_input())
            traced.save(model_path)

        config = getattr(self.model, "config", None)
        if config:
            config_path = os.path.join(path_dir, model_name + ".json")
            with open(config_path, "w") as f:
                json.dump(config, f)

        if export and self.enable_wandb:
            model_artifact = wandb.Artifact(export, type="model")
            model_artifact.add_file(model_path, full_model_name)
            if config:
                model_artifact.add_file(model_path, config_path)

            wandb.run.log_artifact(model_artifact)


def train_model(
    model_trainer: ModelTrainer,
    train_dataloader: DataLoader,
    validation_dataloader: Optional[DataLoader],
    testing_dataloader: Optional[DataLoader],
    optimizer: Optimizer,
    scheduler: LambdaLR,
    epochs: int,
    metric: MetricWrapper,
    validation_frequency: int = 100,
    gradient_accumulation_factor: int = 4,
    seed=123456,
    checkpoint_frequency=0
):
    """
    Train a model.

    :param model_trainer an instance of ModelTrainer.
    :param train_dataloader a dataloader
    :param validation_dataloader a dataloader for validation. This gets called after every
           `validation_frequency` steps.
           If None no validation step is performed
    :param testing_dataloader a dataloader for testing.
    :param epochs
    :param validation_frequency how often to perform validation.
    :param gradient_accumulation_factor if provided, accumulate the gradient for the given number
           of steps. This can be useful in order to avoid OOM issues with CUDA.
    :param seed if provided, set up the seed.
    :param checkpoint_frequency the frequency to save model checkpoints to. If 0, no checkpoint
           is saved.
    """

    # train_example_ct = 0

    torch.manual_seed(seed)
    np.random.seed(seed)

    for epoch in range(epochs):
        model_trainer.training = True
        optimizer.zero_grad()

        pending_updates = False
        train_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            loss = cast(torch.Tensor, model_trainer.step(
                batch, metric)) / gradient_accumulation_factor

            loss.backward()

            train_loss += float(loss)

            pending_updates = True

            clip_grad_norm_(parameters=model_trainer.model.parameters(),
                            max_norm=MAX_GRAD_NORM)

            if (batch_idx + 1) % gradient_accumulation_factor == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                pending_updates = False

            del loss

            if (batch_idx + 1) % 25 == 0:
                model_trainer.train_log(train_loss / 25, epoch)
                train_loss = 0.0

            # Every `validation_frequency` steps validations must be performed
            if validation_dataloader and (batch_idx + 1) % validation_frequency == 0:
                model_trainer.training = False
                optimizer.zero_grad()
                metric.reset(True)

                with torch.no_grad():
                    for validation_batch in tqdm(validation_dataloader):
                        model_trainer.step(validation_batch, metric, is_validation=True)

                metric.compute(epoch)

                model_trainer.training = True

            # DEBUG
            if checkpoint_frequency > 0 and (batch_idx + 1) % checkpoint_frequency == 0:
                model_trainer.save_models(f"{model_trainer.run_name}__{epoch}_{batch_idx}")

        if pending_updates:
            model_trainer.training = True
            optimizer.step()
            scheduler.step()
            model_trainer.zero_grad()
            pending_updates = False

        metric.reset(False)

        if testing_dataloader:
            with torch.no_grad():
                model_trainer.training = False
                for test_batch in tqdm(testing_dataloader):
                    model_trainer.step(test_batch, metric, is_validation=False)

                metric.compute(epoch)

        # TODO: should we also serialise optimizer and scheduler?
        model_trainer.save_models(model_trainer.run_name + f"_checkpoint_epoch_{epoch}")
    model_trainer.training = False

    model_trainer.save_models(model_trainer.run_name, export="final_model")


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
