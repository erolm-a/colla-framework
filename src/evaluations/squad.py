import sys
import argparse

import datasets
import numpy as np
from torch.utils.data import DataLoader

import wandb

from tools.dataloaders import SQuADDataloader

from models import EntitiesAsExperts, EaEForQuestionAnswering, EntitiesAsExpertsOutputs, EaEForQuestionAnsweringOutput
from models.device import get_available_device


#import math
from typing import List, Optional

import pprint

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
import tqdm
from transformers import AutoTokenizer

from models.training import (train_model, get_optimizer, get_schedule,
                            MetricWrapper, ModelTrainer)

import wandb


class SquadModelTrainer(ModelTrainer):
    @staticmethod
    def load_from_dataloader(batch):
        return tuple(batch), tuple()

class SQuADMetric(MetricWrapper):
    def __init__(self,
        dataloader: DataLoader,
        enable_wandb=False,
        enable_example_wandb=False
    ):
        super().__init__(dataloader, enable_wandb)
        self.enable_example_wandb = enable_example_wandb
    
    def reset(self, is_validation: bool):
        super().reset(is_validation)
        self.loss = 0.0
        self.squad_metric = datasets.load_metric('squad')
        self.n = 0
    
    def add_batch(
        self,
        inputs: List[torch.Tensor],
        outputs: EaEForQuestionAnsweringOutput,
        loss: float
    ):

        with torch.no_grad():
            self.loss += loss
            self.n += len(inputs[0][0])

            batch_input = inputs[-1]

            # outputs = total_loss, answer_start_logits, answer_end_logits
            answer_start_logits = outputs[1].detach().cpu()
            answer_end_logits = outputs[1].detach().cpu()

            answer_starts = torch.argmax(answer_start_logits, 1).tolist()
            answer_ends = torch.argmax(answer_end_logits, 1).tolist()

            input_ids = inputs[0].detach().cpu().tolist()

            prediction_texts = self.squad_dataset.reconstruct_sentences(input_ids, answer_starts, answer_ends)

            predictions = [{
                "id": id,
                "prediction_text": prediction_text,
            } for id, prediction_text in zip(batch_input["id"], prediction_texts)]


            references = [{
                "id": id,
                "answers": answers
            } for id, answers in zip(batch_input["id"], batch_input['answers'])]

            self.squad_metric.add_batch(predictions=predictions, references=references)

    # return validation loss
    def compute(self, epoch: int) -> float:

        total_length = self.dataloader_length * self.dataloader.batch_size
        avg_loss = self.loss / total_length
        prefix = "val_" if self.is_validation else "test_"

        metric_loss = self.squad_metric.compute()

        payload = {f'{prefix}exact_match': metric_loss['exact_match'],
                   f'{prefix}f1': metric_loss['f1'],
                   f'{prefix}loss': avg_loss,
                   'epoch': epoch}
        if self.enable_wandb:
            wandb.log(payload)
        else:
            pprint.pprint(payload)

        return avg_loss

NUM_WORKERS = 16
ENABLE_WANDB = False

def get_dataloaders(
    squad_dataset: SQuADDataloader,
    batch_size: int,
    is_dev: bool
):

    squad_training_dataset = getattr(squad_dataset,
        f"{'dev_' if is_dev else ''}train_dataset")
    squad_test_dataset = getattr(squad_dataset,
        f"{'dev_' if is_dev else ''}validation_dataset")


    # Reserve 1% for validation
    squad_validation_length = int(len(squad_training_dataset) * 0.01)
    squad_training_length = len(squad_training_dataset) - squad_validation_length
 
    squad_training_dataset, squad_validation_dataset = random_split(
        squad_training_dataset,
        [squad_training_length, squad_validation_length],
        generator=torch.Generator().manual_seed(42)
    )

    squad_test_dataset = squad_dataset.validation_dataset

    """
    def squad_collate_fn(rows):
        keys = rows[0].keys()
        return {key: [row[key] for row in rows] for key in keys}
    """

    return [DataLoader(dataset, batch_size=batch_size, num_workers=NUM_WORKERS)
            for dataset in (squad_training_dataset, squad_validation_dataset, squad_test_dataset)]

def main(variant: str, run_id: Optional[str], wandb_args: dict):
    np.random.seed(42)

    if ENABLE_WANDB and run_id is not None:
        wandb.init(project="EaEPretraining", config="configs/eae_squad.yaml", job_type="squad_evaluation")
        batch_size = wandb.config.batch_size
        is_dev = wandb.config.is_dev
        gradient_accum_size = wandb.config.gradient_accum_size
        learning_rate = wandb.config.learning_rate
        full_finetuning = wandb.config.full_finetuning
        epochs = wandb.config.pretraining_epochs
        pretraining_model = EntitiesAsExperts.from_pretrained(variant, run_id)
        run_name = f"squad{'_dev' if is_dev else ''}_{epochs}"
    else:
        batch_size = 1
        gradient_accum_size = 1
        is_dev = True
        learning_rate = 1e-4
        full_finetuning = False
        epochs = 1
        pretraining_model = EntitiesAsExperts.from_pretrained(variant, run_id, as_wandb=None)
        run_name = "squad_dev_experiment"


    squad_dataset = SQuADDataloader()
    squad_train_dataloader, squad_validation_dataloader, squad_test_dataloader = \
        get_dataloaders(squad_dataset, batch_size, is_dev)

    squad_model = EaEForQuestionAnswering(pretraining_model)

    metric = SQuADMetric(
        squad_validation_dataloader,
        enable_wandb=ENABLE_WANDB
    )

    model_trainer = SquadModelTrainer(
        squad_model,
        run_name,
        watch_wandb=ENABLE_WANDB,
        enable_wandb=ENABLE_WANDB
    )

    
    optimizer = get_optimizer(squad_model,
                                learning_rate=learning_rate,
                                full_finetuning=full_finetuning)
    scheduler = get_schedule(epochs, optimizer, squad_train_dataloader)

    train_model(model_trainer, squad_train_dataloader, squad_validation_dataloader,
                squad_test_dataloader, optimizer, scheduler, epochs, metric,
                validation_frequency= 5 * batch_size,
                gradient_accumulation_factor=gradient_accum_size,
                checkpoint_frequency=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(  
        description='Perform evaluation on SQuAD')
    parser.add_argument(
        '--variant', type=str, required=False,
        help='The W&B eae model checkpoint variant.')
    parser.add_argument(
        '--run_id', type=str, required=True,
        help='The W&B run identifier of the EaE checkpoint.')

    args = parser.parse_args() 
    main(args.variant, args.run_id, args)
