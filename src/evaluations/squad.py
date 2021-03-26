import sys
import argparse

import datasets
import numpy as np
import pandas as pd

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
        selected_batch = (
            torch.LongTensor(batch["input_ids"]),
            torch.FloatTensor(batch["attention_mask"]),
            torch.LongTensor(batch["token_type_ids"]),
            torch.LongTensor(batch["start_position"]),
            torch.LongTensor(batch["end_position"]),
            torch.LongTensor(batch["is_impossible"])
        )

        metric_batch = (
            batch["id"],
            batch["question"],
            batch["context"],
            batch["answers"]
        )

        return selected_batch, metric_batch


class SQuADMetric(MetricWrapper):
    def __init__(self,
        dataloader: DataLoader,
        enable_wandb=False,
        enable_example_wandb=False
    ):
        super().__init__(dataloader, enable_wandb)
        self.enable_example_wandb = enable_example_wandb
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def reset(self, is_validation: bool):
        super().reset(is_validation)
        self.loss = 0.0
        self.squad_metric = datasets.load_metric('squad')
        self.examples = []
        self.n = 0
    
    def log_examples(
        self,
        reconstructed_texts: List[str],
        gold_answers: List[str],
        questions: List[str],
        contexts: List[str]
    ):
        examples = list(zip(questions, contexts, reconstructed_texts, gold_answers))
        self.examples.extend(examples)

    def reconstruct_sentences(
        self,
        input_ids_list: List[List[int]],
        answers_start: List[int],
        answers_end: List[int]
    ) -> List[str]:
        """
        Reconstruct the sentences given a list of token ids and a span.
        Unfortunately there is no way to do that efficiently given that spans are ragged.

        :param input_ids_list
        :param answers_start
        :param answers_end

        :returns a list of strings
        """

        answers = [input_ids[answer_start:answer_end+1] for input_ids, answer_start, answer_end
            in zip(input_ids_list, answers_start, answers_end)]

        return self.tokenizer.batch_decode(answers)

    
    def add_batch(
        self,
        inputs: List[torch.Tensor],
        outputs: EaEForQuestionAnsweringOutput,
        loss: float
    ):
        with torch.no_grad():
            self.loss += loss
            self.n += len(inputs[0][0])

            qas_ids = inputs[-4]
            questions = inputs[-3]
            contexts = inputs[-2]
            answers = inputs[-1]

            # outputs = total_loss, answer_start_logits, answer_end_logits
            answer_start_logits = outputs[1].detach().cpu()
            answer_end_logits = outputs[2].detach().cpu()

            answer_starts = torch.argmax(answer_start_logits, 1).tolist()
            answer_ends = torch.argmax(answer_end_logits, 1).tolist()

            input_ids = inputs[0].detach().cpu().tolist()
            
            prediction_texts = self.reconstruct_sentences(input_ids, answer_starts, answer_ends)

            predictions = [{
                "id": id,
                "prediction_text": prediction_text,
            } for id, prediction_text in zip(qas_ids, prediction_texts)]

            answers_reference = {
                "text": [x[0].lower() for x in answers["text"]],
                # Note that answer_start values are not taken into account to compute the metric.
                "answer_start": [int(x) for x in answers["answer_start"]]
            }

            references = [{
                "id": qas_ids[0],
                "answers": answers_reference
            }]

            self.squad_metric.add_batch(predictions=predictions, references=references)

            self.log_examples(prediction_texts, answers["text"], questions, contexts)

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

            if self.is_validation and len(self.examples) > 0:
                examples = self.examples[::25]
                payload["Validation examples"] = wandb.Table(data=examples,
                    columns=["Question", "Context", "Predicted Answer", "Ground Truth"]
                )
                

            wandb.log(payload)
        else:
            pprint.pprint(payload)

            if self.is_validation and len(self.examples) > 0:
                examples = self.examples[::25]
                pprint.pprint(examples)
                #print(pd.DataFrame(examples,
                #    columns=["Question", "Context", "Predicted Answer", "Ground Truth"]))

        return avg_loss

NUM_WORKERS = 0
ENABLE_WANDB = True

def get_dataloaders(
    squad_dataset: SQuADDataloader,
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
    
    return [DataLoader(dataset, batch_size=1, num_workers=NUM_WORKERS)
            for dataset in (squad_training_dataset, squad_validation_dataset, squad_test_dataset)]

def main(
    variant: str,
    run_id: Optional[str],
    _squad_version: int,
    wandb_args: dict
):
    np.random.seed(42)

    if ENABLE_WANDB and run_id is not None:
        wandb.init(project="EaEPretraining", config="configs/eae_squad.yaml", job_type="squad_evaluation")
        #batch_size = wandb.config.batch_size
        is_dev = wandb.config.is_dev
        gradient_accum_size = wandb.config.gradient_accum_size
        learning_rate = wandb.config.learning_rate
        full_finetuning = wandb.config.full_finetuning
        epochs = wandb.config.epochs
        pretraining_model = EntitiesAsExperts.from_pretrained(variant, run_id)
        run_name = f"squad{'_dev' if is_dev else ''}_{epochs}"
    else:
        #batch_size = 1
        gradient_accum_size = 1
        is_dev = True
        learning_rate = 1e-4
        full_finetuning = False
        epochs = 1
        pretraining_model = EntitiesAsExperts.from_pretrained(variant, run_id, as_wandb=None)
        run_name = "squad_dev_experiment"


    squad_dataset = SQuADDataloader()
    squad_train_dataloader, squad_validation_dataloader, squad_test_dataloader = \
        get_dataloaders(squad_dataset, is_dev)

    squad_model = EaEForQuestionAnswering(pretraining_model)

    metric = SQuADMetric(
        squad_validation_dataloader,
        enable_wandb=ENABLE_WANDB
    )

    model_trainer = SquadModelTrainer(
        squad_model,
        run_name,
        watch_wandb=False, #ENABLE_WANDB,
        enable_wandb=ENABLE_WANDB
    )

    
    optimizer = get_optimizer(squad_model,
                                learning_rate=learning_rate,
                                full_finetuning=full_finetuning)
    scheduler = get_schedule(epochs, optimizer, squad_train_dataloader)

    train_model(model_trainer, squad_train_dataloader, squad_validation_dataloader,
                squad_test_dataloader, optimizer, scheduler, epochs, metric,
                validation_frequency= 500,# * batch_size,
                gradient_accumulation_factor=gradient_accum_size,
                checkpoint_frequency=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(  
        description='Perform evaluation on SQuAD')
    parser.add_argument(
        '--variant', type=str, required=False,
        help='The W&B eae model checkpoint variant.')
    parser.add_argument(
        '--run-id', type=str, required=True,
        help='The W&B run identifier of the EaE checkpoint.')
    parser.add_argument(
        "--squad-version", type=int, required=True,
        help="The variant type of SQuAD to use."
    )

    args = parser.parse_args() 
    main(args.variant, args.run_id, args.squad_version, args)
