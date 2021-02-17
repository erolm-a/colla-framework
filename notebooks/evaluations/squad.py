import sys
import argparse

import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, BertForTokenClassification, BertForQuestionAnswering

import wandb

from tools.dataloaders import SQuADDataloader

from models import EaEForQuestionAnswering, EntitiesAsExperts
from models.training import train_model, get_optimizer, get_schedule, MetricWrapper, save_models
from models import EntitiesAsExperts, EaEForQuestionAnswering
from models.device import get_available_device

class BertQAWrapper(BertForQuestionAnswering):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        return result.loss, result.start_logits, result.end_logits

def parse_batch(batch):
    input_ids = torch.tensor(batch['input_ids'])
    attention_mask = torch.FloatTensor(batch['attention_mask'])
    token_type_ids = torch.tensor(batch['token_type_ids'])
    start = torch.tensor(batch['answer_start'])
    end = torch.tensor(batch['answer_end'])
    
    return (("input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions"),
            (input_ids, attention_mask, token_type_ids, start, end),
            (batch,))

class SQuADMetric(MetricWrapper):
    def __init__(self, squad_dataset: SQuADDataloader):
        self.squad_dataset = squad_dataset
        self.reset()
    
    def reset(self):
        self.squad_metric = datasets.load_metric('squad')
        self.loss = 0.0
        self.n = 0
    
    def add_batch(self, inputs, outputs, loss: torch.tensor):
        self.loss += float(loss)
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
        metric_loss = self.squad_metric.compute()
        wandb.log({'exact_match': metric_loss['exact_match'],
                     'epoch': epoch,
                     'f1': metric_loss['f1'],
                     'val_loss': self.loss / self.n })



def main(run_id: str):
    np.random.seed(42)

    squad_dataset = SQuADDataloader()

    def squad_collate_fn(rows):
        keys = rows[0].keys()
        return {key: [row[key] for row in rows] for key in keys}

    # TODO: this causes a Type Error as it returns a None. Why is that?
    squad_train_dataset = squad_dataset.dev_dataset if wandb.config.squad_is_dev else squad_dataset.train_dataset
    squad_train_dataloader = DataLoader(squad_train_dataset,
                                        batch_size=wandb.config.squad_batch_size,
                                        collate_fn=squad_collate_fn,
                                        num_workers=8)


    squad_validation_dataset = squad_dataset.validation_dataset
    squad_validation_dataloader = DataLoader(squad_validation_dataset,
                                            batch_size=wandb.config.squad_batch_size,
                                            collate_fn=squad_collate_fn,
                                            num_workers=8)


    pretraining_model = EntitiesAsExperts.from_pretrained("pretraining_eae_one_epoch", run_id)

    DEVICE = get_available_device()
    # TODO: make sure that while training a model gets moved to the DEVICE
    model_qa = EaEForQuestionAnswering(pretraining_model).to(DEVICE)
    # model_qa = BertQAWrapper.from_pretrained("bert-base-uncased").to(DEVICE)
    
    # wandb.watch(model_qa)

    my_metric = SQuADMetric(squad_dataset)

    squad_epochs = wandb.config.squad_epochs

    optimizer = get_optimizer(model_qa, learning_rate=float(wandb.config.squad_learning_rate))
    scheduler = get_schedule(squad_epochs, optimizer, squad_train_dataloader)

    train_model(model_qa, squad_train_dataloader, squad_validation_dataloader,
                    parse_batch, optimizer, scheduler, squad_epochs, my_metric,
                    gradient_accumulation_factor=wandb.config.squad_gradient_accum_size)


    save_models(squad_qa_5epoch_dev=model_qa)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(  
        description='Perform evaluation on SQuAD')
    parser.add_argument(
        '--run_id', type=str, required=True,
        help='The W&B run identifier of the EaE checkpoint.')

    args = parser.parse_args()  

    main(args.run_id)