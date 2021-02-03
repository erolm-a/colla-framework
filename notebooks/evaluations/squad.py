import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from transformers import BertForMaskedLM, BertForTokenClassification

import wandb

from tools.dataloaders import SQuADDataloader

from models import EaEForQuestionAnswering, EntitiesAsExperts
from models.training import train_model, get_optimizer, get_schedule, MetricWrapper, load_model
from tools.dataloaders import WikipediaCBOR
from models import EntitiesAsExperts, EaEForQuestionAnswering
from models.device import get_available_device

def parse_batch(batch):
    input_ids = torch.tensor(batch['input_ids'])
    attention_mask = torch.FloatTensor(batch['attention_mask'])
    token_type_ids = torch.tensor(batch['token_type_ids'])
    start = torch.tensor(batch['answer_start'])
    end = torch.tensor(batch['answer_end'])
    
    return (input_ids, attention_mask, token_type_ids, start, end), (batch,)

def parse_batch_2(batch):
    input_ids = torch.tensor(batch['input_ids'])
    attention_mask = torch.FloatTensor(batch['attention_mask'])
    token_type_ids = torch.tensor(batch['token_type_ids'])
    start = torch.tensor(batch['answer_start'])
    end = torch.tensor(batch['answer_end'])
    
    return (input_ids, attention_mask, token_type_ids), (batch,)


class SQuADMetric(MetricWrapper):
    def __init__(self, squad_dataset: SQuADDataloader):
        self.squad_dataset = squad_dataset
        self.reset()
    
    def reset(self):
        self.squad_metric = datasets.load_metric('squad')
        self.loss = 0.0
    
    def add_batch(self, inputs, outputs, loss):
        self.loss += float(loss)
        
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
                     'val_loss': self.loss})



def main():
    squad_metric, squad_v2_metric = datasets.load_metric('squad'), datasets.load_metric('squad_v2')

    np.random.seed(42)

    squad_dataset = SQuADDataloader()

    def squad_collate_fn(rows):
        keys = rows[0].keys()
        return {key: [row[key] for row in rows] for key in keys}

    squad_train_dataset = squad_dataset.train_dataset

    FULL_FINETUNING = wandb.config.squad_is_dev

    if not FULL_FINETUNING:
        squad_dev_size = int(0.1*len(squad_dataset.train_dataset))
        squad_dev_indices = np.random.choice(len(squad_dataset.train_dataset), size=squad_dev_size)
        squad_train_sampler = SubsetRandomSampler(squad_dev_indices,
                                                generator=torch.Generator().manual_seed(42))
        squad_train_dataloader = DataLoader(squad_train_dataset,
                                            sampler=squad_train_sampler,
                                            batch_size=wandb.config.squad_batch_size,
                                            collate_fn=squad_collate_fn)

    else:
        squad_train_dataloader = DataLoader(squad_train_dataset,
                                            batch_size=wandb.config.squad_batch_size,
                                            collate_fn=squad_collate_fn)

    squad_validation_dataset = squad_dataset.validation_dataset
    squad_validation_dataloader = DataLoader(squad_validation_dataset,
                                            batch_size=wandb.config.squad_batch_size,
                                            collate_fn=squad_collate_fn)

    model_masked_lm = BertForMaskedLM.from_pretrained('bert-base-uncased')

     # TODO: move this to the config zone

    wikipedia_cbor = WikipediaCBOR("wikipedia/car-wiki2020-01-01/enwiki2020.cbor", "wikipedia/car-wiki2020-01-01/partitions",
                                        # top 2% most frequent items,  roughly at least 100 occurrences, with a total of  ~ 20000 entities
                                        # cutoff_frequency=0.02, recount=True 
                                        # TODO: is this representative enough?
    )



    pretraining_model = load_model(EntitiesAsExperts,
                                    "pretraining_eae_one_epoch",
                                    model_masked_lm,
                                    wandb.config.eae_l0,
                                    wandb.config.eae_l1,
                                    30703, wandb.config.eae_entity_embedding_size)



    DEVICE = get_available_device()
    # TODO: make sure that while training a model gets moved to the DEVICE
    model_qa = EaEForQuestionAnswering(pretraining_model).to(DEVICE)

    #wandb.watch(model_qa)

    my_metric = SQuADMetric(squad_dataset)

    squad_epochs = wandb.config.squad_epochs

    optimizer = get_optimizer(pretraining_model)
    scheduler = get_schedule(squad_epochs, optimizer, squad_train_dataloader)

    train_model(model_qa, squad_train_dataloader, squad_validation_dataloader,
                    parse_batch, optimizer, scheduler, squad_epochs, my_metric, gradient_accumulation_factor=1)



if __name__ == "__main__":
    main()