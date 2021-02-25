import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from transformers import BertForMaskedLM, BertForTokenClassification, BertForQuestionAnswering

import wandb

from tools.dataloaders import WikipediaCBOR

from models import EntitiesAsExperts
from models.training import train_model, get_optimizer, get_schedule, MetricWrapper, save_models
from models.device import get_available_device
from transformers import BertForMaskedLM, BertForTokenClassification

NUM_WORKERS = 16

def load_batch(batch):
    return (("input_ids", "output_ids", "entity_outputs", "_actual_entity_outputs",
             "mention_boundaries", "attention_mask", "token_type_ids"),
             batch, [])

def main():
    np.random.seed(42)

    wikipedia_cbor = WikipediaCBOR("wikipedia/car-wiki2020-01-01/enwiki2020.cbor", "wikipedia/car-wiki2020-01-01/partitions",
                                       page_lim=wandb.config.eae_wikipedia_article_nums,
    )

    if wandb.config.eae_is_dev:
        wiki_dev_size = int(0.1*len(wikipedia_cbor))
    else:
        wiki_dev_size = len(wikipedia_cbor)
    
    wiki_dev_indices = np.random.choice(len(wikipedia_cbor), size=wiki_dev_size)

    # 80/20% split
    wiki_train_size = int(0.8*wiki_dev_size)

    wiki_train_indices, wiki_validation_indices = wiki_dev_indices[:wiki_train_size], wiki_dev_indices[wiki_train_size:]
    wiki_train_sampler = SubsetRandomSampler(wiki_train_indices,
                                            generator=torch.Generator().manual_seed(42))
    wiki_validation_sampler = SubsetRandomSampler(wiki_validation_indices,
                                                generator=torch.Generator().manual_seed(42))

    wiki_train_dataloader = DataLoader(wikipedia_cbor, sampler=wiki_train_sampler,
                                    batch_size=wandb.config.eae_batch_size,
                                    num_workers=NUM_WORKERS)
    wiki_validation_dataloader = DataLoader(wikipedia_cbor,
                                            sampler=wiki_validation_sampler,
                                            batch_size=wandb.config.eae_batch_size,
                                            num_workers=NUM_WORKERS)

    model_masked_lm = BertForMaskedLM.from_pretrained('bert-base-uncased')

    DEVICE = get_available_device()

    pretraining_model = EntitiesAsExperts(model_masked_lm,
                                        wandb.config.eae_l0,
                                        wandb.config.eae_l1, 
                                        wikipedia_cbor.max_entity_num,
                                        wandb.config.eae_entity_embedding_size).to(DEVICE)

    wandb.watch(pretraining_model)

    squad_epochs = wandb.config.eae_pretraining_epochs

    optimizer = get_optimizer(pretraining_model,
                                learning_rate=float(wandb.config.eae_learning_rate),
                                full_finetuning=wandb.config.eae_full_finetuning)
    scheduler = get_schedule(squad_epochs, optimizer, wiki_train_dataloader)

    train_model(pretraining_model, wiki_train_dataloader, wiki_validation_dataloader,
                    load_batch, optimizer, scheduler, squad_epochs, None,
                    gradient_accumulation_factor=wandb.config.eae_gradient_accum_size)


    save_models(pretraining_eae_100k=pretraining_model)


if __name__ == "__main__":
    main()