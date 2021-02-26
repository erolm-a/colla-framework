from typing import Tuple

import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from transformers import BertModel

import wandb

from tools.dataloaders import WikipediaCBOR

from models import EntitiesAsExperts, EntitiesAsExpertsOutputs
from models.training import train_model, get_optimizer, get_schedule, MetricWrapper, ModelTrainer, save_models
from models.device import get_available_device
from transformers import BertModel, AutoTokenizer

NUM_WORKERS = 16


class PretrainingModelTrainer(ModelTrainer):
    @staticmethod
    def load_from_dataloader(batch):
        return batch, tuple()


class PretrainingMetric(MetricWrapper):
    def __init__(self, dataloader: DataLoader):
        super().__init__(dataloader)
        self.wikipedia_dataset = dataloader.dataset # type: WikipediaCBOR
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def add_batch(
        self,
        inputs,
        outputs: EntitiesAsExpertsOutputs, 
        loss: float
    ):
        super().add_batch(inputs, outputs, loss)

        input_ids, output_ids, masked_links, links = [_.detach().cpu() for _ in inputs[:4]]

        masked_sentences = self.tokenizer.batch_decode(input_ids)
        ground_sentences = self.tokenizer.batch_decode(output_ids)
        masked_links_decoded = self.wikipedia_dataset.decode_compressed_entity_ids(masked_links)
        ground_links_decoded = self.wikipedia_dataset.decode_compressed_entity_ids(links)

        mask_token = self.tokenizer.mask_token_id

        _, token_logits, entity_logits = outputs

        token_logits = token_logits.detach().cpu()
        entity_logits = entity_logits.detach().cpu()

        token_outputs = torch.argmax(token_logits, 1)
        predicted_sentences = self.tokenizer.batch_decode(token_outputs)
        entity_outputs = torch.argmax(entity_logits, 1)
        predicted_links_decoded = self.wikipedia_dataset.decode_compressed_entity_ids(entity_outputs)

        # get the sentences where there is a masked token that is not properly classified
        # positions = torch.nonzero(masked_sentences == mask_token)

        # FIXME: replace with wandb's logging
        print(masked_sentences, ground_sentences, predicted_sentences)
        print(masked_links_decoded, ground_links_decoded, predicted_links_decoded)

NUM_WORKERS = 16

np.random.seed(42)

def get_dataloaders(wikipedia_cbor):
    if wandb.config.is_dev:
        wiki_dev_size = int(0.1*len(wikipedia_cbor))
    else:
        wiki_dev_size = len(wikipedia_cbor)
 
    wiki_dev_indices = np.random.choice(len(wikipedia_cbor), size=wiki_dev_size)

    # Use 80 % of Wikipedia for training, 18% for testing, 2% for validation
    wiki_train_size = int(0.8 * wiki_dev_size)
    wiki_validation_size = int(0.02 * wiki_dev_size)
    wiki_test_size = wiki_dev_size - wiki_train_size - wiki_validation_size

    wiki_train_indices, wiki_validation_indices, wiki_test_indices = (wiki_dev_indices[:wiki_train_size],
                                                                    wiki_dev_indices[wiki_train_size:-wiki_test_size],
                                                                    wiki_dev_indices[-wiki_test_size:])


    def get_dataloader(indices):
        sampler = SubsetRandomSampler(wiki_train_indices,
                                            generator=torch.Generator().manual_seed(42))
        return DataLoader(wikipedia_cbor, sampler=sampler,
                                    batch_size=wandb.config.batch_size,
                                    num_workers=NUM_WORKERS)

    wiki_train_dataloader = get_dataloader(wiki_train_indices)
    wiki_validation_dataloader = get_dataloader(wiki_validation_indices)
    wiki_test_dataloader = get_dataloader(wiki_test_indices)

    return (wiki_train_dataloader, wiki_validation_dataloader, wiki_test_dataloader)

def main():
    np.random.seed(42)

    wandb.init("EaEPretraining", config="configs/eae_pretraining.yaml")
    print(wandb.config)

    wikipedia_cbor = WikipediaCBOR("wikipedia/car-wiki2020-01-01/enwiki2020.cbor", "wikipedia/car-wiki2020-01-01/partitions",
                                       #page_lim=wandb.config.wikipedia_article_nums,
    )

    wiki_train_dataloader, wiki_validation_dataloader, wiki_test_dataloader = get_dataloaders(wikipedia_cbor)

    pretraining_model = EntitiesAsExperts(wandb.config.l0,
                                          wandb.config.l1,
                                          wikipedia_cbor.max_entity_num,
                                          wandb.config.entity_embedding_size)

    epochs = wandb.config.pretraining_epochs

    optimizer = get_optimizer(pretraining_model,
                                learning_rate=float(wandb.config.learning_rate),
                                full_finetuning=wandb.config.full_finetuning)
    scheduler = get_schedule(squad_epochs, optimizer, wiki_train_dataloader)

    metric = PretrainingMetric(wiki_validation_dataloader)
    model_trainer = PretrainingModelTrainer(pretraining_model)

    train_model(model_trainer, wiki_train_dataloader, wiki_validation_dataloader,
                wiki_test_dataloader, optimizer, scheduler, epochs, metric,
                validation_frequency=10,
                gradient_accumulation_factor=wandb.config.gradient_accum_size)


    save_models(pretraining_eae_100k=pretraining_model)


if __name__ == "__main__":
    main()
