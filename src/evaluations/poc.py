"""
Pretraining. This is a temporary debugging script and will be removed.
"""

from io import StringIO
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import tqdm
from transformers import AutoTokenizer

from tools.dataloaders import WikipediaCBOR

from models import EntitiesAsExperts, EntitiesAsExpertsOutputs
from models.training import (train_model, get_optimizer, get_schedule,
                            MetricWrapper, ModelTrainer, save_models)

# TODO: can we avoid recomputing the cross-entropy loss again?
from torch.nn import CrossEntropyLoss

from models.device import get_available_device

NUM_WORKERS = 0

class PretrainingModelTrainer(ModelTrainer):
    @staticmethod
    def load_from_dataloader(batch):
        return tuple(batch), tuple()


class PretrainingMetric(MetricWrapper):
    def __init__(self, dataloader: DataLoader):
        super().__init__(dataloader)
        self.wikipedia_dataset = dataloader.dataset # type: WikipediaCBOR
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        # return a N-dimensional log-likelihood to compute the perplexity with...
        self.loss = CrossEntropyLoss()

    def add_batch(
        self,
        inputs: List[torch.Tensor],
        outputs: EntitiesAsExpertsOutputs,
        loss: float
    ):
        super().add_batch(inputs, outputs, loss)

        with torch.no_grad():

            input_ids, output_ids, masked_links, links = inputs[:4]

            #masked_sentences = self.tokenizer.batch_decode(input_ids)
            ground_sentences = self.tokenizer.batch_decode(output_ids)
            #masked_links_decoded = self.wikipedia_dataset.decode_compressed_entity_ids(masked_links)
            ground_links_decoded = self.wikipedia_dataset.decode_compressed_entity_ids(links)

            mask_token = self.tokenizer.mask_token_id

            _, token_logits, entity_logits = outputs

            # They are technically not logits but probability scores!
            token_proba = token_logits.detach().cpu()
            entity_proba = entity_logits.detach().cpu()

            token_outputs = torch.argmax(token_proba, 1)
            predicted_sentences = self.tokenizer.batch_decode(token_outputs)
            entity_outputs = torch.argmax(entity_proba, 1)
            predicted_links_decoded = self.wikipedia_dataset.decode_compressed_entity_ids(entity_outputs)

            # get the sentences where there is a masked token that is not properly classified
            positions = torch.nonzero(input_ids == mask_token, as_tuples=True)
            
            correct_tokens = output_ids[positions]
            correct_links = links[positions]
            chosen_tokens = token_outputs[positions]
            chosen_entities = token_outputs[positions]

            # small issue, we don't have the perplexity data 
            self.token_perplexity += float(torch.exp(self.loss(token_logits[positions], correct_tokens)))
            self.entity_perplexity += float(torch.exp(self.loss(entity_logits[positions], correct_links)))


            # pretty HTML printing
            for input_id, ground_sentence, predicted_sentence in \
                    zip(input_ids, ground_sentences, predicted_sentences):

                masked_tokens = torch.nonzero(input_id == mask_token).squeeze()

                masked_tok_idx = 0

                predicted_html = StringIO()
                expected_html = StringIO()

                for idx, (ground_tok, predicted_tok) in enumerate(zip(ground_sentence, predicted_sentence)):
                    if masked_tok_idx < len(masked_tokens) and idx == masked_tokens[masked_idx]:
                        if predicted_sentence[idx] != ground_sentences[idx]:
                            expected_html.write(f'<b style="text-color: red">{ground_tok}</b>')
                            predicted_html.write(f'<b style="text-color: red">{predicted_tok}</b>')
                        else:
                            expected_html.write(f'<b style="text-color: green">{ground_tok}</b>')
                            predicted_html.write(f'<b style="text-color: green">{predicted_tok}</b>')
                    else:
                        # We do not care about non-mask tokens. For what we know,
                        # the model may predict utter rubbish and we won't care except for the masked tokens
                        expected_html.write(ground_tok)
                        predicted_html.write(ground_tok)

                    expected_html.write(" ")
                    predicted_html.write(" ")

            # TODO: might be interesting to add an attention visualization graph
            # possible ideas: BertViz

    def compute(self, epoch: int) -> float:
        """
        """
        avg_loss = self.loss / self.dataloader_length

        return avg_loss

    def reset(self):
        super().reset()
        self.token_perplexity = 0.0
        self.entity_perplexity = 0.0


NUM_WORKERS = 1

np.random.seed(42)

def get_dataloaders(wikipedia_cbor):

    batch_size = 1
    is_dev = True
    if is_dev:
        wiki_dev_size = int(0.1*len(wikipedia_cbor))
    else:
        wiki_dev_size = len(wikipedia_cbor)
 
    wiki_dev_indices = np.random.choice(len(wikipedia_cbor), size=wiki_dev_size)

    # Use 80 % of Wikipedia for training, 19% for testing, 1% for validation
    wiki_train_size = int(0.8 * wiki_dev_size)
    wiki_validation_size = int(0.01 * wiki_dev_size)
    wiki_test_size = wiki_dev_size - wiki_train_size - wiki_validation_size

    wiki_train_indices, wiki_validation_indices, wiki_test_indices = (
        wiki_dev_indices[:wiki_train_size],
        wiki_dev_indices[wiki_train_size:-wiki_test_size],
        wiki_dev_indices[-wiki_test_size:])


    def get_dataloader(indices):
        sampler = SubsetRandomSampler(indices,
                                            generator=torch.Generator().manual_seed(42))
        return DataLoader(wikipedia_cbor, sampler=sampler,
                                    batch_size=batch_size,
                                    num_workers=NUM_WORKERS)

    wiki_train_dataloader = get_dataloader(wiki_train_indices)
    wiki_validation_dataloader = get_dataloader(wiki_validation_indices)
    wiki_test_dataloader = get_dataloader(wiki_test_indices)

    return (wiki_train_dataloader, wiki_validation_dataloader, wiki_test_dataloader)

def main():
    np.random.seed(42)

    wikipedia_cbor = WikipediaCBOR("wikipedia/car-wiki2020-01-01/enwiki2020.cbor", "wikipedia/car-wiki2020-01-01/partitions",
                                       #page_lim=wandb.config.wikipedia_article_nums,
    )

    wiki_train_dataloader, wiki_validation_dataloader, wiki_test_dataloader = get_dataloaders(wikipedia_cbor)

    pretraining_model = EntitiesAsExperts(4,
                                          8,
                                          wikipedia_cbor.max_entity_num,
                                          256)

    epochs = 2

    optimizer = get_optimizer(pretraining_model,
                                learning_rate=float(1e-4),
                                full_finetuning=False)
    scheduler = get_schedule(epochs, optimizer, wiki_train_dataloader)

    metric = PretrainingMetric(wiki_validation_dataloader)
    model_trainer = PretrainingModelTrainer(pretraining_model, watch_wandb=False)

    
    train_model(model_trainer, wiki_train_dataloader, wiki_validation_dataloader,
                wiki_test_dataloader, optimizer, scheduler, epochs, metric,
                validation_frequency=10,
                gradient_accumulation_factor=1)


    #save_models(pretraining_eae_100k=pretraining_model)

    """
    DEVICE = get_available_device()
    pretraining_model.eval()
    for batch in tqdm.tqdm(wiki_validation_dataloader):
        batch = [_.to(DEVICE) for _ in batch]

        loss, _ = pretraining_model(*batch)
        #loss.backward()
        #optimizer.step()
        #scheduler.ste