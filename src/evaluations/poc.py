"""
Pretraining. This is a temporary debugging script and will be removed.
"""

import math
from io import StringIO
from typing import List

import pprint

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import tqdm
from transformers import AutoTokenizer

from tools.dataloaders import WikipediaCBOR

from models import EntitiesAsExperts, EntitiesAsExpertsOutputs
from models.training import (train_model, get_optimizer, get_schedule,
                            MetricWrapper, ModelTrainer)

# TODO: can we avoid recomputing the cross-entropy loss again?
from torch.nn import CrossEntropyLoss
import wandb

NUM_WORKERS = 16

class PretrainingModelTrainer(ModelTrainer):
    @staticmethod
    def load_from_dataloader(batch):
        return tuple(batch), tuple()


class PretrainingMetric(MetricWrapper):
    def __init__(
        self,
        dataloader: DataLoader,
        enable_wandb=False,
        enable_example_wandb=False
    ):
        super().__init__(dataloader, enable_wandb)
        self.wikipedia_dataset = dataloader.dataset # type: WikipediaCBOR
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        # for computing the token perplexity
        self.loss_fcn = CrossEntropyLoss(reduction='sum')
        self.enable_example_wandb = enable_example_wandb

    def add_batch(
        self,
        inputs: List[torch.Tensor],
        outputs: EntitiesAsExpertsOutputs,
        loss: float,
    ):
        with torch.no_grad():
            self.loss += loss

            input_ids, output_ids, masked_links, links = inputs[:4]

            mask_token = self.tokenizer.mask_token_id

            _, token_logits, entity_logits = outputs

            token_outputs = torch.argmax(token_logits, 2).detach().cpu()
            entity_outputs = torch.argmax(entity_logits, 2).detach().cpu()

            #ground_links_decoded = self.wikipedia_dataset.decode_compressed_entity_ids(links)
            #predicted_links_decoded = self.wikipedia_dataset.decode_compressed_entity_ids(entity_outputs)

            # get the sentences where there is a masked token that is not properly classified
            token_mask_positions = torch.nonzero(input_ids == mask_token, as_tuple=True)

            self.num_mask_labels += len(token_mask_positions[0])
            self.correctly_predicted_labels += len(torch.nonzero(
                token_outputs[token_mask_positions] == output_ids[token_mask_positions]
            ))

            correct_tokens = output_ids[token_mask_positions]
            #correct_links = links[token_mask_positions].cpu()

            token_logits = token_logits.cpu()
            # entity_logits = entity_logits.cpu()

            # If no masks are detected the loss function yields a NaN.
            if len(token_mask_positions[0]) > 0:
                self.token_perplexity += float(
                    self.loss_fcn(token_logits[token_mask_positions],
                                  correct_tokens))

            # TODO: masked entities should use a different id than 0
            link_mask_positions = torch.nonzero((links > 0) & (masked_links == 0), as_tuple=True)
            self.num_mask_links += len(link_mask_positions[0])
            self.correctly_predicted_links += len(torch.nonzero(
                entity_outputs[link_mask_positions] == links[link_mask_positions]))

            if self.is_validation:
                self.log_example(input_ids, output_ids, token_outputs)


            # TODO: might be interesting to add an attention visualization graph
            # possible ideas: BertViz

    def log_example(
        self,
        input_ids: torch.Tensor,
        output_ids: torch.Tensor,
        token_outputs: torch.Tensor
    ):
        expected_sentences = []
        predicted_sentences = []
        mask_token = self.tokenizer.mask_token_id
        masked_sentences = self.tokenizer.batch_decode(input_ids)
        self.masked_sentences.extend(masked_sentences)

        for input_id, output_id, token_outputs_sentence in \
            zip(input_ids, output_ids, token_outputs):

            ground_sentence = self.tokenizer.convert_ids_to_tokens(
                output_id)
            predicted_sentence = self.tokenizer.convert_ids_to_tokens(
                token_outputs_sentence)

            masked_tokens = torch.nonzero(
                input_id == mask_token).squeeze()
            masked_tok_idx = 0

            expected_html = StringIO()
            predicted_html = StringIO()

            for idx, (ground_tok, predicted_tok) in enumerate(
                zip(ground_sentence, predicted_sentence)
            ):
                if masked_tokens.dim() > 0 and masked_tok_idx < len(masked_tokens):
                    if idx == int(masked_tokens[masked_tok_idx]):
                        masked_tok_idx += 1
                        if predicted_sentence[idx] != ground_sentence[idx]:
                            expected_html.write(
                                f'<b style="color: red">{ground_tok}</b>')
                            predicted_html.write(
                                f'<b style="color: red">{predicted_tok}</b>')
                        else:
                            expected_html.write(
                                f'<b style="color: green">{ground_tok}</b>')
                            predicted_html.write(
                                f'<b style="color: green">{predicted_tok}</b>')
                    elif masked_tokens[masked_tok_idx] != self.tokenizer.pad_token_id:
                        # We do not care about non-mask tokens. For what we know,
                        # the model may predict utter rubbish and we won't care except
                        # for the masked tokens
                        expected_html.write(ground_tok)
                        predicted_html.write(ground_tok)

                expected_html.write(" ")
                predicted_html.write(" ")

            """
            for input_id, masked_link_sentence, ground_link_sentence, predicted_link_sentence in \
                zip(input_ids, masked_links, ground_links_decoded, predicted_links_decoded):

                masked_links = torch.nonzero(masked_links == 0).squeeze()
                masked_link_idx = 0

                expected_link = StringIO()
                predicted_link = StringIO()
            """

            expected_sentences.append(expected_html.getvalue())
            predicted_sentences.append(predicted_html.getvalue())

        self.expected_sentences.extend(expected_sentences)
        self.predicted_sentences.extend(predicted_sentences)

    def compute(self, epoch: int) -> float:
        """
        Compute the metric after the batches have been added.
        This call may call wandb to perform logging.
        """

        total_length = self.dataloader_length * self.dataloader.batch_size
        avg_loss = self.loss / total_length
        prefix = "val_" if self.is_validation else "test_"

        token_accuracy = self.correctly_predicted_labels / self.num_mask_labels \
            if self.num_mask_labels else 0.0

        entity_accuracy = self.correctly_predicted_links / self.num_mask_links \
            if self.num_mask_links else 0.0

        token_perplexity = math.exp(self.token_perplexity / self.num_mask_labels) \
            if self.num_mask_labels else 0.0

        payload = {
                f"{prefix}loss": avg_loss,
                f"{prefix}token_ppl": token_perplexity,
                f"{prefix}token_acc": token_accuracy,
                f"{prefix}entity_acc": entity_accuracy,
                "epoch": epoch
            }

        if self.enable_wandb:
            wandb.log(payload)

            # subsample the examples
            masked_sentences, expected_sentences, predicted_sentences = [x[::25] for x in (self.masked_sentences, self.expected_sentences, self.predicted_sentences)]
            if self.is_validation and len(expected_sentences) > 0:
                if self.enable_example_wandb:
                    art = wandb.Artifact("validation_metrics", type="evaluation")
                    table = wandb.Table(columns=["Masked token inputs",
                        "Expected Output", "Actual TokenPred Output"])
                    for record in zip(masked_sentences, expected_sentences,
                                    predicted_sentences):
                        table.add_data(*map(wandb.Html, record))

                    art.add(table, "html")

                    wandb.log_artifact(art)
        else:
            print("===========")
            pprint.pprint(payload)

        pprint.pprint(self.expected_sentences)
        pprint.pprint(self.predicted_sentences)

        return avg_loss


    def reset(self, is_validation: bool):
        super().reset(is_validation)
        self.loss = 0.0
        self.token_perplexity = 0.0
        self.num_mask_labels = 0
        self.correctly_predicted_labels = 0
        self.num_mask_links = 0
        self.correctly_predicted_links = 0
        self.expected_sentences = []
        self.predicted_sentences = []
        self.masked_sentences = []


def get_dataloaders(wikipedia_cbor: WikipediaCBOR, batch_size: int, is_dev: bool):
    if is_dev:
        wiki_dev_size = int(0.01*len(wikipedia_cbor))
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

ENABLE_WANDB = True
def main():
    np.random.seed(42)

    if ENABLE_WANDB:
        wandb.init(project="EaEPretraining", config="configs/eae_pretraining.yaml", job_type="pretraining")
        wikipedia_article_nums = wandb.config.wikipedia_article_nums
        wikipedia_cutoff_frequency = wandb.config.wikipedia_cutoff_frequency
        batch_size = wandb.config.batch_size
        is_dev = wandb.config.is_dev
        gradient_accum_size = wandb.config.gradient_accum_size
        learning_rate = wandb.config.learning_rate
        full_finetuning = wandb.config.full_finetuning
        epochs = wandb.config.pretraining_epochs
        l0 = wandb.config.l0
        l1 = wandb.config.l1
        entity_embedding_size = wandb.config.entity_embedding_size
    else:
        wikipedia_article_nums = 100000
        wikipedia_cutoff_frequency = 0.03
        batch_size = 1
        gradient_accum_size = 1
        is_dev = True
        l0 = 4
        l1 = 8
        entity_embedding_size = 256
        learning_rate = 1e-4
        full_finetuning = False
        epochs = 1



    wikipedia_cbor = WikipediaCBOR("wikipedia/car-wiki2020-01-01/enwiki2020.cbor",
                                   "wikipedia/car-wiki2020-01-01/partitions",
                                   page_lim=wikipedia_article_nums,
                                   cutoff_frequency=wikipedia_cutoff_frequency,
                                   #clean_cache=True
                                   #recount=True
                                   )

    wiki_train_dataloader, wiki_validation_dataloader, wiki_test_dataloader = \
        get_dataloaders(wikipedia_cbor, batch_size, is_dev)

    pretraining_model = EntitiesAsExperts(l0,
                                          l1,
                                          wikipedia_cbor.max_entity_num,
                                          entity_embedding_size)


    optimizer = get_optimizer(pretraining_model,
                                learning_rate=learning_rate,
                                full_finetuning=full_finetuning)
    scheduler = get_schedule(epochs, optimizer, wiki_train_dataloader)

    metric = PretrainingMetric(wiki_validation_dataloader, enable_wandb=ENABLE_WANDB)
    model_trainer = PretrainingModelTrainer(pretraining_model, "pretraining_10k", watch_wandb=ENABLE_WANDB, enable_wandb=ENABLE_WANDB)


    train_model(model_trainer, wiki_train_dataloader, wiki_validation_dataloader,
                wiki_test_dataloader, optimizer, scheduler, epochs, metric,
                validation_frequency= 500 * batch_size,
                gradient_accumulation_factor=gradient_accum_size,
                checkpoint_frequency=0)


    # ISSUE: can't run torchscript tracing on heterogeneous data structures (e.g. classes)
    # ... but there should be none, right? What about Bert's default stuff?
    # model_trainer.save_models(model_trainer.run_name + "_torchscript_test", network_format="torchscript")

if __name__ == "__main__":
    main()
