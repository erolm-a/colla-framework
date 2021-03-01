"""
Pretraining. This is a temporary debugging script and will be removed.
"""

from io import StringIO
from typing import List

import numpy as np
import pandas as pd
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
import wandb

from models.device import get_available_device

NUM_WORKERS = 0

class PretrainingModelTrainer(ModelTrainer):
    @staticmethod
    def load_from_dataloader(batch):
        return tuple(batch), tuple()


class PretrainingMetric(MetricWrapper):
    def __init__(self, dataloader: DataLoader, enable_wandb=False):
        super().__init__(dataloader, enable_wandb)
        self.wikipedia_dataset = dataloader.dataset # type: WikipediaCBOR
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.num_mask_labels = 0
        self.correctly_predicted_labels = 0
        # return a N-dimensional log-likelihood to compute the perplexity with...
        self.loss_fcn = CrossEntropyLoss()

    def add_batch(
        self,
        inputs: List[torch.Tensor],
        outputs: EntitiesAsExpertsOutputs,
        loss: float
    ):
        super().add_batch(inputs, outputs, loss)

        with torch.no_grad():

            input_ids, output_ids, masked_links, links = inputs[:4]

            mask_token = self.tokenizer.mask_token_id

            _, token_logits, entity_logits = outputs

            token_outputs = torch.argmax(token_logits, 1).detach().cpu()
            #entity_outputs = torch.argmax(entity_logits, 1).detach().cpu()

            ground_links_decoded = self.wikipedia_dataset.decode_compressed_entity_ids(links)
            predicted_links_decoded = self.wikipedia_dataset.decode_compressed_entity_ids(entity_outputs)

            # get the sentences where there is a masked token that is not properly classified
            positions = torch.nonzero(input_ids == mask_token, as_tuple=True)

            if positions.dim() > 0:
                self.num_mask_labels += len(positions)
            
            correct_tokens = output_ids[positions]
            correct_links = links[positions].cpu()

            token_logits = token_logits.cpu()
            entity_logits = entity_logits.cpu()

            self.token_perplexity += float(torch.exp(self.loss_fcn(token_logits[positions], correct_tokens)))
            #self.entity_perplexity += float(torch.exp(self.loss_fcn(entity_logits[positions], correct_links)))

            # pretty HTML printing
            
            masked_sentences = self.tokenizer.batch_decode(input_ids)
            expected_sentences = []
            predicted_sentences = []

            for input_id, output_id, token_outputs_sentence, input_entity_ in zip(input_ids, output_ids, token_outputs):
                ground_sentence = self.tokenizer.convert_ids_to_tokens(output_id)
                predicted_sentence = self.tokenizer.convert_ids_to_tokens(token_outputs_sentence)

                masked_tokens = torch.nonzero(input_id == mask_token).squeeze()
                masked_tok_idx = 0

                expected_html = StringIO()
                predicted_html = StringIO()

                for idx, (ground_tok, predicted_tok) in enumerate(
                    zip(ground_sentence, predicted_sentence)
                ):
                    if masked_tokens.dim() > 0 and masked_tok_idx < len(masked_tokens) \
                        and idx == bool(masked_tokens[masked_tok_idx]):
                        masked_tok_idx += 1
                        if predicted_sentence[idx] != ground_sentence[idx]:
                            expected_html.write(f'<b style="text-color: red">{ground_tok}</b>')
                            predicted_html.write(f'<b style="text-color: red">{predicted_tok}</b>')
                        else:
                            self.correctly_predicted_labels += 1
                            expected_html.write(f'<b style="text-color: green">{ground_tok}</b>')
                            predicted_html.write(f'<b style="text-color: green">{predicted_tok}</b>')
                    else:
                        # We do not care about non-mask tokens. For what we know,
                        # the model may predict utter rubbish and we won't care except for the masked tokens
                        expected_html.write(ground_tok)
                        predicted_html.write(ground_tok)

                    expected_html.write(" ")
                    predicted_html.write(" ")

                expected_sentences.append(expected_html.getvalue())
                predicted_sentences.append(predicted_html.getvalue())

            for input_id, masked_link_sentence, ground_link_sentence, predicted_link_sentence in \
                zip(input_ids, masked_links, ground_links_decoded, predicted_links_decoded):

                masked_links = torch.nonzero(masked_links == 0).squeeze()
                masked_link_idx = 0

                expected_link = StringIO()
                predicted_link = StringIO()

                
            if self.enable_wandb:
                example_toks_table = wandb.Table(
                    dataframe=pd.DataFrame({
                        "Token ground inputs: ": masked_sentences,
                        "Expected TokenPred Output": expected_sentences,
                        "Actual TokenPred Output": predicted_sentences
                        })
                    )

                wandb.log({"validation_example": example_toks_table})
            else:
                print("===========")
                print(expected_sentences)
                print(predicted_sentences)


            # TODO: might be interesting to add an attention visualization graph
            # possible ideas: BertViz

    def compute(self, epoch: int) -> float:
        """
        Compute the metric after the batches have been added.
        This call may call wandb to perform logging.
        """
        avg_loss = self.loss / self.dataloader_length
        accuracy = self.correctly_predicted_labels / self.num_mask_labels

        if self.enable_wandb:
            wandb.log({
                "val_loss": avg_loss,
                "token_ppl": self.token_perplexity,
                "token_acc": accuracy,
                #"entity_ppl": self.entity_perplexity,
                "epoch": epoch})

        return avg_loss


    def reset(self):
        super().reset()
        self.token_perplexity = 0.0
        self.entity_perplexity = 0.0


def get_dataloaders(wikipedia_cbor):

    batch_size = 1
    is_dev = True
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

def main():
    wandb.init(project="EaEPretraining")
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

    metric = PretrainingMetric(wiki_validation_dataloader, enable_wandb=True)
    model_trainer = PretrainingModelTrainer(pretraining_model, watch_wandb=True, enable_wandb=True)

    
    train_model(model_trainer, wiki_train_dataloader, wiki_validation_dataloader,
                wiki_test_dataloader, optimizer, scheduler, epochs, metric,
                validation_frequency=200,
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
    """
    
if __name__ == "__main__":
    main()