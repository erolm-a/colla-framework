"""
An implementation of Entity As Expert.
"""

from typing import Optional
from copy import deepcopy

import torch
from torch.nn import Module, Dropout, Linear, CrossEntropyLoss, LayerNorm
import torch.nn.functional as F

from transformers import BertForMaskedLM

from .device import DEVICE


class TruncatedEncoder(Module):
    """
    Like BertEncoder, but with only the first (or last) l0 layers
    """
    def __init__(self, encoder, l0: int, is_first=True):
        super().__init__()
        __doc__ = encoder.__doc__
        self.encoder = deepcopy(encoder)

        if is_first:
            self.encoder.layer = self.encoder.layer[:l0]
        else:
            self.encoder.layer = self.encoder.layer[l0:]

    def forward(self, *args, **kwargs):
        __doc__ = self.encoder.forward.__doc__
        return self.encoder(*args, **kwargs)

# TODO: should we replace this part?


class TruncatedModel(Module):
    def __init__(self, model, l0: int):
        super().__init__()
        self.model = deepcopy(model)
        self.model.encoder = TruncatedEncoder(self.model.encoder, l0)

    def forward(self, *args, **kwargs):
        __doc__ = self.model.forward.__doc__
        return self.model(*args, **kwargs)


class TruncatedModelSecond(Module):
    def __init__(self, model, l1: int):
        super().__init__()
        self.encoder = TruncatedEncoder(model.encoder, l1, False)

    def forward(self, *args, **kwargs):
        # same prototipe of BertEncoder
        return self.encoder(*args, **kwargs)


class BioClassifier(Module):
    """
    BIO classifier head
    """

    def __init__(self,  config):
        super().__init__()
        self.config = config
        self.dropout = Dropout(config.hidden_dropout_prob)
        self.num_labels = 3
        self.classifier = Linear(
            config.hidden_size, out_features=self.num_labels)

    def forward(
            self,
            last_hidden_state: torch.tensor,
            attention_mask: torch.tensor,
            labels: Optional[torch.tensor] = None):
        """
        :param last_hidden_state the state returned by BERT.
        """

        # Copycat from BertForTokenClassification.forward() .
        # Since we already have a hidden state as input we can easily
        # skip the first 2/3 statements. And also simplify the code here and there.
        sequence_output = self.dropout(last_hidden_state)
        logits = self.classifier(sequence_output)

        loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    # pylint: disable=not-callable
                    active_loss, labels.view(-1), torch.tensor(
                        loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))

        return loss, logits


class EntityMemory(Module):
    """
    Entity Memory, as described in the paper
    """

    def __init__(self, embedding_size: int, entity_size: int,
                 entity_embedding_size: int):
        """
        :param embedding_size the size of an embedding. In the EaE paper it is called d_emb, previously as d_k
            (attention_heads * embedding_per_head)
        :param entity_size also known as N in the EaE paper, the maximum number of entities we store
        :param entity_embedding_size also known as d_ent in the EaE paper, the embedding of each entity
        
        """
        super().__init__()
        # pylint:disable=invalid-name
        self.N = entity_size
        self.d_emb = embedding_size
        self.d_ent = entity_embedding_size
        # pylint:disable=invalid-name
        self.W_f = Linear(2*embedding_size, self.d_ent)
        # pylint:disable=invalid-name
        self.W_b = Linear(self.d_ent, embedding_size)
        # pylint:disable=invalid-name
        self.E = Linear(self.N, self.d_ent)
        # TODO: Do not make these hardcoded.
        # The BIO class used to hold these but it got deprecated...
        self.begin = 1
        self.inner = 2
        self.out = 0

    def forward(
        self,
        X,
        bio_output: Optional[torch.LongTensor],
        entities_output: Optional[torch.LongTensor],
        k=100
    ) -> (torch.tensor, torch.tensor):
        """
        :param x the (raw) output of the first transformer block. It has a shape:
                B x N x (embed_size). If not provided no loss is returned
                (which is required during a training stage).
        :param entities_output the detected entities. If not provided no loss is returned
                (which is required during a training stage).
        
        
        :returns a pair (loss, transformer_output). If either of entities_output or bio_output is
                  None loss will be None as well.
        """

        calculate_loss = bio_output is not None and entities_output is not None

        begin_positions = torch.nonzero(bio_output == self.begin)

        y = torch.zeros_like(X).to(DEVICE)

        if calculate_loss:
            loss = torch.zeros((1,)).to(DEVICE)
        else:
            loss = None

        for pos in begin_positions:
            end_mention = pos[1]
            while end_mention < self.d_emb and bio_output[pos[0], end_mention] == self.inner:
                end_mention += 1
            end_mention -= 1

            first = X[pos[0], pos[1]]
            second = X[pos[0], end_mention]

            mention_span = torch.cat([first, second], 0).to(DEVICE)
            pseudo_entity_embedding = self.W_f(mention_span)  # d_ent

            # Not sure why Pylint thinks self.train is a constant
            # pylint: disable=using-constant-test
            if self.train:
                alpha = F.softmax(self.E.weight.T.matmul(
                    pseudo_entity_embedding), dim=0)

            else:
                # K nearest neighbours
                topk = torch.topk(self.E.weight.T.matmul(
                    pseudo_entity_embedding), k)
                alpha = F.softmax(topk.values, dim=0)
            picked_entity = self.E.weight.matmul(alpha)

            y[pos[0], pos[1]] = self.W_b(picked_entity)

            if calculate_loss:
                loss += alpha[entities_output[pos[0], pos[1]]]

        if calculate_loss:
            return loss, y

        return y


class TokenPred(Module):
    """
    Just a mere wrapper on top of BertForMaskedLM
    """

    def __init__(self, masked_language_model: BertForMaskedLM):
        """
        :param masked_language_model an instance of a BertForMaskedLM
        """
        super().__init__()
        self.cls = deepcopy(masked_language_model.cls)

    def forward(self, *args, **kwargs):
        __doc__ = self.cls.forward.__doc__
        return self.cls(*args, **kwargs)


class Pretraining(Module):
    """
    This is a mere wrapper used to pretrain contemporarely a bio classifier and EntityMemory.

    We also need TokenPred but we'll ignore it at the moment.
    """

    def __init__(
            self,
            bert_masked_language_model: BertForMaskedLM,
            l0: int,
            l1: int,
            entity_size: int,
            entity_embedding_size=256):
        """
        :param bert_masked_language_model: a pretrained bert instance that can perform Masked LM.
                Required for TokenPred
        :param l0 the number of layers to hook the Entity Memory to
        :param l1 the remaining number of layers to use for TokenPred
        :param entity_size the number of entities to store
        :param entity_embedding_size the size of an entity embedding
        """
        super().__init__()

        self.first_block = TruncatedModel(bert_masked_language_model.bert, l0)
        self.second_block = TruncatedModelSecond(
            bert_masked_language_model.bert, l1)
        self.config = bert_masked_language_model.config
        self.entity_memory = EntityMemory(self.config.hidden_size, entity_size,
                                          entity_embedding_size)
        self.bioclassifier = BioClassifier(self.config)
        self.tokenpred = TokenPred(bert_masked_language_model)
        self.layernorm = LayerNorm((768,))

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        entity_outputs: Optional[torch.LongTensor] = None,
        mention_boundaries: Optional[torch.LongTensor] = None,
        output_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None
    ):
        """
        :param input_ids the masked tokenized input of shape B x 512
        :param attention_mask the attention mask of shape B x 512
        :param mention_boundaries the BIO output labels of shape B x 512
        :param output_ids the unmasked tokenized input of shape B x 512
        :param token_type_ids the token type mask that differentiates between CLS and SEP zones.
            Used for QA.

        :returns a triplet loss, logits (for token_pred) and output of the last transformer block
        """

        compute_loss = mention_boundaries is not None and entity_outputs is not None

        first_block_outputs = self.first_block(input_ids,
                                               token_type_ids=token_type_ids,
                                               attention_mask=attention_mask,
                                               output_hidden_states=True)

        X = first_block_outputs[0]
        bio_outputs = self.bioclassifier(X, attention_mask=attention_mask,
                                         labels=mention_boundaries)

        if compute_loss:
            bio_loss = bio_outputs[0]

        bio_choices = torch.argmax(bio_outputs[1], 2)
        entity_memory_outputs = self.entity_memory(
            X, bio_choices, entity_outputs)
        if compute_loss:
            entity_loss, entity_outputs = entity_memory_outputs
        else:
            entity_outputs = entity_memory_outputs

        X = self.second_block(self.layernorm(entity_outputs + X),
                              encoder_attention_mask=attention_mask)

        token_prediction_scores = self.tokenpred(X.last_hidden_state)

        # calculate loss for token prediction
        # Abdridged from transformers's doc
        if compute_loss:
            loss_fct = CrossEntropyLoss()
            token_pred_loss = loss_fct(
                token_prediction_scores.view(-1, self.config.vocab_size),
                output_ids.view(-1))
            loss = entity_loss + bio_loss + token_pred_loss
        else:
            loss = None

        return loss, token_prediction_scores, X


class EaEForQuestionAnswering(Module):
    """
    The Entity as Expert model for Question Answering.
    Inspired from HF's BertForQuestionAnswering but simplified.
    """

    def __init__(self,
                 eae: Pretraining):
        """
        :param pretrained_model the pretrained model.
        """
        super().__init__()
        self.eae = eae
        self.qa_outputs = Linear(eae.config.hidden_size, 2)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        token_type_ids: torch.LongTensor,
        start_positions: Optional[torch.LongTensor],
        end_positions: Optional[torch.LongTensor]
    ):
        """
        :param input_ids the input ids
        :param attention_mask
        :param token_type_ids to distinguish between context and question
        :param start_positions the target start positions
        :parma end_positions the target end positions
        """

        # Copycat from BertForQuestionAnswering.forward()
        # Basically a logit on 2 terms with appropriate clamping
        _, __, hidden_states = self.eae(
            input_ids, attention_mask, token_type_ids=token_type_ids)
        logits = self.qa_outputs(hidden_states.last_hidden_state)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return total_loss, start_logits, end_logits
