"""
An implementation of Entities As Experts.
"""

from copy import deepcopy
import json
from typing import cast, Any, Optional, Tuple, Union, NamedTuple

import torch
from torch.nn import Module, Dropout, Linear, CrossEntropyLoss, LayerNorm, NLLLoss, Sequential, \
    Parameter, GELU
import torch.nn.functional as F
from transformers import BertModel, BertConfig
import wandb

from .device import get_available_device

DEVICE = get_available_device()


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
            last_hidden_state: torch.Tensor,
            attention_mask: torch.Tensor,
            labels: Optional[torch.Tensor] = None):
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

                # pylint: disable=no-member
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
        self.E = Linear(self.N, self.d_ent, bias=False)
        # TODO: Do not make these hardcoded.
        # The BIO class used to hold these but it got deprecated...
        self.begin = 1
        self.inner = 2
        self.out = 0

        self.loss = NLLLoss()

    def _get_last_mention(self, bio_output, pos):
        end_mention = pos[1]

        for end_mention in range(pos[1] + 1, bio_output.size(1)):
            if bio_output[pos[0], end_mention] != self.inner:
                break
        end_mention -= 1

        return end_mention

    def forward(
        self,
        X,
        bio_output: torch.Tensor,
        entities_output: Optional[torch.Tensor],
        k=100
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        :param x the (raw) output of the first transformer block. It has a shape:
                B x N x (embed_size).
        :param bio_output the output of the bio classifier. If not provided no loss is returned
                (which is required during a training stage).
        :param entities_output the detected entities. If not provided no loss is returned
                (which is required during a training stage).
        :param k
        :returns a pair (loss, transformer_output). If either of entities_output or bio_output is
                  None loss will be None as well.
        """

        is_supervised = entities_output is not None
        assert not self.training or is_supervised is not None, \
            "Cannot perform training without entities_output"

        y = torch.zeros_like(X).to(DEVICE)

        # Disable gradient calculation for BIO outputs, but re-enable them
        # for the span
        with torch.no_grad():
            loss = None
            if is_supervised:
                loss = torch.zeros((1,)).to(DEVICE)

            begin_positions = torch.nonzero(bio_output == self.begin)

            # if no mentions are detected skip the entity memory.
            if len(begin_positions) == 0:
                return loss, y

            # FIXME: Not really parallelized (we don't have vmap yet...)
            end_positions = torch.tensor([
                self._get_last_mention(bio_output, pos) for pos in begin_positions]).unsqueeze(1).to(DEVICE)


        # Create the tensor so that it contains the batch position, the begin_positions
        # and the end positions in separate rows.
        positions = torch.cat([begin_positions, end_positions], 1).T

        first = X[positions[0], positions[1]]
        second = X[positions[0], positions[2]]

        mention_span = torch.cat([first, second], 1).to(DEVICE)

        pseudo_entity_embedding = self.W_f(
            mention_span)  # num_of_mentions x d_ent

        if is_supervised:
            alpha = F.softmax(
                pseudo_entity_embedding.matmul(self.E.weight), dim=1)

            alpha_log = torch.log(alpha)
            # Compared to the original paper we use NLLoss.
            # Gradient-wise this should not change anything
            loss = self.loss(
                alpha_log, entities_output[positions[0], positions[1]])
            
        if self.training:
            # shape: B x d_ent
            picked_entity = self.E(alpha)

        else:
            # K nearest neighbours
            # Note: the paper makes a slight notation abuse.
            # When computing the query vector, alpha is the softmax of the topk entities
            # When computing the loss, alpha is the softmax across the whole dictionary
            topk = torch.topk(pseudo_entity_embedding @ self.E.weight, k, dim=1)

            alpha_topk = F.softmax(topk.values, dim=1)

            # mat1 has size (M x d_ent x k), mat2 has size (M x k x 1)
            # the result has size (M x d_ent x 1). Squeeze that out and we've got our
            # entities of size (M x d_ent).
            picked_entity = torch.bmm(
                self.E.weight[:, topk.indices].transpose(0, 1),
                alpha_topk.view((-1, k, 1))).view((-1, self.d_ent))

        y[positions[0], positions[1]] = self.W_b(picked_entity)

        return loss, y


class EaELinearWithLayerNorm(Module):
    """
    Combine a linear layer with an activation function and a layernorm.
    This is an intermediate step that can be used shortly after EaEPredictionHead
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.classifier_head = Sequential(
            Linear(config.hidden_size, config.hidden_size),
            GELU(),
            LayerNorm(config.hidden_size, eps=config.layer_norm_eps))

    def forward(self, hidden_states: torch.Tensor):
        return self.classifier_head(hidden_states)


class EaEPredictionHead(Module):
    """
    A reimplementation of :class `BertLMPredictionHead` that is generalizable
    """

    def __init__(self, config: BertConfig, output_size: int):
        super().__init__()
        self.transform = EaELinearWithLayerNorm(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = Linear(config.hidden_size, output_size, bias=False)

        self.bias = Parameter(torch.zeros(output_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class TokenPredHead(Module):
    """
    General Token prediction head.
    It is a common building block for both TokenPred (output_head_number = vocab size) and EntityPred
    (output_head_number = entity vocab size)
    """

    def __init__(self, config: BertConfig, output_head_number: int):
        super().__init__()
        # TODO: create a new EaEConfig struct
        self.output_head_number = output_head_number
        self.cls = EaEPredictionHead(config, output_head_number)

    def forward(
        self,
        sequence_output: torch.Tensor,
        labels: Optional[torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        :param sequence_output the output of the attention block
        :param labels the wished labels. The domain size must be output_head_number
        """
        prediction_scores = self.cls(sequence_output)

        loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token ?
            loss = loss_fct(prediction_scores.view(-1,
                                                   self.output_head_number), labels.view(-1))

        return loss, prediction_scores


class EntityPred(TokenPredHead):
    """
    Entity Predictor head
    """

    def __init__(self, config: BertConfig, entity_vocab_size: int):
        # TODO: create a new EaEConfig struct
        super().__init__(config, entity_vocab_size)


class TokenPred(TokenPredHead):
    """
    Token Prediction head for EaE
    """

    def __init__(self, config: BertConfig):
        super().__init__(config, config.vocab_size)


class EntitiesAsExpertsOutputs(NamedTuple):
    hidden_attention: torch.Tensor
    token_prediction_scores: torch.Tensor
    entity_prediction_scores: torch.Tensor


class EntitiesAsExperts(Module):
    """
    This is the Entities As Experts implementation. Similarly to Transformers' Bert,
    task-specific heads should be built on top of this class.
    """

    def __init__(
            self,
            l0: int,
            l1: int,
            entity_size: int,
            entity_embedding_size=256,
            bert_model_variant="bert-base-uncased"
    ):
        """
        :param bert_model: a pretrained bert instance that can perform Masked LM.
                Required for TokenPred
        :param l0 the number of layers to hook the Entity Memory to
        :param l1 the remaining number of layers to use for TokenPred
        :param entity_size the number of entities to store
        :param entity_embedding_size the size of an entity embedding
        """
        super().__init__()

        self.bert_model_variant = bert_model_variant

        bert_model = BertModel.from_pretrained(bert_model_variant)
        self.first_block = TruncatedModel(bert_model, l0)
        self.second_block = TruncatedModelSecond(
            bert_model, l1)
        self._config = bert_model.config
        self.entity_memory = EntityMemory(self._config.hidden_size, entity_size,
                                          entity_embedding_size)
        self.bioclassifier = BioClassifier(self._config)
        self.tokenpred = TokenPred(self._config)
        self.entitypred = EntityPred(self._config, entity_size)
        self.layernorm = LayerNorm(768)

        self.loss_fct = CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        output_ids: Optional[torch.Tensor] = None,
        entity_inputs: Optional[torch.Tensor] = None,
        # not clear if we need to perform Entity Prediction and supervise with that
        entity_outputs: Optional[torch.Tensor] = None,
        mention_boundaries: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,

    ):
        """
        :param input_ids the masked tokenized input of shape B x 512
        :param attention_mask the attention mask of shape B x 512
        :param mention_boundaries the BIO output labels of shape B x 512
        :param output_ids the unmasked tokenized input of shape B x 512
        :param token_type_ids the token type mask that differentiates between CLS and SEP zones.
               Used for QA.
        :param return_logits if true return the scores of the token prediction and entity prediction
               heads.

        :returns a triplet loss, logits (for token_pred) and output of the last transformer block
        """

        compute_loss = mention_boundaries is not None and entity_inputs is not None

        first_block_outputs = self.first_block(input_ids,
                                               token_type_ids=token_type_ids,
                                               attention_mask=attention_mask,
                                               output_hidden_states=True)

        hidden_attention = first_block_outputs[0]
        bio_loss, bio_outputs = self.bioclassifier(hidden_attention,
                                                   attention_mask=attention_mask,
                                                   labels=mention_boundaries)

        if not compute_loss:
            bio_choices = torch.argmax(bio_outputs, 2)
            mention_boundaries = bio_choices

        entity_loss, entity_inputs = self.entity_memory(
            hidden_attention, mention_boundaries, entity_inputs)

        bert_output = self.second_block(self.layernorm(entity_inputs + hidden_attention),
                                        encoder_attention_mask=attention_mask)

        token_pred_loss, token_prediction_scores = self.tokenpred(
            bert_output.last_hidden_state, output_ids)
        entity_pred_loss, entity_prediction_scores = self.entitypred(
            bert_output.last_hidden_state, entity_outputs)

        loss = None

        if compute_loss:
            loss = entity_loss + bio_loss + token_pred_loss + entity_pred_loss

        return (loss, EntitiesAsExpertsOutputs(
            bert_output,
            token_prediction_scores,
            entity_prediction_scores
        ))

    @staticmethod
    def from_pretrained(config: str, run_id: str):
        """
        Load a pretrained model. Probe wandb to get the right checkpoint to use

        :param config the configuration to use
        :run_id the run identifier that states where it has been saved.
        """

        checkpoint_path = config + ".h5"
        model_config_path = config + ".json"

        model_json = wandb.restore(model_config_path, run_id)

        assert model_json is not None, "Could not find the required run"
        config_dict = json.load(model_json)
        checkpoint = wandb.restore(checkpoint_path, run_id)  # type: Any

        model = EntitiesAsExperts(**config_dict)

        model.load_state_dict(torch.load(checkpoint.buffer.raw,
                                         map_location=torch.device('cpu')))

        return model

    @property
    def config(self):
        """
        A configuration dict to be used when loading or saving a model.
        """
        # TODO: extract from the constructor
        return {
            "l0": 4,
            "l1": 8,
            "bert_model_variant": self.bert_model_variant,
            "entity_size": 30703,
            "entity_embedding_size": 256,
            "hidden_size": self._config.hidden_size
        }


class EaEForQuestionAnswering(Module):
    """
    The Entity as Expert model for Question Answering.
    Inspired from HF's BertForQuestionAnswering but simplified.
    """

    def __init__(self,
                 eae: EntitiesAsExperts):
        """
        :param pretrained_model the pretrained model.
        """
        super().__init__()
        self.eae = eae
        self.eae.training = False
        self.qa_outputs = Linear(eae.config["hidden_size"], 2)
        self.config = eae.config

        # Freeze the entity memory
        for param_weight in self.eae.entity_memory.parameters():
            param_weight.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        start_positions: Optional[torch.Tensor],
        end_positions: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        :param input_ids the input ids
        :param attention_mask
        :param token_type_ids to distinguish between context and question
        :param start_positions the target start positions
        :param end_positions the target end positions

        Everything else will be ignored.

        :returns a triple loss, start_logits, end_logits
        """

        # Copycat from BertForQuestionAnswering.forward()
        # Basically a logit on 2 terms with appropriate clamping
        _, hidden_states, *__ = self.eae(
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
