import torch

import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoConfig
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, SequenceClassifierOutputWithPast, TokenClassifierOutput
from transformers.models.llama import LlamaPreTrainedModel, LlamaModel, LlamaConfig
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from typing import Optional

from .configuration_myrm import MyRMConfig

logger = logging.get_logger(__name__)


class Model(nn.Module):
    def __init__(self, input_size, hidden_dim_str):
        super(Model, self).__init__()
        hidden_dims = [input_size] + list(map(int, hidden_dim_str.split(',')))
        self.layers = nn.ModuleList(
            nn.Linear(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1)
        )
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        for layer in self.layers:
            x = torch.nn.functional.leaky_relu(layer(x))
        x = self.output_layer(x)
        return x


AutoConfig.register("myrm", MyRMConfig)

class MyRMForTokenClassification(LlamaPreTrainedModel):
    base_model_prefix = "model"
    config: MyRMConfig

    def __init__(self, config: MyRMConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        # Similar to `self.model = AutoModel.from_config(config)` but allows to change the base model name if needed in the child class
        base_config = LlamaConfig.from_dict(config.to_dict())
        setattr(self, self.base_model_prefix, AutoModel.from_config(base_config))
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        ## We need to replace the self.score with our own score
        ## Make sure the reward model we load has the same Class name with this one
        self.myscore = Model(config.hidden_size, config.hidden_dim_str)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> TokenClassifierOutput:
        outputs: BaseModelOutputWithPast = getattr(self, self.base_model_prefix)(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        sequence_output = outputs.last_hidden_state
        # sequence_output = self.dropout(sequence_output)
        # logits = self.score(sequence_output)
        last_token_indices = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(sequence_output.shape[0], device=sequence_output.device)
        last_token_indices = last_token_indices.to(sequence_output.device)
        sequence_output = sequence_output[batch_indices, last_token_indices]

        logits = self.myscore(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["MyRMForTokenClassification"]