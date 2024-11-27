from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Phi3Config, Phi3Model, Phi3ForCausalLM
from llava.utils import rank0_print
from ..utils import get_loss_fn

class LlavaPhiConfig(Phi3Config):
    model_type = "llava_phi"

class LLavaPhiModel(LlavaMetaModel, Phi3Model):
    config_class = LlavaPhiConfig

    def __init__(self, config: Phi3Config):
        super(LLavaPhiModel, self).__init__(config)


class LlavaPhiForCausalLM(Phi3ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaPhiConfig

    def __init__(self, config):
        Phi3ForCausalLM.__init__(self, config)
        config.model_type = "llava_phi"
        config.rope_scaling = None
        
        self.model = LLavaPhiModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

        # Initialize weights and apply final processing
        self.post_init()
        
        # define kl_loss
        self.bce_loss = nn.BCEWithLogitsLoss()

        # add some variables for retrieval eval
        self.eval_pointer = 0
        self.eval_acc = 0
        self.eval_samples = 0

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            sen_sim: Optional[torch.FloatTensor] = None,
            lb_sim: Optional[torch.FloatTensor] = None,
            sentences: Optional[List[str]] = None,
            ids: Optional[List[str]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            image_sizes: Optional[List[List[int]]] = None,
            return_dict: Optional[bool] = None,
            modalities: Optional[List[str]] = ["image"],
            dpo_forward: Optional[bool] = None,
            cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
                
        if inputs_embeds is None:
            if 'infer' in self.model.config.train_step:
                if 'longlong' in self.model.config.train_step:
                    longlong=True
                else:
                    longlong=False
                (input_ids, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal_infer(input_ids, position_ids, attention_mask, None, None, sentences, images, modalities, image_sizes, longlong=longlong)
            else:
                (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, resampler_cls, sim_score, ids) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, sen_sim, lb_sim, sentences, ids, images, modalities, image_sizes)

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            
            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels
        
        else:
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            
            if 'infer' in self.model.config.train_step:
                return outputs
            
            sim_loss, current_acc, correctv2 = get_loss_fn(resampler_cls=resampler_cls, sim_score=sim_score, loss_fn=self.bce_loss, weight_ce=self.config.weight_ce, train_step=self.config.train_step)
            self.eval_acc += correctv2
            if self.config.train_step == 'sft':
                self.eval_samples += len(resampler_cls)
            else:
                self.eval_samples += resampler_cls[0].size(0)

            avg_acc = None
            if self.eval_pointer % 100 == 0:
                avg_acc = self.eval_acc / self.eval_samples
                rank0_print(f"[*] Retrieval Accuracy for recent 100 batches : {avg_acc}")
                self.eval_acc = 0
                self.eval_samples = 0
            self.eval_pointer += 1
            return outputs, sim_loss, current_acc, avg_acc
            
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        sentences:  Optional[List[str]] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        modalities = kwargs.pop("modalities", None) if "modalities" in kwargs and modalities is None else modalities
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if 'longlong' in self.model.config.train_step:
            longlong=True
        else:
            longlong=False

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal_infer(inputs, position_ids, attention_mask, None, None, sentences, images, modalities, image_sizes, longlong=longlong)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
               
        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs

AutoConfig.register("llava_phi", LlavaPhiConfig)
AutoModelForCausalLM.register(LlavaPhiConfig, LlavaPhiForCausalLM)