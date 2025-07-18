from typing import List, Optional, Tuple, Union, Generator
import re
import time
import timeit
import pdb
import torch
import yaml
import torch.nn as nn
from transformers.cache_utils import DynamicCache

from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from omni_speech.model.language_model.omni_speech_qwen2 import OmniSpeechQwen2ForCausalLM
from omni_speech.model.speech_generator.builder import build_speech_generator
from omni_speech.model.speech_generator.generation import GenerationWithCTC
from omni_speech.constants import IGNORE_INDEX, SPEECH_TOKEN_INDEX, PAD_TOKEN_ID_QWEN2_5

SENTENCE_DELIM_RE = re.compile(r'[。：？！.?!\n]$')

class OmniSpeech2SConfig(Qwen2Config):
    model_type = "omni_speech2s_qwen"

class OmniSpeech2SQwen2ForCausalLM(OmniSpeechQwen2ForCausalLM, GenerationWithCTC):
    config_class = OmniSpeech2SConfig

    def __init__(self, config, tokenizer=None):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.tune_speech_generator_only = True
        
        if hasattr(config, "speech_generator_type"):
            self.speech_generator = build_speech_generator(config)
        else:
            self.speech_generator = None
            
        self.reset_streaming_state()
        self.post_init()
    
    def get_speech_decoder(self):  
        return self.speech_generator

    def reset_streaming_state(self):
        self.generated_ids = []
        self.past_key_values = DynamicCache()
        self.cur_hidden_states = []
        self.cur_text = ""
        self.units_preds = []
        self.last_id_embeds = None
        if hasattr(self.config, "speech_generator_type") and self.config.speech_generator_type == "ar_linear_group_stream":
            self.pre_nn_past_key_values = DynamicCache()
            self.speech_gen_past_key_values = DynamicCache()

    def initialize_speech_generator(self, model_args):
        self.config.speech_generator_type = model_args.speech_generator_type
        if self.config.speech_generator_type == 'ctc':
            self.config.speech_generator_config = getattr(model_args, 'speech_generator_config')
            with open(self.config.speech_generator_config, 'r') as file:
                arconfig = yaml.safe_load(file)
            self.config.ctc_decoder_config = arconfig.get('ctc_decoder_config', '(4,4096,32,11008)')
            self.config.ctc_upsample_factor = arconfig.get('ctc_upsample_factor', 25)
            self.config.gen_loss_weight = arconfig.get('ctc_loss_weight', 1.0)
            self.config.unit_vocab_size = arconfig.get('unit_vocab_size', 4096)
            self.tune_speech_generator_only = getattr(model_args, 'tune_speech_generator_only', True)
            if getattr(self, "speech_generator", None) is None:
                self.speech_generator = build_speech_generator(self.config)
        elif 'ar' in self.config.speech_generator_type or 'transducer' in self.config.speech_generator_type:
            self.config.speech_generator_config = getattr(model_args, 'speech_generator_config')
            with open(self.config.speech_generator_config, 'r') as file:
                arconfig = yaml.safe_load(file)
            
            self.config.llm_hidden_size = arconfig.get('llm_hidden_size', 2048)
            self.config.decoder_hidden_size = arconfig.get('decoder_hidden_size', 2048)
            self.config.decoder_num_heads = arconfig.get('decoder_num_heads', 32)
            self.config.decoder_ffn_dim = arconfig.get('decoder_ffn_dim', 8192)
            self.config.decoder_dropout = arconfig.get('decoder_dropout', 0.1)
            self.config.decoder_num_layers = arconfig.get('decoder_num_layers', 4)
            self.config.encoder_num_layers = arconfig.get('encoder_num_layers', 2)  
            self.config.unit_vocab_size = arconfig.get('unit_vocab_size', 6561)
            self.config.max_speech_tokens = arconfig.get('max_speech_tokens', 4096)
            self.config.max_seq_length = arconfig.get('max_seq_length', 8192)  
            self.config.special_tokens = arconfig.get('special_tokens', 4)
            self.config.speech_bos_token_id = arconfig.get('speech_bos_token_id', self.config.unit_vocab_size + 0)
            self.config.speech_sos_token_id = arconfig.get('speech_sos_token_id', self.config.unit_vocab_size + 1)
            self.config.speech_eos_token_id = arconfig.get('speech_eos_token_id', self.config.unit_vocab_size + 2)
            self.config.speech_padding_token_id = arconfig.get('speech_padding_token_id', self.config.unit_vocab_size + 3)
            self.config.switch_token_id = arconfig.get('switch_token_id', self.config.unit_vocab_size + 4)
            self.config.speech_max_position_embeddings = arconfig.get('speech_max_position_embeddings', 2048)
            self.config.gen_loss_weight = arconfig.get('gen_loss_weight', 1.0)
            self.config.group_size = arconfig.get('group_size', 5)
            self.config.txt_token_num = arconfig.get('txt_token_num', 5)
            self.config.speech_token_num = arconfig.get('speech_token_num', 5)
            self.tune_speech_generator_only = getattr(model_args, 'tune_speech_generator_only', True)

            self.speech_generator = build_speech_generator(self.config)
        else:
            raise NotImplementedError(f"Unsupported speech generator type: {self.config.speech_generator_type}")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        speech: Optional[torch.FloatTensor] = None,
        speech_lengths: Optional[torch.LongTensor] = None,
        tgt_units: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_speech_and_text(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                speech,
                speech_lengths
            )
        

        if self.training:
            if self.tune_speech_generator_only:
                with torch.no_grad():
                    qwen_output = super(OmniSpeechQwen2ForCausalLM, self).forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds,
                        labels=labels,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=True,
                        return_dict=return_dict
                    )
                loss = self.speech_generator(qwen_output['hidden_states'][-1], labels, tgt_units)
            else:
                qwen_output = super(OmniSpeechQwen2ForCausalLM, self).forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=True,
                    return_dict=return_dict
                )
                lm_loss = qwen_output.loss
                ctc_loss = self.speech_generator(qwen_output['hidden_states'][-1], labels, tgt_units)
                print(lm_loss, ctc_loss)
                loss = lm_loss + ctc_loss * self.config.gen_loss_weight
        else:
            qwen_output = super(OmniSpeechQwen2ForCausalLM, self).forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict
            )
            loss = qwen_output.loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=qwen_output.logits,
            past_key_values=qwen_output.past_key_values,
            hidden_states=qwen_output.hidden_states,
            attentions=qwen_output.attentions
        )
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        speech: Optional[torch.Tensor] = None,
        speech_lengths: Optional[torch.Tensor] = None,
        streaming_unit_gen=False,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if speech is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_speech_and_text(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                speech,
                speech_lengths
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        outputs = GenerationWithCTC.generate(
            self,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict_in_generate=True,
            streaming_unit_gen=streaming_unit_gen,
            **kwargs
        )

        hidden_states = outputs['hidden_states']
        hidden_states = torch.cat([hidden_states[0][-1][:, -1:, :]] + [hidden_states[i][-1] for i in range(1, len(hidden_states))], dim=1)

        if self.config.speech_generator_type == 'ctc':
            units_pred = self.speech_generator.predict(hidden_states.squeeze(0))
        elif 'ar' in self.config.speech_generator_type:
            units_pred = self.speech_generator.predict(hidden_states)

        return outputs.sequences, units_pred
    @torch.no_grad()
    def time_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        speech: Optional[torch.Tensor] = None,
        speech_lengths: Optional[torch.Tensor] = None,
        streaming_unit_gen=False,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        
        if speech is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_speech_and_text(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                speech,
                speech_lengths
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        start_time = timeit.default_timer()
        outputs = GenerationWithCTC.generate(
            self,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict_in_generate=True,
            streaming_unit_gen=streaming_unit_gen,
            **kwargs
        )
        output_time_ms = (timeit.default_timer() - start_time) * 1000

        hidden_states = outputs['hidden_states']
        hidden_states = torch.cat([hidden_states[0][-1][:, -1:, :]] + [hidden_states[i][-1] for i in range(1, len(hidden_states))], dim=1)

        start_time = timeit.default_timer()
        if self.config.speech_generator_type == 'ctc':
            units_pred = self.speech_generator.predict(hidden_states.squeeze(0))
        elif 'ar' in self.config.speech_generator_type:
            units_pred = self.speech_generator.predict(hidden_states)
        speech_gen_time_ms = (timeit.default_timer() - start_time) * 1000

        return outputs.sequences, units_pred, output_time_ms, speech_gen_time_ms

    @torch.no_grad()
    def streaming_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        speech: Optional[torch.Tensor] = None,
        speech_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Generator[Tuple[str, Optional[List[torch.Tensor]]], None, None]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        
        first_step = self.past_key_values.get_seq_length() == 0
        if first_step:
            if speech is not None:
                (
                    inputs,
                    position_ids,
                    attention_mask,
                    _,
                    inputs_embeds,
                    _
                ) = self.prepare_inputs_labels_for_speech_and_text(
                    inputs,
                    position_ids,
                    attention_mask,
                    None,
                    None,
                    speech,
                    speech_lengths
                )
            else:
                inputs_embeds = self.get_model().embed_tokens(inputs)
            current_attention_mask = attention_mask
        else:
            current_attention_mask = torch.ones(1, self.past_key_values.get_seq_length() + 1, device=self.device)

        generated_ids_list = []
        while True:
            last_id, self.past_key_values, hidden_state = self._generate_one_step(
                inputs_embeds=inputs_embeds if first_step else self.last_id_embeds,
                attention_mask=current_attention_mask,
                past_key_values=self.past_key_values,
                **kwargs
            )

            if last_id[0][0] == PAD_TOKEN_ID_QWEN2_5:
                if self.cur_hidden_states:
                    accumulated_hidden = torch.cat(self.cur_hidden_states, dim=1)
                    if self.config.speech_generator_type == 'ctc':
                        units_pred = self.speech_generator.predict(accumulated_hidden.squeeze(0))
                    elif 'ar' in self.config.speech_generator_type:
                        units_pred = self.speech_generator.predict(accumulated_hidden)
                    self.units_preds.append(units_pred)
                    self.cur_hidden_states = []
                yield self.cur_text, self.units_preds.copy()
                break

            generated_ids_list.append(last_id)
            self.cur_hidden_states.append(hidden_state)

            concat_ids = torch.cat(generated_ids_list, dim=1)
            decoded_text = self.tokenizer.decode(concat_ids.squeeze(0), skip_special_tokens=True)
            self.cur_text = decoded_text

            if SENTENCE_DELIM_RE.search(self.cur_text):
                accumulated_hidden = torch.cat(self.cur_hidden_states, dim=1)
                if self.config.speech_generator_type == 'ctc':
                    units_pred = self.speech_generator.predict(accumulated_hidden.squeeze(0))
                elif 'ar' in self.config.speech_generator_type:
                    units_pred = self.speech_generator.predict(accumulated_hidden)
                self.units_preds.append(units_pred)
                yield self.cur_text, self.units_preds.copy()
                
                generated_ids_list.clear()
                self.cur_hidden_states = []
                self.cur_text = ""
                self.units_preds = []

            self.last_id_embeds = self.get_model().embed_tokens(last_id)
            current_attention_mask = torch.cat([
                current_attention_mask,
                torch.ones(1, 1, device=self.device)
            ], dim=1)
            first_step = False

    @torch.no_grad()
    def streaming_generate_inturn(
        self,
        inputs: Optional[torch.Tensor] = None,
        speech: Optional[torch.Tensor] = None,
        speech_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Generator[Tuple[str, Optional[List[torch.Tensor]]], None, None]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        
        first_step = self.past_key_values.get_seq_length() == 0
        if first_step:
            if speech is not None:
                (
                    inputs,
                    position_ids,
                    attention_mask,
                    _,
                    inputs_embeds,
                    _
                ) = self.prepare_inputs_labels_for_speech_and_text(
                    inputs,
                    position_ids,
                    attention_mask,
                    None,
                    None,
                    speech,
                    speech_lengths
                )
            else:
                inputs_embeds = self.get_model().embed_tokens(inputs)
            current_attention_mask = attention_mask
        else:
            current_attention_mask = torch.ones(1, self.past_key_values.get_seq_length() + 1, device=self.device)

        while True:
            self.cur_hidden_states = []
            for _ in range(self.config.txt_token_num):
                last_id, self.past_key_values, hidden_state = self._generate_one_step(
                    inputs_embeds=inputs_embeds if first_step else self.last_id_embeds,
                    attention_mask=current_attention_mask,
                    past_key_values=self.past_key_values,
                    **kwargs
                )
                self.generated_ids.append(last_id)
                self.cur_hidden_states.append(hidden_state)
                self.last_id_embeds = self.get_model().embed_tokens(last_id)
                current_attention_mask = torch.cat([current_attention_mask, torch.ones(1, 1, device=self.device)], dim=1)
                first_step = False

                if last_id[0][0] == self.tokenizer.eos_token_id:
                    break

            if self.generated_ids[-1] != self.tokenizer.eos_token_id:
                accumulated_hidden = torch.cat(self.cur_hidden_states, dim=1)
                if 'ar' in self.config.speech_generator_type:
                    units_pred = self.speech_generator._predict_one_step(
                        accumulated_hidden, 
                        self.pre_nn_past_key_values,
                        self.speech_gen_past_key_values
                    )
                yield self.tokenizer.decode(torch.cat(self.generated_ids, dim=1).squeeze()), units_pred, False
            else:
                final_units = self.speech_generator._predict_final_step(
                    torch.cat(self.cur_hidden_states, dim=1),
                    self.pre_nn_past_key_values,
                    self.speech_gen_past_key_values
                )
                yield self.tokenizer.decode(torch.cat(self.generated_ids, dim=1).squeeze()), final_units, True
                break

AutoConfig.register("omni_speech2s_qwen", OmniSpeech2SConfig)
AutoModelForCausalLM.register(OmniSpeech2SConfig, OmniSpeech2SQwen2ForCausalLM)