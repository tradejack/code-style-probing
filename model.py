import torch
from torch.nn import CrossEntropyLoss
from transformers import PLBartForConditionalGeneration
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int):
    """
    Shift input ids one token to the right, and wrap the last non pad token (the <LID> token) Note that MBart does not
    have a single `decoder_start_token_id` in contrast to other Bart-like models.
    """
    prev_output_tokens = input_ids.clone()

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)

    index_of_eos = (
        prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1
    ).unsqueeze(-1)
    decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
    prev_output_tokens[:, 0] = decoder_start_tokens

    return prev_output_tokens


class Modifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_1, hidden_2):
        super(Modifier, self).__init__()
        self.linear_1 = torch.nn.Linear(input_dim, hidden_1)
        self.linear_2 = torch.nn.Linear(hidden_1, hidden_2)
        self.linear_3 = torch.nn.Linear(hidden_2, output_dim)

    def forward(self, x):
        out = self.linear_1(x)
        out = torch.nn.functional.relu(out)

        out = self.linear_2(out)
        out = torch.nn.functional.relu(out)

        out = self.linear_3(out)

        return out


class InRepPlusGAN(torch.nn.Module):
    def __init__(self, style_dim):
        super(InRepPlusGAN, self).__init__()
        self.model = PLBartForConditionalGeneration.from_pretrained(
            "uclanlp/plbart-multi_task-python",
        )
        self.encoder = self.model.get_encoder()
        self.decoder = self.model.get_decoder()
        self.config = self.model.config
        self.modifier = Modifier(
            input_dim=self.config.d_model + style_dim,
            output_dim=self.config.d_model,
            hidden_1=768,
            hidden_2=768,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        style_encoding: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.LongTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds=None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id
                )

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = (
            use_cache if use_cache is not None else self.config.use_cache
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        # different to other models, PLBart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id
            )

        # encoder E, with no grad
        if encoder_outputs is None:
            with torch.no_grad():
                encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1]
                if len(encoder_outputs) > 1
                else None,
                attentions=encoder_outputs[2]
                if len(encoder_outputs) > 2
                else None,
            )

        # need an additional tunable encoder M

        batch_size = encoder_outputs[0].shape[0]
        seq_len = encoder_outputs[0].shape[1]

        style_encoding = style_encoding.unsqueeze(1).expand(-1, seq_len, -1)
        #         for _ in range(1, seq_len):
        #             style_encoding = torch.cat((style_encoding, style_encoding.unsqueeze(1)), dim=1)

        #         print(encoder_outputs[0].shape, style_encoding.shape)
        combined_encoding = torch.cat(
            (encoder_outputs[0], style_encoding), dim=-1
        )
        modifier_outputs = []
        for i in range(seq_len):
            modifier_output = self.modifier(combined_encoding[:, i, :])
            modifier_outputs += [modifier_output.unsqueeze(1)]
        modifier_outputs = torch.cat(modifier_outputs, dim=1)

        # decoder G, with no grad
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        with torch.no_grad():
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=modifier_outputs,
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        outputs = None
        if not return_dict:
            outputs = decoder_outputs + encoder_outputs
        else:
            outputs = Seq2SeqModelOutput(
                last_hidden_state=decoder_outputs.last_hidden_state,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )

        with torch.no_grad():
            lm_logits = (
                self.model.lm_head(outputs[0]) + self.model.final_logits_bias
            )

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                lm_logits.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return (
                ((masked_lm_loss,) + output)
                if masked_lm_loss is not None
                else output
            )

        return (
            Seq2SeqLMOutput(
                loss=masked_lm_loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            ),
            modifier_outputs,
        )

    def get_encoding(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        style_encoding: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.LongTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds=None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        with torch.no_grad():
            # encoder E, with no grad
            if encoder_outputs is None:
                encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
            elif return_dict and not isinstance(
                encoder_outputs, BaseModelOutput
            ):
                encoder_outputs = BaseModelOutput(
                    last_hidden_state=encoder_outputs[0],
                    hidden_states=encoder_outputs[1]
                    if len(encoder_outputs) > 1
                    else None,
                    attentions=encoder_outputs[2]
                    if len(encoder_outputs) > 2
                    else None,
                )
            return encoder_outputs

    # def forward(self, **inputs):
    #     outputs = self.model(**inputs)
    #     return outputs


class Discriminator(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        embedding_layer,
        output_size,
        style_dim,
        device="cpu",
    ):
        super(Discriminator, self).__init__()

        self.output_size = output_size
        self.style_dim = style_dim
        self.device = device

        self.embedding = embedding_layer
        self.rnn = torch.nn.RNN(
            embedding_dim, output_size, 1, batch_first=True
        )
        self.linear = torch.nn.Linear(output_size, style_dim)

        self.softmax = torch.nn.Softmax(dim=1)

        # self.l2 = torch.nn.Linear(self.config.d_model + style_dim, self.config.d_model)
        # self.l3 = torch.nn.Linear(self.config.d_model + style_dim, self.config.d_model)

    def forward(self, x):
        batch_size = x.shape[0]

        with torch.no_grad():
            embedded_x = self.embedding(x)

        # RNN Layer
        init_hidden = torch.zeros(1, batch_size, self.output_size).to(
            device=self.device
        )
        output, hidden = self.rnn(embedded_x, init_hidden)

        # Linear Layer
        hidden = hidden.squeeze(0)
        output = self.linear(hidden)
        logits = self.softmax(output)
        return logits
