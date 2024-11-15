import argparse
import copy
import math
from typing import List, Optional, Union

from PIL import Image
import torch
from transformers import GenerationConfig, TextStreamer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList, LogitsWarper
import numpy as np

def softmax(x):
    """计算输入数组的软最大值（Softmax）"""
    # exp_x = np.exp(x - np.max(x))  # 减去最大值以防止溢出
    exp_x = np.exp(x) # 减去最大值以防止溢出
    return exp_x / exp_x.sum()

def calculate_entropy(prob_distribution, epsilon=1e-10):
    # 使用 Softmax 将输入数组转换为概率分布
    prob_distribution = softmax(prob_distribution)
    
    # 用极小值替代零值
    # prob_distribution = np.where(prob_distribution == 0, epsilon, prob_distribution)
    
    # 计算熵
    ent = entropy(prob_distribution, base=2)  # 使用 base=2 得到以比特为单位的熵

    return ent


class EnsembleLogitsProcessor(LogitsProcessor):
    r"""
    Logits processor for Classifier-Free Guidance (CFG). The processors computes a weighted average across scores
    from prompt conditional and prompt unconditional (or negative) logits, parameterized by the `guidance_scale`.
    The unconditional scores are computed internally by prompting `model` with the `unconditional_ids` branch.

    See [the paper](https://arxiv.org/abs/2306.17806) for more information.

    Args:
        guidance_scale (`float`):
            The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale != 1`.
            Higher guidance scale encourages the model to generate samples that are more closely linked to the input
            prompt, usually at the expense of poorer quality. A value smaller than 1 has the opposite effect, while
            making the negative prompt provided with negative_prompt_ids (if any) act as a positive prompt.
        model (`PreTrainedModel`):
            The model computing the unconditional scores. Supposedly the same as the one computing the conditional
            scores. Both models must use the same tokenizer.
        unconditional_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of input sequence tokens in the vocabulary for the unconditional branch. If unset, will default to
            the last token of the prompt.
        unconditional_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Attention mask for unconditional_ids.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to cache key/values during the negative prompt forward pass.


    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    >>> inputs = tokenizer(["Today, a dragon flew over Paris, France,"], return_tensors="pt")
    >>> out = model.generate(inputs["input_ids"], guidance_scale=1.5)
    >>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    'Today, a dragon flew over Paris, France, killing at least 50 people and injuring more than 100'

    >>> # with a negative prompt
    >>> neg_inputs = tokenizer(["A very happy event happened,"], return_tensors="pt")
    >>> out = model.generate(inputs["input_ids"], guidance_scale=2, negative_prompt_ids=neg_inputs["input_ids"])
    >>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    'Today, a dragon flew over Paris, France, killing at least 130 people. French media reported that'

    >>> # with a positive prompt
    >>> neg_inputs = tokenizer(["A very happy event happened,"], return_tensors="pt")
    >>> out = model.generate(inputs["input_ids"], guidance_scale=0, negative_prompt_ids=neg_inputs["input_ids"])
    >>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    "Today, a dragon flew over Paris, France, and I'm very happy to be here. I"
    ```
    """

    def __init__(
        self,
        guidance_scale: float,
        model,
        image_start_token_id,
        image_end_token_id,
        image_next_line_token_id,
        patch_size,
        threshold=None,
        unconditional_ids: Optional[torch.LongTensor] = None,
        unconditional_attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
    ):
        self.guidance_scale = guidance_scale
        self.model = model
        self.unconditional_context_backup = {
            "input_ids": unconditional_ids,
            "attention_mask": unconditional_attention_mask,
            "use_cache": use_cache,
            "past_key_values": None,
            "first_pass": True,
        }
        self.unconditional_context = None

        self.other_context_backup = {
            "input_ids": None,
            "attention_mask": None,
            "use_cache": use_cache,
            "past_key_values": None,
            "first_pass": True,
        }
        self.other_context = None

        self.nums_image_start_tokens = None

        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.image_next_line_token_id = image_next_line_token_id
        self.image_start_token_id_index = None
        self.patch_size = patch_size
        self.h_latent_dim = None
        self.w_latent_dim = None
        self.threshold = threshold

    def get_unconditional_logits(self, input_ids, image_start_token_id_index):

        if self.unconditional_context["first_pass"]:
            if self.unconditional_context["input_ids"] is None:
                self.unconditional_context["input_ids"] = input_ids[:, image_start_token_id_index:]
            if self.unconditional_context["attention_mask"] is None:
                self.unconditional_context["attention_mask"] = torch.ones_like(
                    self.unconditional_context["input_ids"], dtype=torch.long
                )
            input_ids = self.unconditional_context["input_ids"]
            attention_mask = self.unconditional_context["attention_mask"]
            self.unconditional_context["first_pass"] = False
        else:
            attention_mask = torch.cat(
                [
                    self.unconditional_context["attention_mask"],
                    torch.ones_like(input_ids[:, -1:], dtype=torch.long),
                ],
                dim=1,
            )
            if not self.unconditional_context["use_cache"]:
                input_ids = torch.cat([self.unconditional_context["input_ids"], input_ids[:, -1:]], dim=1)
            else:
                input_ids = input_ids[:, -1:]
            self.unconditional_context["input_ids"] = input_ids
            self.unconditional_context["attention_mask"] = attention_mask

        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=self.unconditional_context["use_cache"],
            past_key_values=self.unconditional_context["past_key_values"],
        )
        self.unconditional_context["past_key_values"] = out.get("past_key_values", None)

        return out.logits

    def get_other_logits(self, input_ids, image_start_token_id_index, nums=1):

        ori_input_ids = input_ids

        if self.other_context["first_pass"]:
            if self.other_context["input_ids"] is None:
                self.other_context["input_ids"] = input_ids
            if self.other_context["attention_mask"] is None:
                self.other_context["attention_mask"] = torch.ones_like(
                    self.other_context["input_ids"], dtype=torch.long
                )
            input_ids = self.other_context["input_ids"]
            attention_mask = self.other_context["attention_mask"]
            self.other_context["first_pass"] = False
        else:
            attention_mask = torch.cat(
                [
                    self.other_context["attention_mask"],
                    torch.ones_like(input_ids[:, -1:], dtype=torch.long),
                ],
                dim=1,
            )
            if not self.other_context["use_cache"]:
                input_ids = torch.cat([self.other_context["input_ids"], input_ids[:, -1:]], dim=1)
            else:
                input_ids = input_ids[:, -1:]
            self.other_context["input_ids"] = input_ids
            self.other_context["attention_mask"] = attention_mask

        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=self.other_context["use_cache"],
            past_key_values=self.other_context["past_key_values"],
        )
        self.other_context["past_key_values"] = out.get("past_key_values", None)

        return out.logits

    def __call__(self, input_ids, scores):
        # print(scores.dtype)
        num_image_start_tokens = (input_ids[0] == self.image_start_token_id).sum()
        num_image_end_tokens = (input_ids[0] == self.image_end_token_id).sum()

        if num_image_start_tokens == num_image_end_tokens:
            self.h_latent_dim, self.w_latent_dim = None, None
            self.image_start_token_id_index = None
            self.unconditional_context = None
            self.other_context = None
            return scores

        elif num_image_start_tokens == num_image_end_tokens + 1:
            if self.image_start_token_id_index is None:
                self.image_start_token_id_index = torch.where(input_ids[0] == self.image_start_token_id)[0][-1].item()
            new_token_num = len(input_ids[0][self.image_start_token_id_index + 1 :])
            if new_token_num >= 2:
                if self.h_latent_dim is None or self.w_latent_dim is None:
                    h_grids, w_grids = (
                        input_ids[0][self.image_start_token_id_index + 1] - 8804,
                        input_ids[0][self.image_start_token_id_index + 2] - 8804,
                    )
                    self.h_latent_dim, self.w_latent_dim = h_grids * 2, w_grids * 2

                if self.unconditional_context is None:
                    self.unconditional_context = copy.deepcopy(self.unconditional_context_backup)
                if self.other_context is None:
                    self.other_context = copy.deepcopy(self.other_context_backup)

                if self.guidance_scale == 1.0:
                    return scores

                # e = calculate_entropy(scores)
                # if e < self.threshold:
                #     return scores

                unconditional_logits = self.get_unconditional_logits(input_ids, self.image_start_token_id_index)[:, -1]
                other_logits = self.get_other_logits(input_ids, self.image_start_token_id_index)[:, -1]

                emsemble_scores = (scores + other_logits) / 2
                scores_processed = self.guidance_scale * (emsemble_scores - unconditional_logits) + unconditional_logits

                return scores_processed

        else:
            print("Something wrong in the decoding process.")

        return scores