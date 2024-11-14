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


class DynamicTemperatureLogitsWarper(LogitsProcessor):
    r"""
    [`LogitsProcessor`] for temperature (exponential scaling output probability distribution), which effectively means
    that it can control the randomness of the predicted tokens. Often used together with [`TopPLogitsWarper`] and
    [`TopKLogitsWarper`].

    <Tip>

    Make sure that `do_sample=True` is included in the `generate` arguments otherwise the temperature value won't have
    any effect.

    </Tip>

    Args:
        temperature (`float`):
            Strictly positive float value used to modulate the logits distribution. A value smaller than `1` decreases
            randomness (and vice versa), with `0` being equivalent to shifting all probability mass to the most likely
            token.

    Examples:

    ```python
    >>> import torch
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> set_seed(0)  # for reproducibility

    >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    >>> model.config.pad_token_id = model.config.eos_token_id
    >>> inputs = tokenizer(["Hugging Face Company is"], return_tensors="pt")

    >>> # With temperature=1.0, the default, we consistently get random outputs due to random sampling.
    >>> generate_kwargs = {"max_new_tokens": 10, "do_sample": True, "temperature": 1.0, "num_return_sequences": 2}
    >>> outputs = model.generate(**inputs, **generate_kwargs)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    ['Hugging Face Company is one of these companies that is going to take a',
    "Hugging Face Company is a brand created by Brian A. O'Neil"]

    >>> # However, with temperature close to 0, it approximates greedy decoding strategies (invariant)
    >>> generate_kwargs["temperature"] = 0.0001
    >>> outputs = model.generate(**inputs, **generate_kwargs)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    ['Hugging Face Company is a company that has been around for over 20 years',
    'Hugging Face Company is a company that has been around for over 20 years']
    ```
    """

    def __init__(
        self,
        temperature: float, 
        hot_temperature: float, 
        num_hot_tokens: int,
        image_start_token_id,
        image_end_token_id,
        image_next_line_token_id,
        patch_size,
        use_cache: Optional[bool] = True):
        if not isinstance(temperature, float) or not (temperature > 0):
            except_msg = (
                f"`temperature` (={temperature}) has to be a strictly positive float, otherwise your next token "
                "scores will be invalid."
            )
            if isinstance(temperature, float) and temperature == 0.0:
                except_msg += " If you're looking for greedy decoding strategies, set `do_sample=False`."
            raise ValueError(except_msg)

        self.temperature = temperature
        self.hot_temperature = hot_temperature
        self.num_hot_tokens = num_hot_tokens

        self.nums_image_start_tokens = None
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.image_next_line_token_id = image_next_line_token_id
        self.image_start_token_id_index = None
        self.patch_size = patch_size
        self.h_latent_dim = None
        self.w_latent_dim = None

    # @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    # def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    #     scores_processed = scores / self.temperature
    #     return scores_processed

    def __call__(self, input_ids, scores):

        if self.num_hot_tokens == 0:
            scores_processed = scores / self.temperature
            return scores_processed

        num_image_start_tokens = (input_ids[0] == self.image_start_token_id).sum()
        num_image_end_tokens = (input_ids[0] == self.image_end_token_id).sum()

        if num_image_start_tokens == num_image_end_tokens:
            self.h_latent_dim, self.w_latent_dim = None, None
            self.image_start_token_id_index = None
            scores_processed = scores / self.temperature
            return scores_processed

        elif num_image_start_tokens == num_image_end_tokens + 1:
            if self.image_start_token_id_index is None:
                self.image_start_token_id_index = torch.where(input_ids[0] == self.image_start_token_id)[0][-1].item()
            new_token_num = len(input_ids[0][self.image_start_token_id_index + 1 :])
            if new_token_num >= 2 and (new_token_num - 2) < self.num_hot_tokens:
                if self.h_latent_dim is None or self.w_latent_dim is None:
                    h_grids, w_grids = (
                        input_ids[0][self.image_start_token_id_index + 1] - 8804,
                        input_ids[0][self.image_start_token_id_index + 2] - 8804,
                    )
                    self.h_latent_dim, self.w_latent_dim = h_grids * 2, w_grids * 2

                scores_processed = scores / self.hot_temperature
                return scores_processed

        else:
            print("Something wrong in the decoding process.")

        return scores / self.temperature