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

class MinPLogitsWarper(LogitsWarper):
    """
    [`LogitsWarper`] that performs min-p, i.e. keeps all tokens that are above a minimum probability, scaled by the
    probability of the most likely token. As a result, the filter becomes more agressive in the presence of
    high-probability tokens, which is a sign of a confident output that we shouldn't deviate from.

    Often used together with [`TemperatureLogitsWarper`]. Used as an alternative to [`TopPLogitsWarper`] and
    [`TopKLogitsWarper`].

    Created by @menhguin and @kalomaze (github handles). Code adapted from [this external PR](https://github.com/oobabooga/text-generation-webui/pull/4449/files)

    Args:
        min_p (`float`):
            Minimum token probability, which will be scaled by the probability of the most likely token. It must be a
            value between 0 and 1. Typical values are in the 0.01-0.2 range, comparably selective as setting `top_p` in
            the 0.99-0.8 range (use the opposite of normal `top_p` values).
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> set_seed(1)
    >>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

    >>> inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")

    >>> # With sampling, the output is unexpected -- sometimes too unexpected.
    >>> outputs = model.generate(**inputs, do_sample=True)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 3 | < 4 (left-hand pointer) ;
    <BLANKLINE>
    <BLANKLINE>

    >>> # With `min_p` sampling, the output gets restricted to high-probability tokens.
    >>> # Pro tip: In practice, LLMs use `min_p` in the 0.01-0.2 range.
    >>> outputs = model.generate(**inputs, do_sample=True, min_p=0.1)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
    ```
    """

    def __init__(self, min_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if not (0 <= min_p <= 1.0):
            raise ValueError(f"`min_p` has to be a float in the [0, 1] interval, but is {min_p}")
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

        self.min_p = min_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Convert logits to probabilities
        probs = torch.softmax(scores, dim=-1)
        # Get the probability of the top token for each sequence in the batch
        top_probs, _ = probs.max(dim=-1, keepdim=True)
        # Calculate the actual min_p threshold by scaling min_p with the top token's probability
        scaled_min_p = self.min_p * top_probs
        # Create a mask for tokens that have a probability less than the scaled min_p
        tokens_to_remove = probs < scaled_min_p

        sorted_indices = torch.argsort(scores, descending=True, dim=-1)
        sorted_indices_to_remove = torch.gather(tokens_to_remove, dim=-1, index=sorted_indices)
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., : self.min_tokens_to_keep] = False

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed