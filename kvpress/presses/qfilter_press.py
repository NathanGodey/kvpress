# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
from torch import nn

from huggingface_hub import PyTorchModelHubMixin

from kvpress.presses.base_press import BasePress

class QFilters(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, num_layers, num_kv_heads, kv_head_dim):
        super().__init__()
        self.q_filters = torch.nn.Parameter(torch.randn(num_layers, num_kv_heads, kv_head_dim))


class QFilterPress(BasePress):
    """Prune KV pairs with Q-filters"""
    q_filters = None

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        layer_q_filter = self.q_filters[module.layer_idx].to(keys.device)
        scores = -(layer_q_filter[None,:,None]*keys).sum(dim=-1)
        return scores
