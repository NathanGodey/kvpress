# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
from torch import nn

from kvpress.presses.base_press import BasePress


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
        layer_q_filter = self.q_filters[module.layer_idx]
        return -layer_q_filter*keys.sum(dim=-1)
