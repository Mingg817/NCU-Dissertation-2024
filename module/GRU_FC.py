#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   gru_fc.py
@Time    :   2024/05/20
@Author  :   LI YIMING
@Version :   1.0
@Site    :   https://github.com/Mingg817
@Desc    :   GRU-FC模型
"""

import torch
from torch import nn


class GRU_FC(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(GRU_FC, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

        # Xavier initialization
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)

    def init_hidden(self, batch_size):
        return (torch.rand(self.num_layers, batch_size, self.hidden_size) * 0.01).to(
            "cuda"
        )

    def forcast(self, x, h0, forcast_length):
        forcast_out = torch.zeros(x.shape[0], forcast_length).to("cuda")
        for i in range(forcast_length):
            # teaching force
            # x -> (B,L)
            x = x.unsqueeze(2)
            # x -> (B,L,1)
            out, h1 = self.gru(x, h0)
            # out -> (B,L,H)
            out = self.fc(out[:, -1, :])
            # out -> (B,1)
            forcast_out[:, i] = out.squeeze()
            x = out
            h0 = h1
        return torch.mean(forcast_out, 1), h1

    def forward(self, x, h0=None, forcast_length=1, **args):
        if h0 is None:
            h0 = self.init_hidden(x.shape[0])
        return self.forcast(x, h0, forcast_length)
