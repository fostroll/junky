# -*- coding: utf-8 -*-
# junky lib: layers.HighwayBiLSTM
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides a Highway LSTM implementagion for PyTorch models.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence                               


class HBiLSTM_layer(nn.Module):
    """Highway LSTM implementation, modified from
    from https://github.com/bamtercelboo/pytorch_Highway_Networks/blob/master/models/model_HBiLSTM.py.
    [Original Article](https://arxiv.org/abs/1709.06436).

    Args:
        in_features:        number of features in input.
        out_features:       number of features in output.
        lstm_hidden_dim:    hidden dim for LSTM layer.
        lstm_num_layers:    number of LSTM layers.
        lstm_dropout:       dropout between 2+ LSTM layers.
        init_weight:        whether to init bilstm weights as xavier_uniform_
        init_weight_value:  bilstm weight initialization `gain` will be defined as 
            `np.sqrt(init_weight_value)`
    """

    def __init__(self, in_features, out_features,
                 lstm_hidden_dim, lstm_num_layers,
                 lstm_dropout=0.0, init_weight=True, 
                 init_weight_value=2.0):
        super().__init__()

        self.hidden_dim = lstm_hidden_dim
        self.num_layers = lstm_num_layers
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = lstm_dropout
        self.bilstm = nn.LSTM(in_features, self.hidden_dim, num_layers=self.num_layers, bias=True, 
                              bidirectional=True, dropout=self.dropout)
        
        if init_weight:
            nn.init.xavier_uniform_(self.bilstm.all_weights[0][0], gain=np.sqrt(init_weight_value))
            nn.init.xavier_uniform_(self.bilstm.all_weights[0][1], gain=np.sqrt(init_weight_value))
            nn.init.xavier_uniform_(self.bilstm.all_weights[1][0], gain=np.sqrt(init_weight_value))
            nn.init.xavier_uniform_(self.bilstm.all_weights[1][1], gain=np.sqrt(init_weight_value))

        if self.bilstm.bias is True:
            a = np.sqrt(2/(1 + 600)) * np.sqrt(3)
            nn.init.uniform_(self.bilstm.all_weights[0][2], -a, a)
            nn.init.uniform_(self.bilstm.all_weights[0][3], -a, a)
            nn.init.uniform_(self.bilstm.all_weights[1][2], -a, a)
            nn.init.uniform_(self.bilstm.all_weights[1][3], -a, a)

        self.convert_layer = nn.Linear(in_features=self.hidden_dim * 2,
                                       out_features=self.in_features, bias=True)
        nn.init.constant_(self.convert_layer.bias, -1)

        self.fc1 = nn.Linear(in_features=self.in_features, out_features=self.hidden_dim*2, bias=True)
        nn.init.constant_(self.fc1.bias, -1)

        self.gate_layer = nn.Linear(in_features=self.in_features, out_features=self.hidden_dim*2, bias=True)
        nn.init.constant_(self.gate_layer.bias, -1)

    def forward(self, x, hidden, lens):
        # handle the source input x
        source_x = x
        
        x = pack_padded_sequence(x, lens, enforce_sorted=False)
        x, hidden = self.bilstm(x, hidden)
        x, _ = pad_packed_sequence(x)
        
        normal_fc = torch.transpose(x, 0, 1)

        x = x.permute(1, 2, 0)

        # normal layer in the formula is H
        source_x = torch.transpose(source_x, 0, 1)

        # the first way to convert 3D tensor to the Linear

        source_x = source_x.contiguous()
        information_source = source_x.view(source_x.size(0) * source_x.size(1), source_x.size(2))
        information_source = self.gate_layer(information_source)
        information_source = information_source.view(source_x.size(0), source_x.size(1), information_source.size(1))

        # transformation gate layer in the formula is T
        transformation_layer = torch.sigmoid(information_source)
        # carry gate layer in the formula is C
        carry_layer = 1 - transformation_layer
        # formula Y = H * T + x * C
        allow_transformation = torch.mul(normal_fc, transformation_layer)

        # the information_source compare to the source_x is for the same size of x,y,H,T
        allow_carry = torch.mul(information_source, carry_layer)
        # allow_carry = torch.mul(source_x, carry_layer)
        information_flow = torch.add(allow_transformation, allow_carry)

        information_flow = information_flow.contiguous()
        information_convert = information_flow.view(information_flow.size(0) * information_flow.size(1),
                                                    information_flow.size(2))
        information_convert = self.convert_layer(information_convert)
        information_convert = information_convert.view(information_flow.size(0), information_flow.size(1),
                                                       information_convert.size(1))

        information_convert = torch.transpose(information_convert, 0, 1)
        return information_convert, hidden


# HighWay recurrent model
class HighwayBiLSTM(nn.Module):
    """Highway LSTM model implementation, modified from
    from https://github.com/bamtercelboo/pytorch_Highway_Networks/blob/master/models/model_HBiLSTM.py.
    [Original Article](https://arxiv.org/abs/1709.06436).

    Args:
        hw_num_layers:      number of highway biLSTM layers.
        in_features:        number of features in input.
        out_features:       number of features in output.
        lstm_hidden_dim:    hidden dim for LSTM layer.
        lstm_num_layers:    number of LSTM layers.
        lstm_dropout:       dropout between 2+ LSTM layers.
        init_weight:        whether to init bilstm weights as xavier_uniform_
        init_weight_value:  bilstm weight initialization `gain` will be defined as 
            `np.sqrt(init_weight_value)`
        batch_first:        True if input.size(0) == batch_size.
        
    """

    def __init__(self, hw_num_layers, lstm_hidden_dim, lstm_num_layers, 
                 in_features, out_features, lstm_dropout,
                 init_weight=True, init_weight_value=2.0, batch_first=True):
        super().__init__()

        self.hw_num_layers = hw_num_layers
        self.hidden_dim = lstm_hidden_dim
        self.num_layers = lstm_num_layers
        self.batch_first = batch_first

        # multiple HighWay layers List
        self.highway = nn.ModuleList(
                        [HBiLSTM_layer(in_features, out_features, 
                                 lstm_hidden_dim, lstm_num_layers, 
                                 lstm_dropout=lstm_dropout, 
                                 init_weight=True, init_weight_value=2.0) 
                         for _ in range(hw_num_layers)]
        )
        self.output_layer = nn.Linear(in_features=in_features, out_features=out_features, bias=True)

        if self.output_layer.bias is True:
            a = np.sqrt(2/(1 + in_features)) * np.sqrt(3)
            nn.init.uniform_(self.output_layer, -a, a)

    def init_hidden(self, num_layers, batch_size, device):
        # the first is the hidden h
        # the second is the cell c
        return (Variable(torch.zeros(2 * num_layers, batch_size, self.hidden_dim)).to(device),
                Variable(torch.zeros(2 * num_layers, batch_size, self.hidden_dim)).to(device))

    def forward(self, x, lens):
        device = next(self.parameters()).device

        self.hidden = self.init_hidden(self.num_layers, x.size(0), device)

        if self.batch_first:
            x = x.transpose(0, 1)     # to (seq_len, batch_size, emb_dim)

        for current_layer in self.highway:
            x, self.hidden = current_layer(x, self.hidden, lens)

        x = torch.transpose(x, 0, 1)
        x = torch.tanh(x)
        output_layer = self.output_layer(x)

        return output_layer
