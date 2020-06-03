# -*- coding: utf-8 -*-
# junky lib: layers.HighwayNetwork
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides a Highway Netword implementation for PyTorch models.
"""
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F


class HighwayNetwork(nn.Module):
    """ 
    Highway Network is described in
    https://arxiv.org/abs/1505.00387 and https://arxiv.org/abs/1507.06228 and
    it's formalation is: H(x)*T(x) + x*(1 - T(x)), where:
    .. H(x) - affine trainsformation followed by a non-linear activation;
    .. T(x) - transformation gate: affine transformation followed by a sigmoid
           activation;
    .. * - element-wise multiplication.

    There are some variations of it, so we implement more universal 
    architectute: U(x)*H(x)*T(x) + x*C(x), where:
    .. U(x) - user defined layer that we make Highway around; By default,
           U(x) = I (identity matrix);
    .. C(x) - carry gate: generally, affine transformation followed by a sigmoid
           activation. By default, C(x) = 1 - T(x).

    Args:
        in_features: number of features in input.
        out_features: number of features in output. If ``None`` (default),
            **out_features** = **in_features**.
        U_layer: layer that implements U(x). Default is ``None``. If U_layer
            is callable, it will be used to create the layer; elsewise, we'll
            use it as is (if **num_layers** > 1, we'll copy it). Note that
            number of input features of U_layer must be equal to
            **out_features** if **num_layers** > 1.
        U_init_: callable to inplace init weights of **U_layer**.
        U_dropout: if non-zero, introduces a Dropout layer on the outputs of
            U(x) on each layer, with dropout probability equal to
            **U_dropout**. Default: 0.
        H_features: number of input features of H(x). If ``None`` (default),
            H_features = in_features. If ``0``, don't use H(x).
        H_activation: non-linear activation after H(x). If ``None``, then no
            activation function is used. Default is ``F.relu``.
        H_dropout: if non-zero, introduces a Dropout layer on the outputs of
            H(x) on each layer, with dropout probability equal to
            **U_dropout**. Default: 0.
        gate_type: a type of the transform and carry gates:
            'generic' (default): C(x) = 1 - T(x);
            'independent': use both independent C(x) and T(x);
            'T_only': don't use carry gate: C(x) = I;
            'C_only': don't use carry gate: T(x) = I;
            'none': C(x) = T(x) = I.
        global_highway_input: if ``True``, we treat the input of all the
            network as the highway input of every layer. Thus, we use T(x)
            and C(x) only once. If **global_highway_input** is ``False``
            (default), every layer receives the output of the previous layer
            as the highway input. So, T(x) and C(x) use different weights
            matrices in each layer.
        num_layers: number of highway layers.
    """
    __constants__ = ['H_activation', 'H_dropout', 'H_features', 'U_dropout',
                     'U_init', 'U_layer', 'gate_type', 'global_highway_input',
                     'last_dropout', 'out_dim', 'num_layers']

    def __init__(self, in_features, out_features=None,
                 U_layer=None, U_init_=None, U_dropout=0,
                 H_features=None, H_activation=F.relu, H_dropout=0,
                 gate_type='generic', global_highway_input=False,
                 num_layers=1):
        super().__init__()

        if out_features is None:
            out_features = in_features
        if H_features is None:
            H_features = in_features

        self.in_features = in_features
        self.out_features = out_features
        self.U_layer = U_layer
        self.U_init_ = U_init_
        self.U_dropout = U_dropout
        self.H_features = H_features
        self.H_activation = H_activation
        self.H_dropout = H_dropout
        self.gate_type = gate_type
        self.global_highway_input = global_highway_input
        self.num_layers = num_layers

        if U_layer:
            self._U = U_layer() if callable(U_layer) else U_layer
            if U_init_:
                U_init_(U_layer)
        else:
            self._U = None
        self._H = nn.Linear(H_features, out_features) if H_features else None
        if self.gate_type not in ['C_only', 'none']:
            self._T = nn.Linear(in_features, out_features)
            nn.init.constant_(self._T.bias, -1)
        if self.gate_type not in ['generic', 'T_only', 'none']:
            self._C = nn.Linear(in_features, out_features)
            nn.init.constant_(self._C.bias, 1)

        self._U_do = \
            nn.Dropout(p=U_dropout) if self._U and U_dropout else None
        self._H_do = \
            nn.Dropout(p=H_dropout) if self._H and H_dropout else None

        self._H_activation = H_activation
        self._T_activation = torch.sigmoid
        self._C_activation = torch.sigmoid

        if self.num_layers > 1:
            self._Us = nn.ModuleList() if U_layer else None
            self._Hs = nn.ModuleList() if H_features else None
            if not self.global_highway_input:
                if self.gate_type not in ['C_only', 'none']:
                    self._Ts = nn.ModuleList()
                if self.gate_type not in ['generic', 'T_only', 'none']:
                    self._Cs = nn.ModuleList()

            for i in range(self.num_layers - 1):
                if self._Us is not None:
                    U = U_layer() if callable(U_layer) else deepcopy(U_layer)
                    if U_init_:
                       U_init_(U)
                    self._Us.append(U)
                if self._Hs is not None:
                    self._Hs.append(nn.Linear(out_features, out_features))
                if not self.global_highway_input:
                    if self.gate_type not in ['C_only', 'none']:
                        T = nn.Linear(in_features, out_features)
                        nn.init.constant_(T.bias, -1)
                        self._Ts.append(T)
                    if self.gate_type not in ['generic', 'T_only', 'none']:
                        C = nn.Linear(in_features, out_features)
                        nn.init.constant_(C.bias, -1)
                        self._Cs.append(C)

    def forward(self, x, x_hw, *U_args, **U_kwargs):
        """
        :param x: tensor with shape [batch_size, seq_len, emb_size]
        :param x_hw: tensor with shape [batch_size, in_features, emb_size]
            if ``None``, x is used
        :return: tensor with shape [batch_size, seq_len, emb_size]
        """
        if x_hw is None:
            x_hw = x

        if self._U:
            x = self._U(x, *U_args, **U_kwargs)
            if self._U_do:
                x = self._U_do(x)
        if self._H:
            x = self._H(x)
        if self._H_activation:
            x = self._H_activation(x)
            if self._H_do:
                x = self._H_do(x)

        if self.gate_type == 'generic':
            x_t = self._T_activation(self._T(x_hw))
            x = x_t * x
            x_hw = (1 - x_t) * x_hw
        elif self.gate_type not in ['C_only', 'none']:
            x = self._T_activation(self._T(x_hw)) * x
        elif self.gate_type not in ['T_only', 'none']:
            x_hw = self._C_activation(self._C(x_hw)) * x_hw

        x += x_hw

        if self.num_layers > 1:
            for i in range(self.num_layers - 1):
                if self._Us:
                    x = self._Us[i](x, *U_args, **U_kwargs)
                    if self._U_do:
                        x = self._U_do(x)
                if self._Hs:
                    x = self._Hs[i](x)
                if self._H_activation:
                    x = self._H_activation(x)
                    if self._H_do:
                        x = self._H_do(x)

                if not self.global_highway_input:
                    if self.gate_type == 'generic':
                        x_t = self._T_activation(self._Ts[i](x_hw))
                        x = x_t * x
                        x_hw = (1 - x_t) * x_hw
                    elif self.gate_type not in ['C_only', 'none']:
                        x = self._T_activation(self._Ts[i](x_hw)) * x
                    elif self.gate_type not in ['T_only', 'none']:
                        x_hw = self._C_activation(self._Cs[i](x_hw)) * x_hw

                x += x_hw
                if not self.global_highway_input:
                    x_hw = x

        return x

    def extra_repr(self):
        return (
            '{}, {}, U_layer={}, U_init_={}, U_dropout={}, '
            'H_features={}, H_activation={}, H_dropout={}, '
            "gate_type='{}', global_highway_input={}, num_layers={}"
        ).format(self.in_features, self.out_features,
                 None if not self.U_layer else
                 '<callable>' if callable(self.U_layer) else
                 '<layer>' if isinstance(self.U_layer, nn.Module) else
                 '<ERROR>',
                 None if not self.U_init_ else
                 '<callable>' if callable(self.U_init_) else
                 '<ERROR>', self.U_dropout,
                 self.H_features,
                 None if not self.H_activation else
                 '<callable>' if callable(self.H_activation) else
                 '<ERROR>', self.H_dropout,
                 self.gate_type, self.global_highway_input, self.num_layers)
