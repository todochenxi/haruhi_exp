#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time ： 2024/9/4 18:51
@Auth ： chenxi
@File ：gatedgcn_lspe2_layer.py
@IDE ：PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import function as fn

class GatedGCNLayer(nn.Module):
    """
        GatedGCN layer
        Residual Gated Graph ConvNets
        https://arxiv.org/pdf/1711.07553.pdf
    """
    def __init__(self, in_dim, out_dim, dropout=0, residual=False, act='relu',
                 equivstable_pe=False):
        super(GatedGCNLayer, self).__init__()
        self.activation = getattr(F, act)
        self.A = nn.Linear(in_dim, out_dim, bias=True)
        self.B = nn.Linear(in_dim, out_dim, bias=True)
        self.C = nn.Linear(in_dim, out_dim, bias=True)
        self.D = nn.Linear(in_dim, out_dim, bias=True)
        self.E = nn.Linear(in_dim, out_dim, bias=True)

        # Handling for Equivariant and Stable PE using LapPE
        self.EquivStablePE = equivstable_pe
        if self.EquivStablePE:
            self.mlp_r_ij = nn.Sequential(
                nn.Linear(1, out_dim),
                self.activation(),
                nn.Linear(out_dim, 1),
                nn.Sigmoid())

        self.bn_node_x = nn.BatchNorm1d(out_dim)
        self.bn_edge_e = nn.BatchNorm1d(out_dim)
        self.dropout = dropout
        self.residual = residual

    def forward(self, g):
        x = g.ndata['x']  # 节点特征
        e = g.edata['e']  # 边特征

        if self.residual:
            x_in = x
            e_in = e

        Ax = self.A(x)
        Bx = self.B(x)
        Ce = self.C(e)
        Dx = self.D(x)
        Ex = self.E(x)

        # Equivariant and Stable PE
        pe_LapPE = g.ndata['pe_EquivStableLapPE'] if self.EquivStablePE else None

        # 保存特征到图的边数据
        g.ndata['Bx'] = Bx
        g.ndata['Dx'] = Dx
        g.ndata['Ex'] = Ex
        g.edata['Ce'] = Ce
        g.edata['Ax'] = Ax

        # 进行消息传递
        g.update_all(
            self.message_func(pe_LapPE),
            self.reduce_func(),
            self.apply_func(Ax)
        )

        x = g.ndata['x']
        e = g.edata['e']

        # 批标准化和激活函数
        x = self.bn_node_x(x)
        e = self.bn_edge_e(e)

        x = self.activation(x)
        e = self.activation(e)

        # Dropout
        x = F.dropout(x, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        # 残差连接
        if self.residual:
            x = x_in + x
            e = e_in + e

        g.ndata['x'] = x
        g.edata['e'] = e

        return g

    def message_func(self, pe_LapPE):
        def msg_func(edges):
            Dx_i = edges.src['Dx']
            Ex_j = edges.dst['Ex']
            Ce = edges.data['Ce']

            e_ij = Dx_i + Ex_j + Ce
            sigma_ij = torch.sigmoid(e_ij)

            if self.EquivStablePE and pe_LapPE is not None:
                PE_i = edges.src['pe_EquivStableLapPE']
                PE_j = edges.dst['pe_EquivStableLapPE']
                r_ij = ((PE_i - PE_j) ** 2).sum(dim=-1, keepdim=True)
                r_ij = self.mlp_r_ij(r_ij)
                sigma_ij = sigma_ij * r_ij

            edges.data['sigma_ij'] = sigma_ij
            return {'sigma_ij': sigma_ij, 'Bx_j': edges.src['Bx']}

        return msg_func

    def reduce_func(self):
        def red_func(nodes):
            sum_sigma_x = nodes.mailbox['sigma_ij'] * nodes.mailbox['Bx_j']
            numerator_eta_xj = sum_sigma_x.sum(dim=1)

            sum_sigma = nodes.mailbox['sigma_ij'].sum(dim=1)
            denominator_eta_xj = sum_sigma + 1e-6

            out = numerator_eta_xj / denominator_eta_xj
            return {'x': out}

        return red_func

    def apply_func(self, Ax):
        def app_func(nodes):
            x = Ax + nodes.data['x']
            return {'x': x}

        def app_func_edges(edges):
            return {'e': edges.data['sigma_ij']}

        return app_func, app_func_edges
