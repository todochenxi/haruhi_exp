#!/user/bin/env python3
# -*- coding: utf-8 -*-
# rwse、lappe、signnet、deepnet

import torch
from scipy import sparse as sp
import dgl
import numpy as np


def lap_positional_encoding_tkg(g, pe_dim, device):
    """
    Graph positional encoding v/ Laplacian eigenvectors
    g: 包含每个时间戳的子图列表
    pe_dim: 位置编码的维度
    """
    # print("---------------------> lappe")
    # for g in g:
    # 计算每个子图的拉普拉斯矩阵并生成位置编码
    # A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    A = g.adj_external(scipy_fmt="csr")
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])

    # 保存位置编码
    g.ndata['pe_init'] = torch.from_numpy(EigVec[:, 1:pe_dim + 1]).float().to(device)
    # g.ndata['pe_init'] = g.ndata['pe_init'].to('cuda')
    return g


def rw_positional_encoding_tkg(g, pe_dim, device):
    """
    Initializing positional encoding with RWPE for Temporal Knowledge Graphs (TKG)

    Args:
        g: 包含每个时间戳的子图列表
        pe_dim: 位置编码的维度
        type_init: 初始化方式，当前支持 'rand_walk'（随机游走）

    Returns:
        g: 包含位置编码的子图列表
    """
    # print('------------------------> rwpe')
    # for g in g:
    # n = g.number_of_nodes()

    # Geometric diffusion features with Random Walk
    A = g.adj_external(scipy_fmt="csr")
    Dinv = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float)  # D^-1
    RW = A * Dinv
    M = RW

    # Iterate to get positional encoding
    nb_pe_init = pe_dim
    PE = [torch.from_numpy(M.diagonal()).float()]
    M_power = M
    for _ in range(nb_pe_init - 1):
        M_power = M_power * M
        PE.append(torch.from_numpy(M_power.diagonal()).float())
    PE = torch.stack(PE, dim=-1)
    g.ndata['p'] = PE.to('cuda')

    return g

def lap_rw_positional_encoding_tkg(g, pe_dim, device):
    """
    Initializing positional encoding with Laplacian and RWPE for Temporal Knowledge Graphs (TKG)

    Args:
        g: 包含每个时间戳的子图列表
        pe_dim: 位置编码的维度
    Returns:
        g: 包含位置编码的子图列表
    """
    # print('---------------------> Initializing positional encoding with Laplacian and RWPE for Temporal Knowledge Graphs (TKG)')
    g1 = lap_positional_encoding_tkg(g, pe_dim, device)
    p1 = g1.ndata['pe_init']
    g2 = rw_positional_encoding_tkg(g, pe_dim, device)
    p2 = g2.ndata['pe_init']
    g.ndata['pe_init'] = torch.cat((p1, p2), dim=1)
    return g

