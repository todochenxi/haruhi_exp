import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

import dgl

"""
    GatedGCNLSPE: GatedGCN with LSPE
"""

class GatedGCNLSPELayer(nn.Module):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, use_lapeig_loss=False, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        self.use_lapeig_loss = use_lapeig_loss
        
        if input_dim != output_dim:
            self.residual = False
        
        self.A1 = nn.Linear(input_dim*2, output_dim, bias=True)  # 处理拼接后的节点特征和位置编码
        self.A2 = nn.Linear(input_dim*2, output_dim, bias=True)  # 处理拼接后的节点特征和位置编码
        self.B1 = nn.Linear(input_dim, output_dim, bias=True)  # 分别处理不同的输入特征，用于计算边上的权重
        self.B2 = nn.Linear(input_dim, output_dim, bias=True)  # 分别处理不同的输入特征，用于计算边上的权重
        self.B3 = nn.Linear(input_dim, output_dim, bias=True)  # 分别处理不同的输入特征，用于计算边上的权重
        self.C1 = nn.Linear(input_dim, output_dim, bias=True)  # 处理位置编码
        self.C2 = nn.Linear(input_dim, output_dim, bias=True)  # 处理位置编码

        self.bn_node_h = nn.BatchNorm1d(output_dim)  # 批量归一化节点特征
        self.bn_node_e = nn.BatchNorm1d(output_dim)  # 批量归一化边的特征
        # self.bn_node_p = nn.BatchNorm1d(output_dim)

    def message_func_for_vij(self, edges):
        hj = edges.src['h'] # h_j  计算节点j的特征
        pj = edges.src['p'] # p_j  计算节点j的位置编码特征
        vij = self.A2(torch.cat((hj, pj), -1))  # 生成消息v_ij,用于更新节点特征
        return {'v_ij': vij} 
    
    def message_func_for_pj(self, edges):
        pj = edges.src['p'] # p_j
        return {'C2_pj': self.C2(pj)}  # 源节点的位置编码经过线性层处理
       
    def compute_normalized_eta(self, edges):
        # 归一化边的权重，
        return {'eta_ij': edges.data['sigma_hat_eta'] / (edges.dst['sum_sigma_hat_eta'] + 1e-6)} # sigma_hat_eta_ij/ sum_j' sigma_hat_eta_ij'
      
    def forward(self, g, h, p, e):

        with g.local_scope():
        
            # for residual connection
            h_in = h 
            p_in = p 
            e_in = e 

            # For the h's
            g.ndata['h']  = h 
            g.ndata['A1_h'] = self.A1(torch.cat((h, p), -1)) 
            # self.A2 being used in message_func_for_vij() function
            g.ndata['B1_h'] = self.B1(h)
            g.ndata['B2_h'] = self.B2(h) 

            # For the p's
            g.ndata['p'] = p
            g.ndata['C1_p'] = self.C1(p)
            # self.C2 being used in message_func_for_pj() function

            # For the e's
            g.edata['e'] = e
            g.edata['B3_e'] = self.B3(e) 

            #--------------------------------------------------------------------------------------#
            # Calculation of h
            g.apply_edges(fn.u_add_v('B1_h', 'B2_h', 'B1_B2_h'))
            g.edata['hat_eta'] = g.edata['B1_B2_h'] + g.edata['B3_e']  # 边特征的计算方式，src+dist + src_dist
            g.edata['sigma_hat_eta'] = torch.sigmoid(g.edata['hat_eta'])
            # 在整个图上执行消息传递和聚合操作， copy_e 用于从边特征中复制数据，key，temp_key
            # sum 是聚合函数，msg=m,out=sum_sigma_hat_eta,作为存储结果的键，表示对于每个节点，会将所有来自其相邻边的消息m进行相加并将结果存储在节点特征sum_sigma_hat_eta中
            g.update_all(fn.copy_e('sigma_hat_eta', 'm'), fn.sum('m', 'sum_sigma_hat_eta')) # sum_j' sigma_hat_eta_ij'
            # 归一化边的特征权重
            g.apply_edges(self.compute_normalized_eta) # sigma_hat_eta_ij/ sum_j' sigma_hat_eta_ij'
            g.apply_edges(self.message_func_for_vij) # v_ij
            g.edata['eta_mul_v'] = g.edata['eta_ij'] * g.edata['v_ij'] # eta_ij * v_ij
            g.update_all(fn.copy_e('eta_mul_v', 'm'), fn.sum('m', 'sum_eta_v')) # sum_j eta_ij * v_ij
            g.ndata['h'] = g.ndata['A1_h'] + g.ndata['sum_eta_v']

            # Calculation of p
            g.apply_edges(self.message_func_for_pj) # p_j
            g.edata['eta_mul_p'] = g.edata['eta_ij'] * g.edata['C2_pj'] # eta_ij * C2_pj
            g.update_all(fn.copy_e('eta_mul_p', 'm'), fn.sum('m', 'sum_eta_p')) # sum_j eta_ij * C2_pj
            g.ndata['p'] = g.ndata['C1_p'] + g.ndata['sum_eta_p']

            #--------------------------------------------------------------------------------------#

            # passing towards output
            h = g.ndata['h'] 
            p = g.ndata['p']
            e = g.edata['hat_eta'] 

            # GN from benchmarking-gnns-v1
            # h = h * snorm_n
            
            # batch normalization  
            if self.batch_norm:
                h = self.bn_node_h(h)
                e = self.bn_node_e(e)
                # No BN for p

            # non-linear activation
            h = F.relu(h) 
            e = F.relu(e) 
            p = torch.tanh(p)

            # residual connection
            if self.residual:
                h = h_in + h 
                p = p_in + p
                e = e_in + e 

            # dropout
            h = F.dropout(h, self.dropout, training=self.training)
            p = F.dropout(p, self.dropout, training=self.training)
            e = F.dropout(e, self.dropout, training=self.training)

            return h, p, e
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels)