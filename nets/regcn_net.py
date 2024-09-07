#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time ： 2024/8/28 22:26
@Auth ： chenxi
@File ：regcn_net.py
@IDE ：PyCharm
"""
import math

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from rgcn.layers import RGCNBlockLayer as RGCNLayer
from haruhi_exp.layers.regcn_layer import UnionRGCNLayer, RGCNBlockLayer, BaseRGCN
from haruhi_exp.decoder.convtranse import ConvTransE
from haruhi_exp.decoder.convtransr import ConvTransR


class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:  # 如果不是第一层
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True  # 当前层不是第一个隐藏层
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                                  activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc,
                                  rel_emb=self.rel_emb, pos_init=self.pos_init)
        else:
            raise NotImplementedError

    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "uvrgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [],  r[i])  # UnionRGCNLayer的forword 参数，g, prev_h, rel_emb
            return g.ndata['h'], g.ndata['pos_enc']  # 返回图的节点数据h，并从图数据中删除h
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            # g.ndata['h'] = torch.cat(g.ndata['h'], pos_encoder.lap_positional_encoding_tkg(g, 3), dim=1)
            # g.ndata['h'] = torch.cat(g.ndata['h'], pos_encoder.rw_positional_encoding_tkg(g, 3), dim=1)

            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')


class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, num_static_rels, num_words, h_dim, opn,
                 sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', weight=1, discount=0, angle=0, use_static=False,
                 entity_prediction=False, relation_prediction=False, use_cuda=False,
                 gpu=0, analysis=False, pos_init='rw', pos_dim=3):
        """
        :param decoder_name:  解码器
        :param encoder_name: 编码器
        :param num_ents: 实体数量
        :param num_rels: 关系数量
        :param num_static_rels: 唯一静态关系数量
        :param num_words: 唯一实体数
        :param h_dim: 隐藏单元嵌入维度
        :param opn: 组合节点和关系的嵌入
        :param sequence_len: history_len
        :param num_bases: 为关系分配多少权重块
        :param num_basis: 为compgcn 分配多少权重块
        :param num_hidden_layers: 传播的轮数
        :param dropout: rgcncell 和 RGCNBlockLayer
        :param self_loop: 进行自连接
        :param skip_connect: 计算prev_h 的权重
        :param layer_norm:
        :param input_dropout:
        :param hidden_dropout:
        :param feat_dropout:
        :param aggregation:
        :param weight:
        :param discount:
        :param angle:
        :param use_static:
        :param entity_prediction:
        :param relation_prediction:
        :param use_cuda:
        :param gpu:
        :param analysis:
        """
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.weight = weight
        self.discount = discount
        self.use_static = use_static
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.emb_rel = None
        self.gpu = gpu
        self.pos_init = pos_init
        self.pos_dim = pos_dim
        self.num_layers = num_hidden_layers
        self.device = torch.device(f'cuda:{self.gpu}' if use_cuda else 'cpu')

        self.w1 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w1)

        self.w2 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w2)

        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)

        # ps
        self.Whp = nn.Linear(h_dim+self.pos_dim, h_dim)

        if self.use_static:
            self.words_emb = torch.nn.Parameter(torch.Tensor(self.num_words, h_dim), requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_static_rels * 2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=False,
                                                    skip_connect=False, pos_init=self.pos_init, pos_dim=self.pos_dim,)
            self.static_loss = torch.nn.MSELoss()

        self.loss_r = torch.nn.CrossEntropyLoss()
        self.loss_e = torch.nn.CrossEntropyLoss()

        self.rgcn = RGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda,
                             analysis,
                             pos_init=self.pos_init)

        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.global_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_uniform_(self.global_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)
        self.global_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.global_gate_bias)

        # GRU cell for relation evolving
        self.relation_cell_1 = nn.GRUCell(self.h_dim * 2, self.h_dim)

        # decoder
        if decoder_name == "convtranse":
            self.decoder_ob = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder = ConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        else:
            raise NotImplementedError

        # global
        self.global_atten = torch.nn.MultiheadAttention(h_dim, 1, dropout=0.2, batch_first=True)
        if pos_init == "lap_rw":
            self.embedding_hp = nn.Linear(h_dim+self.pos_dim*2, h_dim)
        else:
            self.embedding_hp = nn.Linear(h_dim + self.pos_dim, h_dim)
        self.tanh = nn.Tanh()
    def forward(self, g_list, static_graph):
        gate_list = []
        degree_list = []


        if self.use_static:

            static_graph = static_graph.to(self.device)
            static_graph.ndata['h'] = torch.cat((self.dynamic_emb, self.words_emb), dim=0)  # 演化得到的表示，和wordemb满足静态图约束
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            self.h = static_emb
        else:
            self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None

        history_embs = []

        for i, g in enumerate(g_list):
            g = g.to(self.device)
            temp_e = self.h[g.r_to_e]  # 保存了和当前关系有关的实体的embedding
            x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().to(self.device)
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]:span[1], :]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                x_input[r_idx] = x_mean  # 和r有关的实体嵌入的平均值
            if i == 0:
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.emb_rel)  # 第1层输入
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            else:
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.h_0)  # 第2层输出==下一时刻第一层输入
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            # pos_enc = self.embedding_p(g.ndata["pos_enc"])
            current_h, current_p = self.rgcn.forward(g, self.h, [self.h_0]*self.num_layers)
            atten_in = self.tanh(self.embedding_hp(torch.cat((g.ndata['h'], g.ndata['pos_enc']), dim=1)))
            global_h, _ = self.global_atten(atten_in.unsqueeze(1), atten_in.unsqueeze(1), atten_in.unsqueeze(1))
            global_h = global_h.squeeze(1)
            # p_means = dgl.mean_nodes(g, 'pos_enc')
            # batch_wise_p_means = p_means.repeat_interleave(g.batch_num_nodes(), 0)
            # p = current_p - batch_wise_p_means
            #
            # # Implementing p_g = p_g / torch.norm(p_g, p=2, dim=0)
            # g.ndata['pos_enc'] = p
            # g.ndata['p2'] = g.ndata['pos_enc'] ** 2
            # norms = dgl.sum_nodes(g, 'p2')
            # norms = torch.sqrt(norms)
            # batch_wise_p_l2_norms = norms.repeat_interleave(g.batch_num_nodes(), 0)
            # p = p / batch_wise_p_l2_norms
            # g.ndata['pos_enc'] = p
            #
            # # Concat h and p
            # hp = self.Whp(torch.cat((current_h, g.ndata['pos_enc']), dim=-1))
            # g.ndata['h'] = hp
            # hp = current_h
            current_h = F.normalize(current_h) if self.layer_norm else current_h
            # current_p = self.embedding_p(current_p)
            time_weight = F.sigmoid(torch.mm(self.h, self.time_gate_weight) + self.time_gate_bias)
            self.h = time_weight * current_h + (1 - time_weight) * self.h
            # 计算全局门控权重
            global_weight = F.sigmoid(torch.mm(self.h, self.global_gate_weight) + self.global_gate_bias)
            # 将时间更新后的节点表示 self.h 与全局表示 global_h 融合
            fused_h = global_weight * global_h + (1 - global_weight) * self.h
            history_embs.append(fused_h)
        return history_embs, static_emb, self.h_0, gate_list, degree_list, current_p

    def predict(self, test_graph, num_rels, static_graph, test_triplets):
        with torch.no_grad():
            inverse_test_triplets = test_triplets[:, [2, 1, 0]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels  # 将逆关系换成逆关系的id
            all_triples = torch.cat((test_triplets, inverse_test_triplets))

            evolve_embs, _, r_emb, _, _, _ = self.forward(test_graph, static_graph)
            embedding = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

            score = self.decoder_ob.forward(embedding, r_emb, all_triples, mode="test")
            score_rel = self.rdecoder.forward(embedding, r_emb, all_triples, mode="test")
            return all_triples, score, score_rel

    def get_loss(self, glist, triples, static_graph, use_cuda):
        """
        :param glist:
        :param triplets:
        :param static_graph:
        :param use_cuda:
        :return:
        """
        loss_ent = torch.zeros(1).cuda().to(self.device) if use_cuda else torch.zeros(1)
        loss_rel = torch.zeros(1).cuda().to(self.device) if use_cuda else torch.zeros(1)
        loss_static = torch.zeros(1).cuda().to(self.device) if use_cuda else torch.zeros(1)

        inverse_triples = triples[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triples, inverse_triples])
        all_triples = all_triples.to(self.device)

        evolve_embs, static_emb, r_emb, _, _, pos_enc = self.forward(glist, static_graph, use_cuda)
        pre_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

        if self.entity_prediction:
            scores_ob = self.decoder_ob.forward(pre_emb, r_emb, all_triples).view(-1, self.num_ents)
            loss_ent += self.loss_e(scores_ob, all_triples[:, 2])

        if self.relation_prediction:
            score_rel = self.rdecoder.forward(pre_emb, r_emb, all_triples, mode="train").view(-1, 2 * self.num_rels)
            loss_rel += self.loss_r(score_rel, all_triples[:, 1])

        if self.use_static:
            if self.discount == 1:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180) * (time_step + 1)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
            elif self.discount == 0:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
        return loss_ent, loss_rel, loss_static
