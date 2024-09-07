import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import math
from scipy import sparse as sp
from scipy.sparse.linalg import norm

from haruhi_exp.decoder.convtranse import ConvTransE
from haruhi_exp.decoder.convtransr import ConvTransR

"""
    GatedGCN and GatedGCN-LSPE
    
"""
from haruhi_exp.layers.gatedgcn_lspe_layer import GatedGCNLSPELayer
from haruhi_exp.layers.gatedgcn_layer import GatedGCNLayer
class GatedGCNNet(nn.Module):
    def __init__(self, decoder_params, num_nodes, num_rels, net_params, task_weight, pe_init, pe_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.task_weight = task_weight
        self.pe_init = pe_init
        self.pe_dim = pe_dim
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        self.n_layers = net_params['num_layers']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.readout = net_params["readout"]
        # self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        
        self.use_lapeig_loss = net_params['use_lapeig_loss']
        self.lambda_loss = net_params['lambda_loss']
        self.alpha_loss = net_params['alpha_loss']
        self.entity_prediction = net_params["entity_prediction"]
        self.relation_prediction = net_params["relation_prediction"]
        self.loss_r = torch.nn.CrossEntropyLoss()
        self.loss_e = torch.nn.CrossEntropyLoss()
        self.layer_norm = net_params["layer_norm"]
        
        if self.pe_init in ['rand_walk', 'lap_pe']:
            self.embedding_p = nn.Linear(self.pe_dim, hidden_dim)

        self.embedding_h = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_normal_(self.embedding_h.weight)

        self.embedding_e = nn.Embedding(num_rels*2, hidden_dim)
        nn.init.xavier_normal_(self.embedding_e.weight)

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        if self.pe_init == 'rand_walk':
            # LSPE
            self.layers = nn.ModuleList([GatedGCNLSPELayer(hidden_dim, hidden_dim, dropout,
                                                        self.batch_norm, residual=self.residual) for _ in range(self.n_layers-1) ]) 
            self.layers.append(GatedGCNLSPELayer(hidden_dim, out_dim, dropout, self.batch_norm, residual=self.residual))
        else: 
            # NoPE or LapPE
            self.layers = nn.ModuleList([GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                        self.batch_norm, residual=self.residual, graph_norm=False) for _ in range(self.n_layers-1) ]) 
            self.layers.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, residual=self.residual, graph_norm=False))
        

        if self.pe_init == 'rand_walk':
            self.p_out = nn.Linear(out_dim, self.pe_dim)
            self.Whp = nn.Linear(out_dim+self.pe_dim, out_dim)
        
        if decoder_params["name"] == "convtranse":
            self.decoder_ob = ConvTransE(num_nodes, hidden_dim, decoder_params["input_dropout"], decoder_params["hidden_dropout"], decoder_params["feat_dropout"]).to(self.device)
            self.rdecoder = ConvTransR(num_rels, hidden_dim, decoder_params["input_dropout"], decoder_params["hidden_dropout"], decoder_params["feat_dropout"]).to(self.device)
        
    def forward(self, g_list, static_graph):
        gate_list = []
        degree_list = []
        history_embs = []
        history_rels = torch.Tensor([]).to(self.device)
        temp_dic = {}
        for i, g in enumerate(g_list):
            # print("-------------->", g.number_of_nodes, self.num_rels)
            # input embedding
            node_id = g.ndata["id"].squeeze()
            g.ndata['h'] = self.embedding_h(node_id)
            h = self.in_feat_dropout(g.ndata['h'])
            p = g.ndata["p"]

            if self.pe_init in ['rand_walk', 'lap_pe']:
                p = self.embedding_p(p)

            if self.pe_init == 'lap_pe':
                h = h + p
                p = None
            e = self.embedding_e(g.edata["type"])
            # print("g.edata['type']", g.edata["type"].shape, "e.shape", e.shape, g.edata['type'])

            print()


            # convnets
            for conv in self.layers:
                h, p, e = conv(g, h, p, e)

            g.ndata['h'] = h

            if self.pe_init == 'rand_walk':
                # Implementing p_g = p_g - torch.mean(p_g, dim=0)
                p = self.p_out(p)
                g.ndata['p'] = p
                means = dgl.mean_nodes(g, 'p')
                batch_wise_p_means = means.repeat_interleave(g.batch_num_nodes(), 0)
                p = p - batch_wise_p_means

                # Implementing p_g = p_g / torch.norm(p_g, p=2, dim=0)
                g.ndata['p'] = p
                g.ndata['p2'] = g.ndata['p']**2
                norms = dgl.sum_nodes(g, 'p2')
                norms = torch.sqrt(norms)
                batch_wise_p_l2_norms = norms.repeat_interleave(g.batch_num_nodes(), 0)
                p = p / batch_wise_p_l2_norms
                g.ndata['p'] = p

                # Concat h and p
                hp = self.Whp(torch.cat((g.ndata['h'], g.ndata['p']), dim=-1))
                g.ndata['h'] = hp
            for index, emb in zip(g.edata["type"].tolist(), e):
                if index not in temp_dic:
                    temp_dic[index] = []
                temp_dic[index].append(emb)
            history_embs.append(g.ndata['h'])
        temp_dic = dict(sorted(temp_dic.items()))
        for emb_list in temp_dic.values():
            avg_emb = sum(emb_list) / len(emb_list)
            print(avg_emb.shape, "avg_emb.shape")
            history_rels = torch.cat((history_rels, avg_emb.unsqueeze(0)), dim=0)
        # lens = set()
        # lenn = set()
        # for n, e in zip(history_embs, history_rels):
        #     print(type(n), type(e))
        #     lens.add(e.shape)
        #     lenn.add(n.shape)
        # print(lens, lenn)
        print("num_rels", self.num_rels)
        print("num_history_rels.shape", history_rels.shape)
        return history_embs, history_rels, 1

    def predict(self, test_graph, num_rels, static_graph, test_triplets):
        with torch.no_grad():
            inverse_test_triplets = test_triplets[:, [2, 1, 0]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels  # 将逆关系换成逆关系的id
            all_triples = torch.cat((test_triplets, inverse_test_triplets))

            evolve_embs,  r_emb = self.forward(test_graph, static_graph)
            embedding = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

            score = self.decoder_ob.forward(embedding, r_emb, all_triples, mode="test")
            score_rel = self.rdecoder.forward(embedding, r_emb, all_triples, mode="test")
            return all_triples, score, score_rel

    def get_loss(self, glist, triples, static_graph):

        loss_ent = torch.zeros(1).cuda().to(self.device)
        loss_rel = torch.zeros(1).cuda().to(self.device)
        loss_static = torch.zeros(1).cuda().to(self.device)

        inverse_triples = triples[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triples, inverse_triples])
        all_triples = all_triples.to(self.device)

        evolve_embs, r_emb, static_emb = self.forward(glist, static_graph)
        pre_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
        # print("--------------begin", len(evolve_embs), evolve_embs[-1].shape, pre_emb.shape)
        if self.entity_prediction:
            scores_ob = self.decoder_ob.forward(pre_emb, r_emb, all_triples).view(-1, self.num_nodes)
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

        
        if self.use_lapeig_loss:
            # Loss B: Laplacian Eigenvector Loss --------------------------------------------
            g = self.g
            n = g.number_of_nodes()

            # Laplacian 
            A = g.adjacency_matrix(scipy_fmt="csr")
            N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
            L = sp.eye(n) - N * A * N

            p = g.ndata['p']
            pT = torch.transpose(p, 1, 0)
            loss_b_1 = torch.trace(torch.mm(torch.mm(pT, torch.Tensor(L.todense()).to(self.device)), p))

            # Correct batch-graph wise loss_b_2 implementation; using a block diagonal matrix
            bg = dgl.unbatch(g)
            batch_size = len(bg)
            P = sp.block_diag([bg[i].ndata['p'].detach().cpu() for i in range(batch_size)])
            PTP_In = P.T * P - sp.eye(P.shape[1])
            loss_b_2 = torch.tensor(norm(PTP_In, 'fro')**2).float().to(self.device)

            loss_b = ( loss_b_1 + self.lambda_loss * loss_b_2 ) / ( self.pe_init * batch_size * n)

            del bg, P, PTP_In, loss_b_1, loss_b_2

            # loss = loss_a + self.alpha_loss * loss_b

        
        return loss_ent, loss_rel, loss_b

    
    
    
    