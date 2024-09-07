import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseRGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, encoder_name="", opn="sub", rel_emb=None, use_cuda=False, analysis=False, pos_init="rw"):
        super(BaseRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_basis = num_basis
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.skip_connect = skip_connect
        self.self_loop = self_loop
        self.encoder_name = encoder_name
        self.use_cuda = use_cuda
        self.run_analysis = analysis
        self.skip_connect = skip_connect
        print("use layer :{}".format(encoder_name))
        self.rel_emb = rel_emb
        self.opn = opn
        self.pos_init = pos_init
        # create rgcn layers
        self.build_model()
        # create initial features
        self.features = self.create_features()


    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers):

            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    # initialize feature for each node
    def create_features(self):
        return None

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g):
        if self.features is not None:
            g.ndata['id'] = self.features
        print("h before GCN message passing")
        print(g.ndata['h'])
        print("h behind GCN message passing")
        for layer in self.layers:
            layer(g)
        print(g.ndata['h'])
        return g.ndata.pop('h')


class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, bias=None, activation=None,
                 self_loop=False, skip_connect=False, dropout=0.0, layer_norm=False):
        super(RGCNLayer, self).__init__()
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.skip_connect = skip_connect
        self.layer_norm = layer_norm

        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
            # self.loop_weight = nn.Parameter(torch.eye(out_feat), requires_grad=False)

        if self.skip_connect:
            self.skip_connect_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))   # 和self-loop不一样，是跨层的计算
            nn.init.xavier_uniform_(self.skip_connect_weight,
                                    gain=nn.init.calculate_gain('relu'))

            self.skip_connect_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.skip_connect_bias)  # 初始化设置为0

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        if self.layer_norm:
            self.normalization_layer = nn.LayerNorm(out_feat, elementwise_affine=False)

    # define how propagation is done in subclass
    def propagate(self, g):
        raise NotImplementedError

    def forward(self, g, prev_h=[]):
        if self.self_loop:
            #print(self.loop_weight)
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)
        # self.skip_connect_weight.register_hook(lambda g: print("grad of skip connect weight: {}".format(g)))
        if len(prev_h) != 0 and self.skip_connect:
            skip_weight = F.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)     # 使用sigmoid，让值在0~1
            # print("skip_ weight")
            # print(skip_weight)
            # print("skip connect weight")
            # print(self.skip_connect_weight)
            # print(torch.mm(prev_h, self.skip_connect_weight))

        self.propagate(g)  # 这里是在计算从周围节点传来的信息

        # apply bias and activation
        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.bias
        # print(len(prev_h))
        if len(prev_h) != 0 and self.skip_connect:   # 两次计算loop_message的方式不一样，前者激活后再加权
            previous_node_repr = (1 - skip_weight) * prev_h
            if self.activation:
                node_repr = self.activation(node_repr)
            if self.self_loop:
                if self.activation:
                    loop_message = skip_weight * self.activation(loop_message)
                else:
                    loop_message = skip_weight * loop_message
                node_repr = node_repr + loop_message
            node_repr = node_repr + previous_node_repr
        else:
            if self.self_loop:
                node_repr = node_repr + loop_message
            if self.layer_norm:
                node_repr = self.normalization_layer(node_repr)
            if self.activation:
                node_repr = self.activation(node_repr)
            # print("node_repr")
            # print(node_repr)
        g.ndata['h'] = node_repr
        return node_repr


class RGCNBasisLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False):
        super(RGCNBasisLayer, self).__init__(in_feat, out_feat, bias, activation)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.is_input_layer = is_input_layer
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # add basis weights
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                self.out_feat))
        if self.num_bases < self.num_rels:
            # linear combination coefficients
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels,
                                                    self.num_bases))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('relu'))

    def propagate(self, g):
        if self.num_bases < self.num_rels:
            # generate all weights from bases
            weight = self.weight.view(self.num_bases,
                                      self.in_feat * self.out_feat)
            weight = torch.matmul(self.w_comp, weight).view(
                self.num_rels, self.in_feat, self.out_feat)
        else:
            weight = self.weight

        if self.is_input_layer:
            def msg_func(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
                embed = weight.view(-1, self.out_feat)
                index = edges.data['type'] * self.in_feat + edges.src['id']
                return {'msg': embed.index_select(0, index)}
        else:
            def msg_func(edges):
                w = weight.index_select(0, edges.data['type'])
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
                return {'msg': msg}

        def apply_func(nodes):
            return {'h': nodes.data['h'] * nodes.data['norm']}

        g.update_all(msg_func, fn.sum(msg='msg', out='h'), apply_func)


class RGCNBlockLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases, bias=None,
                 activation=None, self_loop=False, dropout=0.0, skip_connect=False, layer_norm=False, pos_init="rw", pos_dim=3):
        """
        :param in_feat: 输入特征维度
        :param out_feat: 输出特征维度
        :param num_rels: 关系类型的数量
        :param num_bases: 基础矩阵的数量，用于减少参数数量
        :param bias: 偏置
        :param activation: 激活函数
        :param self_loop:
        :param dropout:
        :param skip_connect:
        :param layer_norm:
        """
        super(RGCNBlockLayer, self).__init__(in_feat, out_feat, bias,
                                             activation, self_loop=self_loop, skip_connect=skip_connect,
                                             dropout=dropout)
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.pos_dim = pos_dim
        self.pos_init = pos_init

        assert self.num_bases > 0

        self.out_feat = out_feat
        # 计算每个基础矩阵的输入和输出矩阵的维度
        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases

        # assuming in_feat and out_feat are both divisible by num_bases
        self.weight = nn.Parameter(torch.Tensor(
            self.num_rels, self.num_bases * self.submat_in * self.submat_out))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # pe
        if self.pos_init == "lap_rw":
            self.embedding_hp = nn.Linear(in_feat+self.pos_dim*2, out_feat)
        else:
            self.embedding_hp = nn.Linear(in_feat + self.pos_dim, out_feat)

    def msg_func(self, edges):
        pos_enc = edges.src['pos_enc']
        weight = self.weight.index_select(0, edges.data['type']).view(
                    -1, self.submat_in, self.submat_out)   # (num_edges, num_bases * submat_in * submat_out)  -> (num_edges * num_bases,submat_in, submat_out)
        node = self.embedding_hp(torch.cat((edges.src['h'], pos_enc), dim=1)).view(-1, 1, self.submat_in)  # (num_edges, in_feat) -> (num_edges, 1, submat_in*num_bases)->(num_edges * num_bases, 1, submat_in)
        msg = torch.bmm(node, weight).view(-1, self.out_feat)  # (num_edges * num_bases, 1, submat_out) -> (num_edges ,submat_out, num_bases)
        return {'msg': msg}

    def propagate(self, g):
        g.update_all(self.msg_func, fn.sum(msg='msg', out='h'), self.apply_func)
        # g.updata_all ({'msg': msg} , fn.sum(msg='msg', out='h'), {'h': nodes.data['h'] * nodes.data[''norm]})

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}


class UnionRGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1,  bias=None,
                 activation=None, self_loop=False, dropout=0.0, skip_connect=False, rel_emb=None, pos_init="rw"):
        super(UnionRGCNLayer, self).__init__()

        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.num_rels = num_rels
        self.rel_emb = None
        self.skip_connect = skip_connect
        self.ob = None
        self.sub = None
        self.pos_dim = 3
        self.pos_init = pos_init

        # WL
        self.weight_neighbor = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))

        # pe
        # self.embedding_p = nn.Linear(self.pos_dim, self.in_feat)

        self.weight_neighbor_p = nn.Parameter(torch.Tensor(self.pos_dim, self.pos_dim))
        nn.init.xavier_uniform_(self.weight_neighbor_p, gain=nn.init.calculate_gain('relu'))
        if self.pos_init == "lap_rw":
            self.embedding_hp = nn.Linear(self.in_feat+self.pos_dim*2, self.out_feat)
        else:
            self.embedding_hp = nn.Linear(self.in_feat+self.pos_dim, self.out_feat)
        self.embedding_rp = nn.Linear(self.out_feat, self.pos_dim)

        if self.self_loop:
            # 传统的自连接权重矩阵，在每个时间步都会对节点的自连接特征进行线性变换（入度大于0才进行处理）
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
            if self.pos_init == "lap_rw":
                self.loop_weight_p = nn.Parameter(torch.Tensor(self.pos_dim*2, self.pos_dim*2))
                nn.init.xavier_uniform_(self.loop_weight_p, gain=nn.init.calculate_gain('relu'))
                self.evolve_loop_weight_p = nn.Parameter(torch.Tensor(self.pos_dim*2, self.pos_dim*2))
                nn.init.xavier_uniform_(self.evolve_loop_weight_p, gain=nn.init.calculate_gain('relu'))
            else:
                self.loop_weight_p = nn.Parameter(torch.Tensor(self.pos_dim, self.pos_dim))
                nn.init.xavier_uniform_(self.loop_weight_p, gain=nn.init.calculate_gain('relu'))
                self.evolve_loop_weight_p = nn.Parameter(torch.Tensor(self.pos_dim, self.pos_dim))
                nn.init.xavier_uniform_(self.evolve_loop_weight_p, gain=nn.init.calculate_gain('relu'))
            # 演化的自连接权重矩阵，默认对所有节点的特征进行线性变换
            self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))



        if self.skip_connect:
            self.skip_connect_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))   # 和self-loop不一样，是跨层的计算
            nn.init.xavier_uniform_(self.skip_connect_weight,gain=nn.init.calculate_gain('relu'))
            self.skip_connect_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.skip_connect_bias)  # 初始化设置为0

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def propagate(self, g):
        # 消息传递，聚合，特征更新
        # msg=msg 指聚合的消息来自msg_fun 返回的消息 msg中，聚合结构存储在目标节点的特征 h中
        g.update_all(lambda x: self.msg_func(x), fn.sum(msg='msg', out='h'), self.apply_func)

    def propagate_p(self, g):
        g.update_all(lambda x: self.msg_func_p(x), fn.sum(msg='msg_p', out='pos_enc'), self.apply_func_p)

    def forward(self, g, prev_h, emb_rel):
        self.rel_emb = emb_rel
        # self.sub = sub
        # self.ob = ob
        pos_enc = g.ndata['pos_enc']
        if self.self_loop:
            #loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            # masked_index = torch.masked_select(torch.arange(0, g.number_of_nodes(), dtype=torch.long), (g.in_degrees(range(g.number_of_nodes())) > 0))
            masked_index = torch.masked_select(
                torch.arange(0, g.number_of_nodes(), dtype=torch.long).to("cuda"),  # 选择的张量索引
                (g.in_degrees(range(g.number_of_nodes())) > 0))  # 挑选的布尔掩码
            loop_message = torch.mm(g.ndata['h'], self.evolve_loop_weight)
            loop_message[masked_index, :] = torch.mm(g.ndata['h'], self.loop_weight)[masked_index, :]
            loop_message_p = torch.mm(pos_enc, self.evolve_loop_weight_p)
            loop_message_p[masked_index, :] = torch.mm(pos_enc, self.loop_weight_p)[masked_index, :]
        if len(prev_h) != 0 and self.skip_connect:
            skip_weight = F.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)     # 使用sigmoid，让值在0~1

        # calculate the neighbor message with weight_neighbor
        self.propagate(g)
        # self.propagate_p(g)

        pos_enc = g.ndata['pos_enc']

        node_repr = g.ndata['h']

        # print(len(prev_h))
        if len(prev_h) != 0 and self.skip_connect:  # 两次计算loop_message的方式不一样，前者激活后再加权
            if self.self_loop:
                node_repr = node_repr + loop_message
            node_repr = skip_weight * node_repr + (1 - skip_weight) * prev_h
        else:
            if self.self_loop:
                node_repr = node_repr + loop_message
                pos_enc = pos_enc + loop_message_p

        if self.activation:
            node_repr = self.activation(node_repr)
            pos_enc = self.activation(pos_enc)
        if self.dropout is not None:
            node_repr = self.dropout(node_repr)
            pos_enc = self.dropout(pos_enc)
        g.ndata['h'] = node_repr
        g.ndata['pos_enc'] = pos_enc
        return node_repr, pos_enc

    def msg_func(self, edges):
        # if reverse:
        #     relation = self.rel_emb.index_select(0, edges.data['type_o']).view(-1, self.out_feat)
        # else:
        #     relation = self.rel_emb.index_select(0, edges.data['type_s']).view(-1, self.out_feat)
        # 选择出特定关系类型的嵌入并reshape      （按行，类型）
        relation = self.rel_emb.index_select(0, edges.data['type']).view(-1, self.out_feat)
        edge_type = edges.data['type']
        # print(edges.src['pos_enc'].shape, edges.src['h'].shape)
        # cat(h, pe)
        pos_enc = edges.src["pos_enc"]
        edges.src['h'] = self.embedding_hp(torch.cat((edges.src['h'], pos_enc), dim=1))
        node = edges.src['h'].view(-1, self.out_feat)
        # node = torch.cat([torch.matmul(node[:edge_num // 2, :], self.sub),
        #                  torch.matmul(node[edge_num // 2:, :], self.ob)])
        # node = torch.matmul(node, self.sub)

        # after add inverse edges, we only use message pass when h as tail entity
        # 这里计算的是每个节点发出的消息，节点发出消息时其作为头实体
        # msg = torch.cat((node, relation), dim=1)
        msg = node + relation
        # calculate the neighbor message with weight_neighbor
        # print(node.shape, relation.shape)
        # print(msg.shape, self.weight_neighbor.shape)
        msg = torch.mm(msg, self.weight_neighbor)
        return {'msg': msg}


    def msg_func_p(self, edges):
        relation = self.rel_emb.index_select(0, edges.data['type'])
        # edge_type = edges.data['type']
        # node = edges.src['pos_enc'].view(-1, 6)
        # print(self.embedding_rp(relation).shape, edges.src['pos_enc'].shape)
        pos_enc = edges.src["pos_enc"]
        msg = pos_enc + self.embedding_rp(relation)
        # calculate the neighbor message with weight_neighbor
        msg = torch.mm(msg, self.weight_neighbor_p)
        return {'msg_p': msg}


    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}

    def apply_func_p(self, nodes):
        return {'pos_enc': nodes.data["pos_enc"] * nodes.data['norm']}
