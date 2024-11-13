import torch
import torch.nn as nn
import torch.optim as optim
from dgl.utils import expand_as_pair
from dgl import function as fn
from dgl.base import DGLError
from dgl.nn.functional import edge_softmax
import numpy as np

from methods.gtan_psa.my_add import GTANWithPSA

cat_features = ["Target",
                "Type",
                "Location"]


class PosEncoding(nn.Module):

    def __init__(self, dim, device, base=10000, bias=0):

        super(PosEncoding, self).__init__()
        """
        Initialize the posencoding component
        :param dim: the encoding dimension 
		:param device: where to train model
		:param base: the encoding base
		:param bias: the encoding bias
        """
        p = []
        sft = []
        for i in range(dim):
            b = (i - i % 2) / dim
            p.append(base ** -b)
            if i % 2:
                sft.append(np.pi / 2.0 + bias)
            else:
                sft.append(bias)
        self.device = device
        self.sft = torch.tensor(
            sft, dtype=torch.float32).view(1, -1).to(device)
        self.base = torch.tensor(p, dtype=torch.float32).view(1, -1).to(device)

    def forward(self, pos):
        with torch.no_grad():
            if isinstance(pos, list):
                pos = torch.tensor(pos, dtype=torch.float32).to(self.device)
            pos = pos.view(-1, 1)
            x = pos / self.base + self.sft
            return torch.sin(x)


class TransEmbedding(nn.Module):

    def __init__(self, df=None, device='cpu', dropout=0.2, in_feats=82, cat_features=None):
        """
        Initialize the attribute embedding and feature learning compoent

        :param df: the feature
                :param device: where to train model
                :param dropout: the dropout rate
                :param in_feat: the shape of input feature in dimension 1
                :param cat_feature: category features
        """
        super(TransEmbedding, self).__init__()
        self.time_pe = PosEncoding(dim=in_feats, device=device, base=100)
        #time_emb = time_pe(torch.sin(torch.tensor(df['time_span'].values)/86400*torch.pi))
        self.cat_table = nn.ModuleDict({col: nn.Embedding(max(df[col].unique(
        ))+1, in_feats).to(device) for col in cat_features if col not in {"Labels", "Time"}})
        self.label_table = nn.Embedding(3, in_feats, padding_idx=2).to(device)
        self.time_emb = None
        self.emb_dict = None
        self.label_emb = None
        self.cat_features = cat_features
        self.forward_mlp = nn.ModuleList(
            [nn.Linear(in_feats, in_feats) for i in range(len(cat_features))])
        self.dropout = nn.Dropout(dropout)

    def forward_emb(self, df):
        if self.emb_dict is None:
            self.emb_dict = self.cat_table
        # print(self.emb_dict)
        # print(df['trans_md'])
        support = {col: self.emb_dict[col](
            df[col]) for col in self.cat_features if col not in {"Labels", "Time"}}
        #self.time_emb = self.time_pe(torch.sin(torch.tensor(df['time_span'])/86400*torch.pi))
        #support['time_span'] = self.time_emb
        #support['labels'] = self.label_table(df['labels'])
        return support

    def forward(self, df):
        support = self.forward_emb(df)
        output = 0
        for i, k in enumerate(support.keys()):
            # if k =='time_span':
            #    print(df[k].shape)
            support[k] = self.dropout(support[k])
            support[k] = self.forward_mlp[i](support[k])
            output = output + support[k]
        return output


# 输入：{Tensor:(2321,126)}、输出：Tensor: (681,256)}
class TransformerConv(nn.Module):

    def __init__(self,
                 in_feats,  # in_feats = 126
                 out_feats,  # out_feats = 64
                 num_heads,  # num_heads=4
                 bias=True,
                 allow_zero_in_degree=False,
                 skip_feat=True,
                 gated=True,  # gated=True
                 layer_norm=True,  # layer_norm=True
                 activation=nn.PReLU(),
                 distance=None):  # activation=nn.PReLU()
        """
        Initialize the transformer layer.
        Attentional weights are jointly optimized in an end-to-end mechanism with graph neural networks and fraud detection networks.
            :param in_feat: the shape of input feature
            :param out_feats: the shape of output feature
            :param num_heads: the number of multi-head attention
            :param bias: whether to use bias
            :param allow_zero_in_degree: whether to allow zero in degree
            :param skip_feat: whether to skip some feature
            :param gated: whether to use gate
            :param layer_norm: whether to use layer regularization
            :param activation: the type of activation function
        """

        super(TransformerConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)  # expand_as_pair将输入特征维度 in_feats 拓展为一对源和目标特征维度
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree  # allow_zero_in_degree=False
        self._num_heads = num_heads

        # 构造transformer中的q/k/v三个线性层
        self.lin_query = nn.Linear(
            self._in_src_feats, self._out_feats*self._num_heads, bias=bias)  # (输入：126, 输出64*4=256, bias=True)
        self.lin_key = nn.Linear(
            self._in_src_feats, self._out_feats*self._num_heads, bias=bias)  # (输入：126, 输出64*4=256, bias=True)
        self.lin_value = nn.Linear(
            self._in_src_feats, self._out_feats*self._num_heads, bias=bias)  # (输入：126, 输出64*4=256, bias=True)


        if skip_feat:  # skip_feat=True
            self.skip_feat = nn.Linear(
                self._in_src_feats, self._out_feats*self._num_heads, bias=bias)  # (输入：126, 输出64*4=256, bias=True)
        else:
            self.skip_feat = None
        if gated:  # gated=True
            self.gate = nn.Linear(
                3*self._out_feats*self._num_heads, 1, bias=bias)  # (输入：3*64*4=768, 输出1, bias=True)，前面有个3的原因：目标节点原始特征、目标节点聚合后特征、两者特征之差，将这3者沿最后一个维度拼接了。
        else:
            self.gate = None
        if layer_norm:  # layer_norm=True
            self.layer_norm = nn.LayerNorm(self._out_feats*self._num_heads)  # 64*4=256
        else:
            self.layer_norm = None
        self.activation = activation  # activation=nn.PReLU()

    # 调用代码： h = self.output_drop(self.layers[l+4](blocks[l], h))# h ={Tensor:(2321,126)} => h = {Tensor: (681,256)}
    # graph：当前batch根据“种子节点”决定的子图 [Block(num src nodes=2321, num dst nodes=681, num edges=7506), Block(num src nodes=681, num dst nodes=128.,num edges=1408)]
    # feat：是合并了【数字特征】+【类别特征】+【标签】的特征,feat ={Tensor:(2321,126)}
    def forward(self, graph, feat, get_attention=False, distance=1):  # feat ={Tensor:(2321,126)}
        """
        Description: Transformer Graph Convolution
        :param graph: input graph
            :param feat: input feat
            :param get_attention: whether to get attention
        """

        graph = graph.local_var()  # 将graph对象转换为局部变量

        if not self._allow_zero_in_degree:
            if (graph.in_degrees() == 0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                               'output for those nodes will be invalid. '
                               'This is harmful for some applications, '
                               'causing silent performance regression. '
                               'Adding self-loop on the input graph by '
                               'calling `g = dgl.add_self_loop(g)` will resolve '
                               'the issue. Setting ``allow_zero_in_degree`` '
                               'to be `True` when constructing this module will '
                               'suppress the check and let the code run.')

        # check if feat is a tuple
        if isinstance(feat, tuple):
            h_src = feat[0]
            h_dst = feat[1]
        else:
            h_src = feat  # feat ={Tensor:(2321,126)}，h_src ={Tensor:(2321,126)}
            # 当从一个大图中采样一个子图时，通常会包括目标节点及其邻居节点，所有邻居节点就是这里的【源节点】，因为是有自环图
            # 注意：这里的graph = graph.local_var()是局部变量，是针对每个子图的graph，所以h_dst取了h_src的前多少个，不是一直不变的，每个子图是不一样的
            # h_dst ={Tensor:(681,126)}
            h_dst = h_src[:graph.number_of_dst_nodes()]  # graph = {DGLBlock} [Block(num src nodes=2321, num dst nodes=681, num edges=7506), Block(num src nodes=681, num dst nodes=128.,num edges=1408)]

        # Step 0. q, k, v
        #  view 方法重新调整形状
        # _num_heads 是注意力头的数量
        # _out_feats 是每个头的输出特征维度
        #  self.lin_query = nn.Linear(self._in_src_feats, self._out_feats*self._num_heads, bias=bias)  # (输入：126, 输出64*4=256, bias=True)
        q_src = self.lin_query(h_src).view(-1, self._num_heads, self._out_feats)  # q src={Tensor:(2321, 4, 64)}
        k_dst = self.lin_key(h_dst).view(-1, self._num_heads, self._out_feats)  # k src={Tensor:(681, 4, 64)}
        v_src = self.lin_value(h_src).view(-1, self._num_heads, self._out_feats)  # v src={Tensor:(2321, 4, 64)}

        # Assign features to nodes
        # 更新图中源节点的特征数据，将源节点的特征 q_src 和 v_src 存储在名为 'ft' 和 'ft_v' 的特征字段中
        graph.srcdata.update({'ft': q_src, 'ft_v': v_src})
        # 更新图中目标节点的特征数据，将目标节点的特征 k_dst 存储在名为 'ft' 的特征字段中
        graph.dstdata.update({'ft': k_dst})

        # Step 1. dot product
        graph.apply_edges(fn.u_dot_v('ft', 'ft', 'a'))  # 对图中的每条边， 计算每条边连接的两个节点特征的点积。存储结果在边的数据属性 'a' 中,这代表了原始的注意力相关度量

        # print("graph.edata['a']1111111111")
        # print(graph.edata['a'].shape)
        if distance == 2:
            # 计算衰减系数：例如指数衰减，可以根据需要选择不同的衰减方式
            graph.edata['decay'] = torch.exp(-self.decay_factor * distance)  # 自定义的衰减因子
            graph.edata['a'] *= graph.edata['decay']  # 应用衰减因子
        # print("graph.edata['a']222222222222")
        # print(graph.edata['a'].shape)

# 公式3
        # Step 2. edge softmax 去计算注意力分数，公式3
        # **用于执行幂运算
        # 再将a归一化后，更新边属性【graph.edata['sa']】
        graph.edata['sa'] = edge_softmax(graph, graph.edata['a'] / self._out_feats**0.5)  # 这部分代码实际上是对边的得分进行归一化处理，这里除以self._out_feats**0.5是为了实现特征的缩放，有助于数值稳定

        # Step 3. 对图中的所有节点进行更新操作。更新操作由两个函数组成
        graph.update_all(fn.u_mul_e('ft_v', 'sa', 'attn'),  # 对于每个目标节点，将其所有入边对应的 ft_v（源节点的值特征）乘以对应的注意力权重（sa），结果存储在节点的属性attn中
                         fn.sum('attn', 'agg_u'))  # 然后，将所有入边的加权特征求和，得到目标节点更新后的表示（agg_u）。这一步是节点信息的聚合过程。
        # 在消息传递的过程中，每个源节点向其邻居（目标节点）发送信息，聚合操作（如 fn.sum）是在目标节点上进行的。这意味着，目标节点从其所有邻居（源节点）接收信息并将其聚合

        # 将目标节点更新后的表示（agg_u）重新塑形，以便后续处理。这里假设使用了多头注意力机制，因此将输出特征维度乘以头数（self._num_heads）
        # rst = {Tensor: (681,256)}是聚合特征
        rst = graph.dstdata['agg_u'].reshape(-1,self._out_feats*self._num_heads)  # 第二维：每个特征向量的长度*头的数量（4*64=256）
# 公式4
        if self.skip_feat is not None:  # skip_feat=True
            skip_feat = self.skip_feat(feat[:graph.number_of_dst_nodes()])  # skip_feat = {Tensor: (681,256)}提取前 `graph.number_of_dst_nodes()` 个节点的特征，记为 `skip_feat`
            if self.gate is not None:
                gate = torch.sigmoid(  # self.gate：(输入：3*64*4=768, 输出1, bias=True)
                    self.gate(  # 将 `skip_feat`、`rst` 和 `skip_feat - rst` 沿着最后一个维度拼接起来，然后经过 `torch.sigmoid` 函数处理，得到门控值 `gate`
                        torch.concat([skip_feat, rst, skip_feat - rst], dim=-1)))  # skip_feat - rst：跳跃连接特征和变换后特征之间的差异，提供了关于输入和输出特征之间差异的信息，有助于模型理解变换前后的变化程度
                rst = gate * skip_feat + (1 - gate) * rst  # rst = {Tensor: (681,256)}。通过门控值 `gate` 对 `skip_feat` 和 `rst` 进行加权相加，更新 `rst`
            else:
                rst = skip_feat + rst

        if self.layer_norm is not None:
            rst = self.layer_norm(rst)  # self.layer_norm = nn.LayerNorm(self._out_feats*self._num_heads)  # 64*4=256

        if self.activation is not None:
            rst = self.activation(rst)  # rst = {Tensor: (681,256)}

        # get_attention=False
        # if get_attention:
        #     return rst, graph.edata['sa']
        # else:
        return rst  # rst = {Tensor: (681,256)}


class GraphAttnModel(nn.Module):
    def __init__(self,
                 in_feats,
                 hidden_dim,
                 n_layers,
                 n_classes,
                 heads,
                 activation,
                 skip_feat=True,
                 gated=True,
                 layer_norm=True,
                 post_proc=True,
                 n2v_feat=True,
                 drop=None,
                 ref_df=None,
                 cat_features=None,
                 nei_features=None,
                 device='cpu'):
        """
        Initialize the GTAN-GNN model
        :param in_feats: the shape of input feature
                :param hidden_dim: model hidden layer dimension
                :param n_layers: the number of GTAN layers
                :param n_classes: the number of classification
                :param heads: the number of multi-head attention
                :param activation: the type of activation function
                :param skip_feat: whether to skip some feature
                :param gated: whether to use gate
        :param layer_norm: whether to use layer regularization
                :param post_proc: whether to use post processing
                :param n2v_feat: whether to use n2v features
        :param drop: whether to use drop
                :param ref_df: whether to refer other node features
                :param cat_features: category features
                :param nei_features: neighborhood statistic features
        :param device: where to train model
        """

        super(GraphAttnModel, self).__init__()
        self.in_feats = in_feats
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.heads = heads
        self.activation = activation
        #self.input_drop = lambda x: x
        self.input_drop = nn.Dropout(drop[0])
        self.drop = drop[1]
        self.output_drop = nn.Dropout(self.drop)
        # self.pn = PairNorm(mode=pairnorm)
        if n2v_feat:
            self.n2v_mlp = TransEmbedding(
                ref_df, device=device, in_feats=in_feats, cat_features=cat_features)
        else:
            self.n2v_mlp = lambda x: x
        self.layers = nn.ModuleList()
        self.layers.append(nn.Embedding(
            n_classes+1, in_feats, padding_idx=n_classes))
        self.layers.append(
            nn.Linear(self.in_feats, self.hidden_dim*self.heads[0]))
        self.layers.append(
            nn.Linear(self.in_feats, self.hidden_dim*self.heads[0]))
        self.layers.append(nn.Sequential(nn.BatchNorm1d(self.hidden_dim*self.heads[0]),
                                         nn.PReLU(),
                                         nn.Dropout(self.drop),
                                         nn.Linear(self.hidden_dim *
                                                   self.heads[0], in_feats)
                                         ))

        # build multiple layers
        self.layers.append(TransformerConv(in_feats=self.in_feats,
                                           out_feats=self.hidden_dim,
                                           num_heads=self.heads[0],
                                           skip_feat=skip_feat,
                                           gated=gated,
                                           layer_norm=layer_norm,
                                           activation=self.activation,
                                           distance=1))

        for l in range(0, (self.n_layers - 1)):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.layers.append(TransformerConv(in_feats=self.hidden_dim * self.heads[l - 1],
                                               out_feats=self.hidden_dim,
                                               num_heads=self.heads[l],
                                               skip_feat=skip_feat,
                                               gated=gated,
                                               layer_norm=layer_norm,
                                               activation=self.activation,
                                               distance=1))
        if post_proc:
            self.layers.append(nn.Sequential(nn.Linear(self.hidden_dim * self.heads[-1], self.hidden_dim * self.heads[-1]),
                                             nn.BatchNorm1d(
                                                 self.hidden_dim * self.heads[-1]),
                                             nn.PReLU(),
                                             nn.Dropout(self.drop),
                                             nn.Linear(self.hidden_dim * self.heads[-1], self.n_classes)))
        else:
            self.layers.append(nn.Linear(self.hidden_dim *
                               self.heads[-1], self.n_classes))
        self.psa = GTANWithPSA(self.in_feats, self.in_feats)

    def forward(self, blocks, features, labels, n2v_feat=None):
        """
        :param blocks: train blocks
        :param features: train features  (|input|, feta_dim)
        :param labels: train labels (|input|, )
        :param n2v_feat: whether to use n2v features
        """

        if n2v_feat is None:
            h = features
        else:
            h = self.n2v_mlp(n2v_feat)
            h = features + h
            # print("***********h.shape****************")
            # print(h.shape)
            # print("**************features.shape*************")
            # print(features.shape)
            h = self.psa(h)

        label_embed = self.input_drop(self.layers[0](labels))
        label_embed = self.layers[1](h) + self.layers[2](label_embed)
        label_embed = self.layers[3](label_embed)
        h = h + label_embed  # residual
        # print("*****h.shape*****")
        # print(h.shape)  # h[11944, 25]

        # h = self.restore_dim(h)

        # l会取 0 和 1
        for l in range(self.n_layers):  # n_layers: 2
            # 针对的是GraphAttnModel中的（4）和（5），观测可以发现，每次传入的h是上一次更新后的h，所以（4）和（5），即2个TransformerConv是串行的，且下一个的输入是上一个的输出
            # res=self.layers[l + 4](blocks[l], h, l+1)
            # print("res:")
            # print(res.shape)
            # print(res)
            # h = self.output_drop(self.layers[l + 4](blocks[l], h))  # h ={Tensor:(2321,126)} => h = {Tensor: (681,256)} => h = {Tensor: (128,256)}
            h = self.output_drop(self.layers[l + 4](blocks[l], h, l+1))  # h ={Tensor:(2321,126)} => h = {Tensor: (681,256)} => h = {Tensor: (128,256)}

        # 走GraphAttnModel中的（6），5个子层达到的效果：256->2
        logits = self.layers[-1](h)  # nn.Linear(self.hidden_dim * self.heads[-1], self.n_classes)))  # （64*4=256,2）

        return logits  # logits = {Tensor: (128,2)}