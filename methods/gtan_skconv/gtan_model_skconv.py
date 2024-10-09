import torch
import torch.nn as nn
from dgl.utils import expand_as_pair
from dgl import function as fn
from dgl.base import DGLError
from dgl.nn.functional import edge_softmax
import numpy as np

from methods.gtan.my_add import GlobalContextBlock, SKConv

cat_features = ["Target",
                "Type",
                "Location"]


class PosEncoding(nn.Module):

    def __init__(self, dim, device, base=10000, bias=0):  # dim = {int} 126表示位置编码的维度，即每个位置的向量长度;base = {int} 100是用于计算位置编码值的基数

        super(PosEncoding, self).__init__()
        """
        Initialize the posencoding component
        :param dim: the encoding dimension 
		:param device: where to train model
		:param base: the encoding base
		:param bias: the encoding bias
        """
        p = []  # p是位置编码的系数
        sft = []
        for i in range(dim):
            b = (i - i % 2) / dim  # 如果 i 是偶数，则 b = (i - i % 2) / dim；这样确保了偶数维度的系数会逐渐减小，奇数维度的系数保持不变
            p.append(base ** -b)  # base ** -b 表示 base 的 -b 次方
            if i % 2:
                sft.append(np.pi / 2.0 + bias)  # 相位偏移 sft：对于偶数维度，相位偏移为 np.pi / 2.0 + bias，对于奇数维度，相位偏移为 bias。
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

    # self.n2v_mlp = TransEmbedding(ref_df, device=device, in_feats=in_feats, cat_features=cat_features)=>df=ref_df
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
        self.time_pe = PosEncoding(dim=in_feats, device=device, base=100)  # 为模型创建一个位置编码（Positional Encoding）模块
        #time_emb = time_pe(torch.sin(torch.tensor(df['time_span'].values)/86400*torch.pi))
        self.cat_table = nn.ModuleDict({col: nn.Embedding(max(df[col].unique(  # 对于每个分类特征列，创建一个nn.Embedding，嵌入维度为in_feats，词汇表大小为该列中唯一值的最大值加1,这些嵌入层被存储在cat_table字典中
        ))+1, in_feats).to(device) for col in cat_features if col not in {"Labels", "Time"}})
        self.label_table = nn.Embedding(3, in_feats, padding_idx=2).to(device)  # 嵌入维度为in_feats，词汇表大小为3，并指定padding_idx=2（即索引为2的嵌入向量将被用作填充标记）
        self.time_emb = None
        self.emb_dict = None
        self.label_emb = None
        self.cat_features = cat_features
        self.forward_mlp = nn.ModuleList(  # forward_mlp是一个nn.ModuleList，包含3个线性层，每个线性层的输入和输出维度均为in_feats
            [nn.Linear(in_feats, in_feats) for i in range(len(cat_features))])
        self.dropout = nn.Dropout(dropout)  # dropout=0.2
# 分别利用对应的embedding模型对“类别特征”获取嵌入表示
    def forward_emb(self, df):  # df = {dict: 3} {'Target': tensor([238,8,0,...,15,0,0]),'Location':tensor([2,0,0,.12,12,0]),'Type': tensor([33,5,0,...,10, 0, 0])}
        if self.emb_dict is None:
            self.emb_dict = self.cat_table  # (cat_table): ModuleDict((Target): Embedding(886, 126)(Location):Embedding(296,126)(Type): Embedding(166, 126))
        # print(self.emb_dict)
        # print(df['trans_md'])
        support = {col: self.emb_dict[col](  # # 为每个特征列进行embedding处理
            df[col]) for col in self.cat_features if col not in {"Labels", "Time"}}  # cat_features = ["Target","Type", "Location"]
        return support  # support = {dict: 3} {'Target': tensor([[-0.6434,]...[,1.0358]], 'Location': tensor([[-0.3648,]...[-0.0962]]), 'Type': tensor([[ 0.8478,]...[,0.6512]])}

    # 调用代码：h = self.n2v_mlp(n2v_feat)  # n2v_feat是类别特征（字典格式）
    def forward(self, df):  # df = {dict: 3} {'Target': tensor([238,8,0,...,15,0,0]),'Location':tensor([2,0,0,.12,12,0]),'Type': tensor([33,5,0,...,10, 0, 0])}
        support = self.forward_emb(df)  # support = {dict: 3} {'Target': tensor([[-0.6434,]...[,1.0358]], 'Location': tensor([[-0.3648,]...[-0.0962]]), 'Type': tensor([[ 0.8478,]...[,0.6512]])}
        output = 0
        for i, k in enumerate(support.keys()):  # i是索引，k是特征列的名称
            support[k] = self.dropout(support[k])  # 应用dropout操作
            support[k] = self.forward_mlp[i](support[k])  # (forward_mlp): ModuleList((0-2): 3 x Linear(in_features=126, out_features=126, bias=True)）


            # 将处理后的3种“类别特征”的值相加
            output = output + support[k]  # output ={Tensor:(2321,126)}->output ={Tensor:(2321,126)},也就是将指定特征列的向量相加
        return output  # output ={Tensor:(2321,126)}


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
                 activation=nn.PReLU()):  # activation=nn.PReLU()
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
            self._in_src_feats, self._out_feats*self._num_heads, bias=bias)
        self.lin_key = nn.Linear(
            self._in_src_feats, self._out_feats*self._num_heads, bias=bias)
        self.lin_value = nn.Linear(
            self._in_src_feats, self._out_feats*self._num_heads, bias=bias)


        if skip_feat:  # skip_feat=True
            self.skip_feat = nn.Linear(
                self._in_src_feats, self._out_feats*self._num_heads, bias=bias)
        else:
            self.skip_feat = None
        if gated:  # gated=True
            self.gate = nn.Linear(
                3*self._out_feats*self._num_heads, 1, bias=bias)  # 3*64*4=768
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
    def forward(self, graph, feat, get_attention=False):  # feat ={Tensor:(2321,126)}
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
# 公式3
        # Step 2. edge softmax 去计算注意力分数，公式3
        # **用于执行幂运算
        # 再将a归一化后，更新边属性【graph.edata['sa']】
        graph.edata['sa'] = edge_softmax(graph, graph.edata['a'] / self._out_feats**0.5)# graph.edata['a'] / self._out_feats**0.5 这部分代码实际上是对边的得分进行归一化处理

        # Step 3. 对图中的所有节点进行更新操作。更新操作由两个函数组成
        graph.update_all(fn.u_mul_e('ft_v', 'sa', 'attn'),  # 表示将节点上的特征ft_v与节点邻接边的特征sa做乘法，结果存储在节点的属性attn中
                         fn.sum('attn', 'agg_u'))  # 表示对节点上的attn属性进行汇总求和，结果存储在节点的属性agg_u中

        # 在消息传递的过程中，每个源节点向其邻居（目标节点）发送信息，聚合操作（如 fn.sum）是在目标节点上进行的。这意味着，目标节点从其所有邻居（源节点）接收信息并将其聚合
        # rst = {Tensor: (681,256)}是聚合特征
        rst = graph.dstdata['agg_u'].reshape(-1,self._out_feats*self._num_heads)  # 第二维：每个特征向量的长度*头的数量
# 公式4
        if self.skip_feat is not None:  # skip_feat=True
            skip_feat = self.skip_feat(feat[:graph.number_of_dst_nodes()])  # skip_feat = {Tensor: (681,256)}提取前 `graph.number_of_dst_nodes()` 个节点的特征，记为 `skip_feat`
            if self.gate is not None:
                gate = torch.sigmoid(  # gate = {Tensor: (681,1)}
                    self.gate(  # 将 `skip_feat`、`rst` 和 `skip_feat - rst` 沿着最后一个维度拼接起来，然后经过 `torch.sigmoid` 函数处理，得到门控值 `gate`
                        torch.concat([skip_feat, rst, skip_feat - rst], dim=-1)))  # skip_feat - rst：跳跃连接特征和变换后特征之间的差异，提供了关于输入和输出特征之间差异的信息，有助于模型理解变换前后的变化程度
                rst = gate * skip_feat + (1 - gate) * rst  # rst = {Tensor: (681,256)}。通过门控值 `gate` 对 `skip_feat` 和 `rst` 进行加权相加，更新 `rst`
            else:
                rst = skip_feat + rst

        if self.layer_norm is not None:
            rst = self.layer_norm(rst)

        if self.activation is not None:
            rst = self.activation(rst)  # rst = {Tensor: (681,256)}

        # get_attention=False
        if get_attention:
            return rst, graph.edata['sa']
        else:
            return rst  # rst = {Tensor: (681,256)}


class GraphAttnModel(nn.Module):
    def __init__(self,
                 in_feats,  # in_feats：126
                 hidden_dim,  # hidden_dim=args['hid_dim']//4,  # hid_dim: 256 =>hidden_dim: 64
                 n_layers,  # # n_layers: 2
                 n_classes,  # n_classes=2
                 heads,  # [4,4]
                 activation,  # activation=nn.PReLU()
                 skip_feat=True,
                 gated=True,
                 layer_norm=True,
                 post_proc=True,
                 n2v_feat=True,
                 drop=None,  # dropout: [0.2, 0.1]
                 ref_df=None,  # ref_df=feat_df.iloc[train_idx],  # ref_df={DataFrame: (62304,126)}
                 cat_features=None,  # S_FFSD数据集：cat_features = {dict: 3}，其余两个cat_features = []
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
        self.in_feats = in_feats  # in_feats = {int} 126
        self.hidden_dim = hidden_dim  # hidden_dim = {int} 64
        self.n_layers = n_layers  # n_layers = {int} 2
        self.n_classes = n_classes  # n_classes = {int} 2
        self.heads = heads  # heads = {list: 2} [4, 4]
        self.activation = activation  # activation ={PReLU} PReLU(num parameters=1)
        #self.input_drop = lambda x: x
        self.input_drop = nn.Dropout(drop[0])
        self.drop = drop[1]  # drop = {list: 2} [0.2, 0.1]
        self.output_drop = nn.Dropout(self.drop)  # self.drop=0.1
        # self.pn = PairNorm(mode=pairnorm)
        if n2v_feat:
            self.n2v_mlp = TransEmbedding(  # TransEmbedding类对象内部构建了一个基于多层感知器（MLP）的模型
                ref_df, device=device, in_feats=in_feats, cat_features=cat_features)  # ref df={DataFrame:(62304,126)}
        else:
            self.n2v_mlp = lambda x: x
        self.layers = nn.ModuleList()
        # layers:0 = {Embedding: (3, 126)}
        self.layers.append(nn.Embedding(
            n_classes+1, in_feats, padding_idx=n_classes))

        # layers:1-2 = {Linear: (126, 256)}
        self.layers.append(
            nn.Linear(self.in_feats, self.hidden_dim*self.heads[0]))  # hidden_dim:64，heads = [4, 4]
        self.layers.append(
            nn.Linear(self.in_feats, self.hidden_dim*self.heads[0]))

        # layers:3
        # 0-3
        self.layers.append(nn.Sequential(nn.BatchNorm1d(self.hidden_dim*self.heads[0]),  # hidden_dim:64，heads = [4, 4]
                                         nn.PReLU(),
                                         nn.Dropout(self.drop),
                                         nn.Linear(self.hidden_dim *
                                                   self.heads[0], in_feats)  # Linear:(256,126)
                                         ))

        # layers:4
        # 7层
        self.layers.append(TransformerConv(in_feats=self.in_feats,  # in_feats = 126
                                           out_feats=self.hidden_dim,  # hidden_dim = {int} 64
                                           num_heads=self.heads[0],  # heads = [4, 4]
                                           skip_feat=skip_feat,  # skip_feat=True
                                           gated=gated,  # gated=True
                                           layer_norm=layer_norm,  # layer_norm=True
                                           activation=self.activation))  # activation=nn.PReLU()
        self.skconv = SKConv(in_ch=self.in_feats)
        # layers:5（同layer 4）
        # 7层
        for l in range(0, (self.n_layers - 1)):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.layers.append(TransformerConv(in_feats=self.hidden_dim * self.heads[l - 1],
                                               out_feats=self.hidden_dim,
                                               num_heads=self.heads[l],
                                               skip_feat=skip_feat,
                                               gated=gated,
                                               layer_norm=layer_norm,
                                               activation=self.activation))
        # layers:6
        # 0-4
        if post_proc:  # post_proc=True
            self.layers.append(nn.Sequential(nn.Linear(self.hidden_dim * self.heads[-1], self.hidden_dim * self.heads[-1]),  # （64*4=256，64*4=256）
                                             nn.BatchNorm1d(
                                                 self.hidden_dim * self.heads[-1]),
                                             nn.PReLU(),
                                             nn.Dropout(self.drop),  # self.drop=0.1
                                             nn.Linear(self.hidden_dim * self.heads[-1], self.n_classes)))  # （64*4=256,2）
        else:
            self.layers.append(nn.Linear(self.hidden_dim *
                               self.heads[-1], self.n_classes))

    # 调用代码：train_batch_logits = model(blocks, batch_inputs, lpa_labels, batch_work_inputs)
    # ①blocks =  { list:2[Block(num src nodes=2321, num dst nodes=681, num edges=7506), Block(num src nodes=681, num dst nodes=128.,num edges=1408)]；
    # ②features ={Tensor:(2321,126)}；“数字特征”
    # ③labels ={Tensor:(2321,)}；“数字特征”对应的“标签”
    # ④n2v_feat = “类别特征”{dict: 3} {'Target': tensor([238,8,0,...,15,0,0]),'Location':tensor([2,0,0,.12,12,0]),'Type': tensor([33,5,0,...,10, 0, 0])}
    def forward(self, blocks, features, labels, n2v_feat=None):
        """
        :param blocks: train blocks
        :param features: train features  (|input|, feta_dim)
        :param labels: train labels (|input|, )
        :param n2v_feat: whether to use n2v features 
        """

        if n2v_feat is None:
            h = features
        else:# n2v_feat = {dict: 3} {'Target': tensor([238,8,0,...,15,0,0]),'Location':tensor([2,0,0,.12,12,0]),'Type': tensor([33,5,0,...,10, 0, 0])}
            h = self.n2v_mlp(n2v_feat)  # h ={Tensor:(2321,126)}，处理了3个“类别特征”，具体就是将其各自embedding之后在对应相加，合并成一个h。
            h = features + h  # h ={Tensor:(2321,126)}，这步是将“数字特征”和“n2v类别特征”相加，得到一个h。
# labels = {Tensor: (2321,)}
        label_embed = self.input_drop(self.layers[0](labels))  # label_embed{Tensor: (2321,126)}

        # 这步是将【数字特征和n2v类别特征合并后的特征】与【标签特征】相加，label_embed{Tensor: (2321,256)}
        label_embed = self.layers[1](h) + self.layers[2](label_embed)
        label_embed = self.layers[3](label_embed)  # label_embed{Tensor: (2321,126)}
        h = h + label_embed  # label embed ={Tensor:(2321,126)};h ={Tensor:(2321,126)} 是标签传播步骤

        h = self.skconv(h)

        # l会取 0 和 1
        for l in range(self.n_layers):  # n_layers: 2
            # 针对的是GraphAttnModel中的（4）和（5），观测可以发现，每次传入的h是上一次更新后的h，所以（4）和（5），即2个TransformerConv是串行的，且下一个的输入是上一个的输出
            h = self.output_drop(self.layers[l+4](blocks[l], h))  # h ={Tensor:(2321,126)} => h = {Tensor: (681,256)}

        # 走GraphAttnModel中的（6），5个子层达到的效果：256->2
        logits = self.layers[-1](h)

        return logits  # logits = {Tensor: (128,2)}
