import networkx as nx
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


# 假设已经有一个 DGLGraph 对象 graph
# def gtan_main(feat_df, graph, train_idx, test_idx, labels, args, cat_features):
def centrality(graph):
    # 将 DGLGraph 转换为 NetworkX 图
    # 将 DGLGraph 转换为 NetworkX 图
    nx_graph = graph.to_networkx()

    # 检查并转换多重图
    if isinstance(nx_graph, (nx.MultiGraph, nx.MultiDiGraph)):
        nx_graph = nx.Graph(nx_graph)  # 将多重图转换为简单图

    # 计算特征向量中心性（作为示例）
    eigenvector_centrality = nx.eigenvector_centrality(nx_graph, max_iter=500)  # 增加迭代次数到 500

    # 将中心性结果转为一个张量
    centrality_tensor = torch.tensor([eigenvector_centrality[node] for node in nx_graph.nodes])

    # 将中心性结果添加到 DGLGraph 的节点特征中
    graph.ndata['centrality'] = centrality_tensor

    # 打印一下中心性张量，确保正确计算
    # print(centrality_tensor.shape)
class MultiScaleMessagePassing(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads=4):
        super(MultiScaleMessagePassing, self).__init__()
        self.gcns = nn.ModuleList([nn.Linear(in_feats, out_feats) for _ in range(num_heads)])
        self.attn = nn.ModuleList([nn.Linear(in_feats, out_feats) for _ in range(num_heads)])
        self.num_heads = num_heads
        # self.gate = nn.Linear(num_heads * out_feats, out_feats)
        self.gate = nn.Linear(num_heads * out_feats, out_feats)
    def forward(self, g, h):
        # print("h.shape*************", h.shape)  # h.shape:[3087, 256])
        outputs = []
        for i in range(self.num_heads):
            gcn_out = self.gcns[i](h)
            # print("gcn_out.shape*************", gcn_out.shape)
            attn_out = self.attn[i](h)
            # print("attn_out.shape*************", attn_out.shape)
            combined_out = gcn_out + attn_out
            # print("combined_out.shape*************", combined_out.shape)
            outputs.append(combined_out)
        multi_scale_out = torch.cat(outputs, dim=-1)
        # print("multi_scale_out.shape********", multi_scale_out.shape)
        gated_out = self.gate(multi_scale_out)
        # print("gated_out.shape********", gated_out.shape)
        return gated_out

def combine_local_and_global_features(local_features, global_position_encoding):
    # 例如，通过拼接局部特征和全局位置编码
    # 确保 global_position_encoding 的形状与 local_features 兼容
    global_position_encoding = global_position_encoding.unsqueeze(1)  # 将形状从 [11944] 扩展为 [11944, 1]

    # 重复 global_position_encoding 以匹配 local_features 的第二个维度
    global_position_encoding = global_position_encoding.expand(-1, local_features.size(1))

    # 将 global_position_encoding 添加到 local_features 上，保持形状为 [11944, 25]
    combined_features = local_features + global_position_encoding

    return combined_features

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


import torch
import torch.nn as nn

class GlobalContextBlock(nn.Module):
    def __init__(self, inplanes, ratio, pooling_type="att", fusion_types=('channel_mul')) -> None:
        super().__init__()
        # 定义有效的融合类型
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert pooling_type in ['avg', 'att']
        assert len(fusion_types) > 0, 'at least one fusion should be used'

        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_type = fusion_types

        # 使用1x1卷积（线性变换）处理不同的特征维度
        if pooling_type == 'att':
            self.conv_mask = nn.Linear(inplanes, 1)
            self.softmax = nn.Softmax(dim=1)
        else:
            self.avg_pool = lambda x: torch.mean(x, dim=1, keepdim=True)

        # 通道加操作
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Linear(self.inplanes, self.planes),
                nn.LayerNorm(self.planes),
                nn.ReLU(inplace=True),
                nn.Linear(self.planes, self.inplanes)
            )
        else:
            self.channel_add_conv = None

        # 通道乘操作
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Linear(self.inplanes, self.planes),
                nn.LayerNorm(self.planes),
                nn.ReLU(inplace=True),
                nn.Linear(self.planes, self.inplanes)
            )
        else:
            self.channel_mul_conv = None

    # 定义空间池化函数
    def spatial_pool(self, x):
        if self.pooling_type == 'att':
            context_mask = self.conv_mask(x)  # (3, 1)
            context_mask = self.softmax(context_mask)  # 在特征维度上应用softmax
            context = torch.sum(context_mask * x, dim=0, keepdim=True)  # 加权求和，得到全局上下文 (1, inplanes)
        else:
            context = self.avg_pool(x)  # 对所有特征取平均值
        return context

    # 定义前向传播函数
    def forward(self, x):
        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))  # 放缩权重
            out = out * channel_mul_term  # 与 x 进行相乘
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out


# def main():
#     # 假设我们有3种特征：Target, Location, Type，每种特征有10维
#     target = torch.randn(10)
#     location = torch.randn(10)
#     type_feat = torch.randn(10)
#
#     # 将三种特征拼接成一个 (3, 10) 的张量
#     x = torch.stack([target, location, type_feat], dim=0)
#
#     # 定义一个 GlobalContextBlock 实例
#     ratio = 0.5  # 隐藏层的特征维度为输入特征维度的0.5倍
#     pooling_type = 'att'  # 使用注意力机制作为池化方式
#     fusion_types = ('channel_mul', 'channel_add')  # 同时使用通道相乘和通道相加的融合方式
#     gcb = GlobalContextBlock(inplanes=10, ratio=ratio, pooling_type=pooling_type, fusion_types=fusion_types)
#
#     # 将输入特征张量传递给 GlobalContextBlock
#     out = gcb(x)
#
#     # 输出结果的形状和结果
#     print("输入特征张量的形状:", x.shape)
#     print("输出特征张量的形状:", out.shape)
#     print("输出特征张量:", out)
#
#
# if __name__ == "__main__":
#     main()


# 输入特征张量的形状: torch.Size([16, 64, 256, 256])
# 输出特征张量的形状: torch.Size([16, 64, 256, 256])
import torch
import torch.nn as nn


class SKConv(nn.Module):
    def __init__(self, in_ch, M=3, r=4, L=32):
        super().__init__()
        d = max(int(in_ch / r), L)  # 计算d的值，确保d不小于L，以免信息损失
        self.M = M  # 分支数量
        self.in_ch = in_ch  # 输入通道数

        # 使用线性层代替卷积层
        self.fc_list = nn.ModuleList([
            nn.Linear(in_ch, in_ch) for _ in range(M)
        ])

        self.fc = nn.Linear(in_ch, d)  # 全连接层，将特征向量降维到d
        self.fcs = nn.ModuleList([
            nn.Linear(d, in_ch) for _ in range(M)
        ])
        self.softmax = nn.Softmax(dim=1)  # Softmax激活，用于归一化注意力向量

    def forward(self, x):
        # x 的形状应为 [batch_size, channels]
        feas = None
        for i, fc in enumerate(self.fc_list):
            # 对输入x应用每个分支的线性操作
            fea = fc(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                # 将不同分支的特征拼接在一起
                feas = torch.cat([feas, fea], dim=1)

        # 将所有分支的特征相加，得到统一特征
        fea_U = torch.sum(feas, dim=1)
        # 通过全连接层fc将fea_U映射到向量fea_z
        fea_z = self.fc(fea_U)

        attention_vectors = None
        for i, fc in enumerate(self.fcs):
            # 为每个分支生成注意力向量
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                # 将不同分支的注意力向量拼接在一起
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)

        # 对注意力向量应用Softmax激活，进行归一化处理
        attention_vectors = self.softmax(attention_vectors)

        # 将注意力向量应用于拼接后的特征，通过加权求和得到最终的输出特征图
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


def get_max_s(in_feats):
    """计算 n 的最大除数（不包括 n 本身），仅在 1 到 10 的范围内查找"""
    for i in range(10, 0, -1):
        if in_feats % i == 0 and i != in_feats:
            return i
    return 1  # 默认返回 1，处理特殊情况



# 调用：GTANWithPSA(64*4, 64*4)
# 输出维度==输入维度
class GTANWithPSA(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GTANWithPSA, self).__init__()
        self.in_feats = in_feats  # self.in_feats：64*4=256
        self.out_feats = out_feats  # self.out_feats：64*4=256
        self.S = get_max_s(in_feats)  # 分成 S 个子空间（8）
        self.fc = nn.Linear(in_feats // self.S, out_feats// self.S )  # self.fc = nn.Linear(32，32)
        self.fc2 = nn.Linear(out_feats// self.S, out_feats)  # self.fc2 = nn.Linear(32,256)

    def forward(self, h):  # h:torch.Size([128, 256])
        # Step 2: 将节点特征分割成多个子空间
        h_split = torch.chunk(h, self.S, dim=1)  # h_split[0]:torch.Size([128, 32]),h_split 是一个包含 8 个张量的列表，每个张量的形状为 [128, 32]
        # Step 3: 对每个子空间应用注意力权重
        weights = [F.softmax(self.fc(h_i), dim=1) for h_i in h_split]  # weights[0]:torch.Size([128, 32]),weights 是一个包含 8 个张量的列表，每个张量的形状为 [128, 32]
        # len(weights):8
        h_fused = sum(w * h_i for w, h_i in zip(weights, h_split))  # h_fused： [128, 32]
        h_fused=self.fc2(h_fused)  # h_fused：[128, 256]

        # Step 4: 输出最终的特征
        return h_fused  # h_fused：[128, 256]

if __name__ == "__main__":
    x = torch.randn(16, 64, 256, 256)
    sk = SKConv(in_ch=64, M=3, G=1, r=2)
    out = sk(x)
    print("输入特征张量的形状:", x.shape)
    print("输出特征张量的形状:", out.shape)
    # in_ch 数据输入维度，M为分指数，G为Conv2d层的组数，基本设置为1，r用来进行求线性层输出通道的。
