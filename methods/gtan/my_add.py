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





