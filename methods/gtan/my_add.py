import networkx as nx
import dgl
import torch


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


def combine_local_and_global_features(local_features, global_position_encoding):
    # 例如，通过拼接局部特征和全局位置编码
    # 确保 global_position_encoding 的形状与 local_features 兼容
    global_position_encoding = global_position_encoding.unsqueeze(1)  # 将形状从 [11944] 扩展为 [11944, 1]

    # 重复 global_position_encoding 以匹配 local_features 的第二个维度
    global_position_encoding = global_position_encoding.expand(-1, local_features.size(1))

    # 将 global_position_encoding 添加到 local_features 上，保持形状为 [11944, 25]
    combined_features = local_features + global_position_encoding

    return combined_features

