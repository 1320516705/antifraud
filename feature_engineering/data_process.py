# %%
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.io import loadmat
import torch
import dgl
import random
import os
import time
import argparse
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
# from . import *
DATADIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", "data/")

'''
    针对每一行而言：在其所决定的时间窗口内，对原数据添加列，针对在时间窗口内的数据进行数据的整理，并将整理的信息作为新列添加到原数据中；
'''
def featmap_gen(tmp_df=None):
    """
    Handle S-FFSD dataset and do some feature engineering
    :param tmp_df: the feature of input dataset
    """
    # time_span = [2, 5, 12, 20, 60, 120, 300, 600, 1500, 3600, 10800, 32400, 64800, 129600,
    #              259200]  # Increase in the number of time windows to increase the characteristics.
    time_span = [2, 3, 5, 15, 20, 50, 100, 150,
                 200, 300, 864, 2590, 5100, 10000, 24000]  # 设置时间窗口
    time_name = [str(i) for i in time_span]  # 时间窗口名称
    time_list = tmp_df['Time']  # 时间列
    post_fe = []
    for trans_idx, trans_feat in tqdm(tmp_df.iterrows()):  # 遍历每一行，trans_idx是索引，trans_feat是当前行特征
        new_df = pd.Series(trans_feat)  # 将特征转换为Series类型，是当前行特征
        temp_time = new_df.Time  # 将当前行的时间列赋值给temp_time
        temp_amt = new_df.Amount  # 将当前行的金额列赋值给temp_amt
        for length, tname in zip(time_span, time_name):
            lowbound = (time_list >= temp_time - length)  # 对当前行而言，找出当前行时间之前的所有行，并且找出当前行时间之后的所有行，是list
            upbound = (time_list <= temp_time)
            correct_data = tmp_df[lowbound & upbound]  # 找出对于当前行的时间而言，在当前时间窗口内的行，直到遍历完所有的时间窗口。
            new_df['trans_at_avg_{}'.format(  # 所有满足条件的行的金额列的平均值、标准差、总和、偏差，行数量，目标数量，位置数量，类型数量
                tname)] = correct_data['Amount'].mean()
            new_df['trans_at_totl_{}'.format(
                tname)] = correct_data['Amount'].sum()
            new_df['trans_at_std_{}'.format(
                tname)] = correct_data['Amount'].std()
            new_df['trans_at_bias_{}'.format(
                tname)] = temp_amt - correct_data['Amount'].mean()
            new_df['trans_at_num_{}'.format(tname)] = len(correct_data)
            new_df['trans_target_num_{}'.format(tname)] = len(
                correct_data.Target.unique())
            new_df['trans_location_num_{}'.format(tname)] = len(
                correct_data.Location.unique())
            new_df['trans_type_num_{}'.format(tname)] = len(
                correct_data.Type.unique())
        post_fe.append(new_df)
    return pd.DataFrame(post_fe)


def sparse_to_adjlist(sp_matrix, filename):
    """
    Transfer sparse matrix to adjacency list
    :param sp_matrix: the sparse matrix
    :param filename: the filename of adjlist
    """
    # add self loop
    homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
    # create adj_list
    adj_lists = defaultdict(set)
    edges = homo_adj.nonzero()
    for index, node in enumerate(edges[0]):
        adj_lists[node].add(edges[1][index])
        adj_lists[edges[1][index]].add(node)
    with open(filename, 'wb') as file:
        pickle.dump(adj_lists, file)
    file.close()


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def MinMaxScaling(data):
    mind, maxd = data.min(), data.max()
    # return mind + (data - mind) / (maxd - mind)
    return (data - mind) / (maxd - mind)


def k_neighs(
    graph: dgl.DGLGraph,
    center_idx: int,
    k: int,
    where: str,
    choose_risk: bool = False,
    risk_label: int = 1
) -> torch.Tensor:
    """return indices of risk k-hop neighbors

    Args:
        graph (dgl.DGLGraph): dgl graph dataset
        center_idx (int): center node idx
        k (int): k-hop neighs
        where (str): {"predecessor", "successor"}
        risk_label (int, optional): value of fruad label. Defaults to 1.
    """
    target_idxs: torch.Tensor
    if k == 1:
        if where == "in":
            neigh_idxs = graph.predecessors(center_idx)
        elif where == "out":
            neigh_idxs = graph.successors(center_idx)

    elif k == 2:
        if where == "in":
            subg_in = dgl.khop_in_subgraph(
                graph, center_idx, 2, store_ids=True)[0]
            neigh_idxs = subg_in.ndata[dgl.NID][subg_in.ndata[dgl.NID] != center_idx]
            # delete center node itself
            neigh1s = graph.predecessors(center_idx)
            neigh_idxs = neigh_idxs[~torch.isin(neigh_idxs, neigh1s)]
        elif where == "out":
            subg_out = dgl.khop_out_subgraph(
                graph, center_idx, 2, store_ids=True)[0]
            neigh_idxs = subg_out.ndata[dgl.NID][subg_in.ndata[dgl.NID] != center_idx]
            neigh1s = graph.successors(center_idx)
            neigh_idxs = neigh_idxs[~torch.isin(neigh_idxs, neigh1s)]

    neigh_labels = graph.ndata['label'][neigh_idxs]
    if choose_risk:
        target_idxs = neigh_idxs[neigh_labels == risk_label]
    else:
        target_idxs = neigh_idxs

    return target_idxs


def count_risk_neighs(
    graph: dgl.DGLGraph,
    risk_label: int = 1
) -> torch.Tensor:

    ret = []
    for center_idx in graph.nodes():
        neigh_idxs = graph.successors(center_idx)  # 获取当前节点 center_idx 的所有邻居节点索引
        neigh_labels = graph.ndata['label'][neigh_idxs]  #  访问邻居节点的标签数据
        risk_neigh_num = (neigh_labels == risk_label).sum()  # 当前节点具有风险标签的邻居节点数量
        ret.append(risk_neigh_num)

    return torch.Tensor(ret)

'''
    分别计算距离中心节点1和2跳的邻居节点的“度”（第二维是0）的求和 以及“风险节点”（第二维是1）的求和
'''
def feat_map():
    tensor_list = []
    feat_names = []
    for idx in tqdm(range(graph.num_nodes())):  # 获取所有节点的所有一跳和二跳前置邻居节点索引，及其 度 和 风险数量 的总和
        neighs_1_of_center = k_neighs(graph, idx, 1, "in")  # 获取当前节点 idx 的所有一跳前置邻居节点索引
        neighs_2_of_center = k_neighs(graph, idx, 2, "in")  # 获取当前节点 idx 的所有二跳前置邻居节点索引
        tensor = torch.FloatTensor([
            edge_feat[neighs_1_of_center, 0].sum().item(),  # 与中心节点距离一跳的邻居节点的度的总和，neighs_1_of_center：第一个维度对应行，而0作为第二个维度对应列
            # edge_feat[neighs_1_of_center, 0].std().item(),
            edge_feat[neighs_2_of_center, 0].sum().item(),  # 与中心节点距离二跳的邻居节点的度的总和
            # edge_feat[neighs_2_of_center, 0].std().item(),
            edge_feat[neighs_1_of_center, 1].sum().item(),  # 与中心节点距离一跳的邻居节点的风险数量的总和
            # edge_feat[neighs_1_of_center, 1].std().item(),
            edge_feat[neighs_2_of_center, 1].sum().item(),  # 与中心节点距离二跳的邻居节点的风险数量的总和
            # edge_feat[neighs_2_of_center, 1].std().item(),
        ])
        tensor_list.append(tensor)
    feat_names = ["1hop_degree", "2hop_degree","1hop_riskstat", "2hop_riskstat"]  # 张量的特征名称
    tensor_list = torch.stack(tensor_list)  # stack:在第0维度上进行堆叠, 将张量列表转换为张量
    return tensor_list, feat_names


if __name__ == "__main__":

    set_seed(42)

    # %%
    """
        For Yelpchi dataset
        Code partially from https://github.com/YingtongDou/CARE-GNN
    """
    print(f"processing YELP data...")
    yelp = loadmat(os.path.join(DATADIR, 'YelpChi.mat'))  # 使用 scipy.io 模块中的 loadmat 函数加载数据，返回一个包含 MATLAB 文件中的数据的字典对象（yelp）
    # 从Yelpchi.mat中提取数据（coo格式的邻接矩阵）
    net_rur = yelp['net_rur']  # net_rur={csc matrix:(45954,45954)}
    net_rtr = yelp['net_rtr']  # net_rtr={csc matrix:(45954,45954)}
    net_rsr = yelp['net_rsr']  # net_rsr={csc matrix:(45954,45954)}
    yelp_homo = yelp['homo']  # homo ={defaultdict: 45954}；yelp_homo存储关于 Yelp 数据集中的某些同质性信息
    # 将coo格式的邻接矩阵保存为adj_list
    sparse_to_adjlist(net_rur, os.path.join(
        DATADIR, "yelp_rur_adjlists.pickle"))
    sparse_to_adjlist(net_rtr, os.path.join(
        DATADIR, "yelp_rtr_adjlists.pickle"))
    sparse_to_adjlist(net_rtr, os.path.join(
        DATADIR, "yelp_rsr_adjlists.pickle"))
    sparse_to_adjlist(net_rtr, os.path.join(
        DATADIR, "yelp_homo_adjlists.pickle"))

    data_file = yelp
    labels = pd.DataFrame(data_file['label'].flatten())[0]
    feat_data = pd.DataFrame(data_file['features'].todense().A)  # ①todense()：这个方法将稀疏矩阵（sparse matrix）转换为密集矩阵（dense matrix）；②A：表示将矩阵转换为数组（array）
    # load the preprocessed adj_lists
    with open(os.path.join(DATADIR, "yelp_homo_adjlists.pickle"), 'rb') as file:
        homo = pickle.load(file)
    file.close()
    # print(homo)  # 思维导图中有具体数据截图
    src = []
    tgt = []
    for i in homo:
        for j in homo[i]:
            src.append(i)
            tgt.append(j)
    src = np.array(src)  # 将src的列表（list）格式转换为NumPy数组类型
    tgt = np.array(tgt)
    g = dgl.graph((src, tgt))  # 用dgl.graph()函数创建图对象g，其中src和tgt分别是源节点和目标节点的索引列表 ② 源码中说明``('coo', (Tensor, Tensor))``这里的Tensor可以被替换成任意可迭代的数据格式，但必须是同质的 (e.g. list, tuple,numpy.ndarray).
    g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
    g.ndata['feat'] = torch.from_numpy(
        feat_data.to_numpy()).to(torch.float32)
    dgl.data.utils.save_graphs(DATADIR + "graph-yelp.bin", [g])

    # %%
    """
        For Amazon dataset
    """
    print(f"processing AMAZON data...")
    amz = loadmat(os.path.join(DATADIR, 'Amazon.mat'))
    net_upu = amz['net_upu']#'net_upu': <11944x11944 sparse matrix of type '<class 'numpy.float64'>'with 351216 stored elements in Compressed Sparse Column format>,
    net_usu = amz['net_usu']#'net_usu': <11944x11944 sparse matrix of type '<class 'numpy.float64'>'with 7132958 stored elements in Compressed Sparse Column format>
    net_uvu = amz['net_uvu']#'net_uvu': <11944x11944 sparse matrix of type '<class 'numpy.float64'>'with 2073474 stored elements in Compressed Sparse Column format>
    amz_homo = amz['homo'] #'homo': <11944x11944 sparse matrix of type '<class 'numpy.float64'>'with 8796784 stored elements in Compressed Sparse Column format>
#'features': <11944x25 sparse matrix of type '<class 'numpy.float64'>'with 174488 stored elements in Compressed Sparse Column format>
#'label': array([[0., 0., 0., ..., 0., 0., 0.]])} (1, 11944)
    sparse_to_adjlist(net_upu, os.path.join(
        DATADIR, "amz_upu_adjlists.pickle"))
    sparse_to_adjlist(net_usu, os.path.join(
        DATADIR, "amz_usu_adjlists.pickle"))
    sparse_to_adjlist(net_uvu, os.path.join(
        DATADIR, "amz_uvu_adjlists.pickle"))
    sparse_to_adjlist(amz_homo, os.path.join(
        DATADIR, "amz_homo_adjlists.pickle"))

    data_file = amz
    labels = pd.DataFrame(data_file['label'].flatten())[0]
    feat_data = pd.DataFrame(data_file['features'].todense().A)
    # load the preprocessed adj_lists
    with open(DATADIR + 'amz_homo_adjlists.pickle', 'rb') as file:
        homo = pickle.load(file)
    file.close()
    src = []
    tgt = []
    print(homo)
    for i in homo:
        for j in homo[i]:
            src.append(i)
            tgt.append(j)
    src = np.array(src)
    tgt = np.array(tgt)
    g = dgl.graph((src, tgt))
    g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
    g.ndata['feat'] = torch.from_numpy(
        feat_data.to_numpy()).to(torch.float32)
    dgl.data.utils.save_graphs(DATADIR + "graph-amazon.bin", [g])

    # %%
    """
        For S-FFSD dataset
    """
    print(f"processing S-FFSD data...")
    data = pd.read_csv(os.path.join(DATADIR, 'S-FFSD.csv'))
    data = featmap_gen(data.reset_index(drop=True))
    data.replace(np.nan, 0, inplace=True)  # 将数据中所有 NaN 值替换为 0
    data.to_csv(os.path.join(DATADIR, 'S-FFSDneofull.csv'), index=None)
    data = pd.read_csv(os.path.join(DATADIR, 'S-FFSDneofull.csv'))

    data = data.reset_index(drop=True)
    out = []
    alls = []
    allt = []
    pair = ["Source", "Target", "Location", "Type"]
    for column in pair:  # 依次遍历特征列中的4列
        src, tgt = [], []
        edge_per_trans = 3
        for c_id, c_df in tqdm(data.groupby(column), desc=column):  # 按照当前遍历到的特征列进行分组，（如果按照source排序）c_id是当前source值，c_df是当前source值对应的所有行
            c_df = c_df.sort_values(by="Time")  # 再按时间进行排序
            df_len = len(c_df)  # 获取排序后当前分组的行数
            sorted_idxs = c_df.index  # 获取排序后当前分组的行 对应的索引
            src.extend([sorted_idxs[i] for i in range(df_len)
                        for j in range(edge_per_trans) if i + j < df_len])
            tgt.extend([sorted_idxs[i+j] for i in range(df_len)
                        for j in range(edge_per_trans) if i + j < df_len])
        alls.extend(src)
        allt.extend(tgt)
    alls = np.array(alls)
    allt = np.array(allt)
    g = dgl.graph((alls, allt))
    cal_list = ["Source", "Target", "Location", "Type"]
    for col in cal_list:
        le = LabelEncoder()  # LabelEncoder将每个类别映射到一个唯一的整数。将分类变量转换为整数编码，因为大多数模型无法直接处理字符串或分类数据
        data[col] = le.fit_transform(data[col].apply(str).values)  # ①（.apply(str)）将data[col]列中的所有值转换为字符串格式；②（.values）将data[col]列中的所有值从 DataFrame 对象转换为NumPy 数组格式
    feat_data = data.drop("Labels", axis=1)
    labels = data["Labels"]
    ###
    prefix = os.path.join(os.path.dirname(__file__), "..", "data/")
    feat_data.to_csv(prefix + "S-FFSD_feat_data.csv", index=None)
    labels.to_csv(prefix + "S-FFSD_label_data.csv", index=None)
    ###
    g.ndata['label'] = torch.from_numpy(
        labels.to_numpy()).to(torch.long)
    g.ndata['feat'] = torch.from_numpy(
        feat_data.to_numpy()).to(torch.float32)
    dgl.data.utils.save_graphs(DATADIR + "graph-S-FFSD.bin", [g])

    # generate neighbor riskstat features
    for file_name in ['S-FFSD', 'yelp', 'amazon']:
        print(
            f"Generating neighbor risk-aware features for {file_name} dataset...")
        graph = dgl.load_graphs(DATADIR + "graph-" + file_name + ".bin")[0][0]  # [0][0]：访问列表中第一个元组的第一个元素，即获取加载的第一个图对象
        graph: dgl.DGLGraph  # 这行注释声明了变量graph的数据类型为dgl.DGLGraph
        print(f"graph info: {graph}")

        edge_feat: torch.Tensor
        degree_feat = graph.in_degrees().unsqueeze_(1).float()  # ①graph.in_degrees(): 返回一个Tensor，其中每个元素表示对应节点的入度；②.unsqueeze_(1): 对于上述得到的入度Tensor，调用此方法将其在维度 1（即第二个维度）上增加一个长度为1的新轴。这相当于在原有的一维特征向量前添加了一个维度，将其转换为形状为 (节点数, 1) 的二维Tensor
        risk_feat = count_risk_neighs(graph).unsqueeze_(1).float()  # 得到所有节点具有风险标签的邻居节点数量，在其维度 1（即第二个维度）上增加一个长度为1的新轴。这相当于在原有的一维特征向量前添加了一个维度

        origin_feat_name = []
        edge_feat = torch.cat([degree_feat, risk_feat], dim=1)  # ① degree_feat表示所有节点的入度，和 risk_feat 表示所有节点具有风险标签的邻居节点数量；② dim=1  沿着第一个维度（列维度）拼接在一起，edge_feat 的 shape 是 (N, 2)
        origin_feat_name = ['degree', 'riskstat']

        features_neigh, feat_names = feat_map()  # 获取了所有节点的所有一跳和二跳前置邻居节点索引，及其 度 和 风险数量 的总和，features_neigh的shape是(N, 4)
        # print(f"feature neigh: {features_neigh.shape}")

        features_neigh = torch.cat(
            (edge_feat, features_neigh), dim=1
        ).numpy()  # features_neigh的shape 是 (N, 6)，edge_feat 是所有节点的入度和风险数量，features_neigh 是所有节点的所有一跳和二跳前置邻居节点索引，及其 度 和 风险数量 的总和
        feat_names = origin_feat_name + feat_names  # ['degree', 'riskstat'] + ['1hop_degree', '2hop_degree', '1hop_riskstat', '2hop_riskstat']
        features_neigh[np.isnan(features_neigh)] = 0.  # 将features_neigh中的所有 NaN 值替换为 0

        output_path = DATADIR + file_name + "_neigh_feat.csv"  # （N，6），列名是['degree', 'riskstat', '1hop_degree', '2hop_degree', '1hop_riskstat', '2hop_riskstat']
        features_neigh = pd.DataFrame(features_neigh, columns=feat_names)  # columns=feat_names是指定列名
        scaler = StandardScaler()
        # features_neigh = np.log(features_neigh + 1)
        features_neigh = pd.DataFrame(scaler.fit_transform(
            features_neigh), columns=features_neigh.columns)  # 对features_neigh进行标准化处理，将新DataFrame对象的列名设置为与原始数据features_neigh的列名相同

        features_neigh.to_csv(output_path, index=False)
