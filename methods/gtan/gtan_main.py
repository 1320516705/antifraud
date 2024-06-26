import numpy as np
import dgl
import torch
import os
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
import torch.optim as optim
from scipy.io import loadmat
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from dgl.dataloading import MultiLayerFullNeighborSampler
# from dgl.dataloading import NodeDataLoader
from dgl.dataloading import DataLoader as NodeDataLoader
from torch.optim.lr_scheduler import MultiStepLR
from .gtan_model import GraphAttnModel
from . import *
from methods.gtan.my_add import combine_local_and_global_features
from methods.gtan.my_add import *

# gtan_main(feat_data, g, train_idx, test_idx, labels, args, cat_features)
def gtan_main(feat_df, graph, train_idx, test_idx, labels, args, cat_features):  # feat_df = ={DataFrame:(77881,126)是节点特征，labels = {Tensor:(77881,)}是标签，train_idx = {list: 62304}是训练集索引，test_idx = {list: 15577}是测试集索引，g是图，cat_features = {list: 3} ['Target', 'Location', 'Type' ]
    device = args['device']
    graph = graph.to(device)
    oof_predictions = torch.from_numpy(  # oof_predictions用于存储“out-of-fold”（交叉验证过程中的训练集预测）的预测结果。oof_predictions
        np.zeros([len(feat_df), 2])).float().to(device)  # oof_predictions ={Tensor:(77881,2)}。2 是因为每个样本有两个预测值
    test_predictions = torch.from_numpy(
        np.zeros([len(feat_df), 2])).float().to(device)  # test_predictions ={Tensor:(77881,2)}。test_predictions用于存储测试集的预测结果
    kfold = StratifiedKFold(  # 创建一个StratifiedKFold对象，它是一种交叉验证的分割方法
        n_splits=args['n_fold'], shuffle=True, random_state=args['seed'])
#  feat_df中的数值型特征（num_feat）和类别型特征（cat_feat）分别转换为PyTorch张量，其实num_feat中包含cat_feat中的特征
    y_target = labels.iloc[train_idx].values  # y_target={ndarray:(62304,)},获得训练集的标签值
    num_feat = torch.from_numpy(feat_df.values).float().to(device)   # num_feat ={Tensor:(77881,126)}。是将feat_df的数值数据（通过.values属性获取）从NumPy数组形式转换为PyTorch张量，feat df ={DataFrame:(77881,126)}
    cat_feat = {col: torch.from_numpy(feat_df[col].values).long().to(  # cat_feat是对feat_df中特定类别特征列（由cat_features列表指定）的子集进行特定转换的字典
        device) for col in cat_features}  # cat feat = {dict: 3} {'Target': tensor([ 0,1,0,...,68,37,108]),'Location':tensor([0, 1,0,...,0,0,10]),'Type':tensor([0, 1,0, ...,7,26,14])}

    y = labels  # y= {Series: (77881,)}
    labels = torch.from_numpy(y.values).long().to(device)  # labels = {Tensor:(77881,)}
    loss_fn = nn.CrossEntropyLoss().to(device)
    # my_model = GCNLayer(in_feats=feat_df.shape[1], out_feats=feat_df.shape[1])
    # new_num_feat = my_model(graph, num_feat)
    # num_feat = num_feat + new_num_feat;

    for fold, (trn_idx, val_idx) in enumerate(kfold.split(feat_df.iloc[train_idx], y_target)):  # trn idx = {ndarray:(49843,)},val idx = ndarray:(12461,)}。①enumerate()是Python内置函数，它将一个可迭代对象（如列表、元组等）组合为一个枚举对象；②返回该折叠的索引fold和对于每折的训练和验证的样本索引trn idx = {ndarray:(49843,)}和val_idx
        print(f'Training fold {fold + 1}')
        trn_ind, val_ind = torch.from_numpy(np.array(train_idx)[trn_idx]).long().to(  # 转成张量形式的索引，本质上还是索引
            device), torch.from_numpy(np.array(train_idx)[val_idx]).long().to(device)  # trn_idx = {ndarray: (49843,)};val_idx={ndarray: (12461,)};trn_ind ={Tensor:(49843,)};val_ind = {Tensor:(12461,)}

        train_sampler = MultiLayerFullNeighborSampler(args['n_layers'])  # 采样器将使每个节点从每个边类型的每个邻居收集消息。
        # train_dataloader = dgl.dataloading.NodeDataLoader(graph,
        train_dataloader = NodeDataLoader(graph,  # 将一个 DGLGraph 和一个采样器打包成一个可迭代的迷你样本批次。
                                          trn_ind,
                                          train_sampler,
                                          device=device,
                                          use_ddp=False,  # 是否使用分布式数据并行。
                                          batch_size=args['batch_size'],  # batch_size: 128
                                          shuffle=True,
                                          drop_last=False,  # 是否丢弃最后一个不完整的批次。
                                          num_workers=0  # 用于数据加载的工作进程数。
                                          )
        val_sampler = MultiLayerFullNeighborSampler(args['n_layers'])  # 这段和上面的train_sampler和val_sampler的代码结构类似
        # val_dataloader = dgl.dataloading.NodeDataLoader(graph,
        val_dataloader = NodeDataLoader(graph,
                                        val_ind,
                                        val_sampler,
                                        use_ddp=False,
                                        device=device,
                                        batch_size=args['batch_size'],
                                        shuffle=True,
                                        drop_last=False,
                                        num_workers=0,
                                        )
        # TODO
        model = GraphAttnModel(in_feats=feat_df.shape[1],  # in_feats：126
                               # 为什么要整除4？
                               hidden_dim=args['hid_dim']//4,  # hid_dim: 256：每个注意力头的输入维度为原始维度除4之后的维度，使每个注意力头关注不同方面的特征
                               n_classes=2,
                               heads=[4]*args['n_layers'],  # [4,4]运算符*对列表进行乘法操作,表示将列表 [4] 重复指定次数args['n_layers']，得到新的列表
                               activation=nn.PReLU(),
                               n_layers=args['n_layers'],  # n_layers: 2
                               drop=args['dropout'],  # dropout: [0.2, 0.1]
                               device=device,
                               gated=args['gated'],  # gated: True
                               ref_df=feat_df.iloc[train_idx],  # ref_df={DataFrame: (62304,126)}。是由data_process.py中def featmap_gen(tmp_df=None):处理（增加列）后的数据
                               cat_features=cat_feat).to(device)  # cat_features = {dict: 3}
        print(model)
        lr = args['lr'] * np.sqrt(args['batch_size']/1024)  # 0.00075：学习率*(batch_size: 128除以1024)的平方根。是为了在更大的batch_size时减小学习率，以避免过拟合
        optimizer = optim.Adam(model.parameters(), lr=lr,
                               weight_decay=args['wd'])  # wd: !!float 1e-4
        lr_scheduler = MultiStepLR(optimizer=optimizer, milestones=[  # 学习率调度器，它通过在给定的里程碑（milestones）处将学习率乘以gamma值来调整优化器的学习率，在这个例子中，学习率会在epoch 4000和12000处分别乘以0.3
                                   4000, 12000], gamma=0.3)

        earlystoper = early_stopper(  # patience参数指定了早停的耐心值，即连续多少次验证集损失没有改善就停止训练；verbose参数用于控制输出的详细程度。
            patience=args['early_stopping'], verbose=True)
        start_epoch, max_epochs = 0, 2000
        # epoch的训练数据是经历（训练集测试集划分 取训练集 -> k折交叉验证取的其中一折 -> 分batch -> 遍历epoch(遍历分batch的数据，共step组)）
        for epoch in range(start_epoch, args['max_epochs']):
            train_loss_list = []


            # 将模型设置为训练模式,而不是直接运行到 GraphAttnModel 的 forward 函数中
            model.train()

            # 对每个批次数据进行迭代，step 是当前批次的索引，(input_nodes, seeds, blocks) 是当前批次的数据。
            # input_nodes：当前批次所需的所有节点，包含了种子节点及其邻居。{Tensor: (2321,)}
            # seeds：当前批次的种子节点:种子节点是当前批次中需要计算损失的目标节点.{Tensor:(128,)}
            # blocks：当前批次的数据块（通常表示从图中采样得到的子图）。每个 block 的层数由train_sampler定义时的args['n_layers']决定的。{ list:2[Block(num src nodes=2321, num dst nodes=681, num edges=7506), Block(num src nodes=681, num dst nodes=128.,num edges=1408)]
            # 数据加载器train_dataloader在初始化时已经根据 batch_size、shuffle 等参数生成了所有批次的数据，所以需要遍历
            # step就是训练集被分成了几个batch的意思
            for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):  # input nodes = {Tensor: (2321,)};seeds ={Tensor:(128,)};blocks =  { list:2[Block(num src nodes=2321, num dst nodes=681, num edges=7506), Block(num src nodes=681, num dst nodes=128.,num edges=1408)]
                # batch_inputs: {Tensor:(2321,126)}，一个batch的“输入节点”的“数字特征”
                # batch_work_inputs = {dict: 3}，与“数字特征”索引对应的“类别特征”
                # lpa_labels={Tensor:(2321,)}，一个batch的“输入节点”对应的“标签”，但是他前128个置为2了

                # batch_labels = {Tensor: (128,)}，一个batch的“种子节点”对应的“标签”
                batch_inputs, batch_work_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat, labels,  # 此函数通过筛选和预处理，为图上的节点特征和标签数据创建了一个子集，适合于进行局部的标签传播算法迭代或图神经网络的小批量训练
                                                                                               seeds, input_nodes, device)
                centrality(graph)
                # 获取全局位置编码
                global_pos_enc_batch = graph.ndata['centrality'][input_nodes]
                # print("batch_inputs.shape********",batch_inputs.shape)
                # print("global_pos_enc_batch.shape********",global_pos_enc_batch.shape)
                combined_inputs = combine_local_and_global_features(batch_inputs, global_pos_enc_batch)
                # print("combined_inputs.shape********",combined_inputs.shape)


                # 使用两跳的邻居采样方法，每一层需要一个子图来表示这些邻居关系，因此每个blocks会有两个 block
                blocks = [block.to(device) for block in blocks]  # blocks =  { list:2[Block(num src nodes=2321, num dst nodes=681, num edges=7506), Block(num src nodes=681, num dst nodes=128.,num edges=1408)]
                # train_batch_logits = model(  # train_batch_logits = {Tensor: (128,2)} 是预测值
                #     blocks, batch_inputs, lpa_labels, batch_work_inputs)  # ①batch inputs ={Tensor:(2321,126)}；②lpa_labels ={Tensor:(2321,)}③batch work inputs = {dict: 3} {'Target': tensor([238,8,0,...,15,0,0]),'Location':tensor([2,0,0,.12,12,0]),'Type': tensor([33,5,0,...,10, 0, 0])}
                # # 标签值 2 表示无效或不需要处理的标签
                train_batch_logits = model(blocks, combined_inputs, lpa_labels, batch_work_inputs)
                # train_batch_logits = model(
                #     blocks, batch_inputs, lpa_labels, batch_work_inputs)
                mask = batch_labels == 2  # 创建一个掩码（mask），其作用是将【种子节点标签】中所有值等于2的元素标识出来

                # 预测值【train_batch_logits】和真实值【batch_labels】
                train_batch_logits = train_batch_logits[~mask]  # train_batch_logits = {Tensor: (128,2)} -> train_batch_logits = {Tensor: (59,2)};~mask：这是对mask取反的操作
                batch_labels = batch_labels[~mask]  # batch labels = {Tensor: (128,)} -> batch labels = {Tensor: (59,)}
                # batch_labels[mask] = 0

                train_loss = loss_fn(train_batch_logits, batch_labels)  # train loss ={Tensor:()}
                # backward
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()  # 调用optimizer对象的step()方法来更新模型的参数
                lr_scheduler.step()  # 根据事先定义的学习率调整策略（如每n个epoch降低学习率）来更新当前的学习率值
                train_loss_list.append(train_loss.cpu().detach().numpy())

                if step % 10 == 0:
                    tr_batch_pred = torch.sum(torch.argmax(train_batch_logits.clone(
                    ).detach(), dim=1) == batch_labels) / batch_labels.shape[0]
                    score = torch.softmax(train_batch_logits.clone().detach(), dim=1)[
                        :, 1].cpu().numpy()

                    # if (len(np.unique(score)) == 1):
                    #     print("all same prediction!")
                    try:
                        print('In epoch:{:03d}|batch:{:04d}, train_loss:{:4f}, '
                              'train_ap:{:.4f}, train_acc:{:.4f}, train_auc:{:.4f}'.format(epoch, step,
                                                                                           np.mean(
                                                                                               train_loss_list),
                                                                                           average_precision_score(
                                                                                               batch_labels.cpu().numpy(), score),
                                                                                           tr_batch_pred.detach(),
                                                                                           roc_auc_score(batch_labels.cpu().numpy(), score)))
                    except:
                        pass

            # mini-batch for validation
            val_loss_list = 0
            val_acc_list = 0
            val_all_list = 0
            model.eval()
            with torch.no_grad():
                for step, (input_nodes, seeds, blocks) in enumerate(val_dataloader):
                    batch_inputs, batch_work_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat, labels,
                                                                                                   seeds, input_nodes, device)

                    blocks = [block.to(device) for block in blocks]
                    val_batch_logits = model(
                        blocks, batch_inputs, lpa_labels, batch_work_inputs)
                    oof_predictions[seeds] = val_batch_logits
                    mask = batch_labels == 2
                    val_batch_logits = val_batch_logits[~mask]
                    batch_labels = batch_labels[~mask]
                    # batch_labels[mask] = 0
                    val_loss_list = val_loss_list + \
                        loss_fn(val_batch_logits, batch_labels)
                    # val_all_list += 1
                    val_batch_pred = torch.sum(torch.argmax(
                        val_batch_logits, dim=1) == batch_labels) / torch.tensor(batch_labels.shape[0])
                    val_acc_list = val_acc_list + val_batch_pred * \
                        torch.tensor(batch_labels.shape[0])
                    val_all_list = val_all_list + batch_labels.shape[0]
                    if step % 10 == 0:
                        score = torch.softmax(val_batch_logits.clone().detach(), dim=1)[
                            :, 1].cpu().numpy()
                        try:
                            print('In epoch:{:03d}|batch:{:04d}, val_loss:{:4f}, val_ap:{:.4f}, '
                                  'val_acc:{:.4f}, val_auc:{:.4f}'.format(epoch,
                                                                          step,
                                                                          val_loss_list/val_all_list,
                                                                          average_precision_score(
                                                                              batch_labels.cpu().numpy(), score),
                                                                          val_batch_pred.detach(),
                                                                          roc_auc_score(batch_labels.cpu().numpy(), score)))
                        except:
                            pass

            # val_acc_list/val_all_list, model)
            earlystoper.earlystop(val_loss_list/val_all_list, model)
            if earlystoper.is_earlystop:
                print("Early Stopping!")
                break
        print("Best val_loss is: {:.7f}".format(earlystoper.best_cv))
        test_ind = torch.from_numpy(np.array(test_idx)).long().to(device)
        test_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        # test_dataloader = dgl.dataloading.NodeDataLoader(graph,
        test_dataloader = NodeDataLoader(graph,
                                         test_ind,
                                         test_sampler,
                                         use_ddp=False,
                                         device=device,
                                         batch_size=args['batch_size'],
                                         shuffle=True,
                                         drop_last=False,
                                         num_workers=0,
                                         )
        b_model = earlystoper.best_model.to(device)
        b_model.eval()
        with torch.no_grad():
            for step, (input_nodes, seeds, blocks) in enumerate(test_dataloader):
                # print(input_nodes)
                batch_inputs, batch_work_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat, labels,
                                                                                               seeds, input_nodes, device)

                blocks = [block.to(device) for block in blocks]
                test_batch_logits = b_model(
                    blocks, batch_inputs, lpa_labels, batch_work_inputs)
                test_predictions[seeds] = test_batch_logits
                test_batch_pred = torch.sum(torch.argmax(
                    test_batch_logits, dim=1) == batch_labels) / torch.tensor(batch_labels.shape[0])
                if step % 10 == 0:
                    print('In test batch:{:04d}'.format(step))
    mask = y_target == 2
    y_target[mask] = 0
    my_ap = average_precision_score(y_target, torch.softmax(
        oof_predictions, dim=1).cpu()[train_idx, 1])
    print("NN out of fold AP is:", my_ap)
    b_models, val_gnn_0, test_gnn_0 = earlystoper.best_model.to(
        'cpu'), oof_predictions, test_predictions

    test_score = torch.softmax(test_gnn_0, dim=1)[test_idx, 1].cpu().numpy()
    y_target = labels[test_idx].cpu().numpy()
    test_score1 = torch.argmax(test_gnn_0, dim=1)[test_idx].cpu().numpy()

    mask = y_target != 2
    test_score = test_score[mask]
    y_target = y_target[mask]
    test_score1 = test_score1[mask]

    print("test AUC:", roc_auc_score(y_target, test_score))
    print("test f1:", f1_score(y_target, test_score1, average="macro"))
    print("test AP:", average_precision_score(y_target, test_score))

# 调用代码：feat_data, labels, train_idx, test_idx, g, cat_features = load_gtan_data(args['dataset'], args['test_size'])
def load_gtan_data(dataset: str, test_size: float):
    """
    Load graph, feature, and label given dataset name
    :param dataset: the dataset name
    :param test_size: the size of test set
    :returns: feature, label, graph, category features
    """
    # prefix = './antifraud/data/'
    prefix = os.path.join(os.path.dirname(__file__), "..", "..", "data/")
    if dataset == "S-FFSD":
        cat_features = ["Target", "Location", "Type"]  # 这几列选作类别特征

        # 1、读数据
        df = pd.read_csv(prefix + "S-FFSDneofull.csv")  # S-FFSDneofull.csv是加了不同时间窗口的对应的基础值列的data
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]  # 使用正则表达式判断列名是否包含'Unnamed'，返回一个布尔型的Series，其中True表示列名不包含'Unnamed'，False表示列名包含'Unnamed'
        data = df[df["Labels"] <= 2]  # 筛选df数据框中Labels列值小于等于2的行
        data = data.reset_index(drop=True)

        # 2、构建graph
        out = []
        alls = []
        allt = []
        pair = ["Source", "Target", "Location", "Type"]
        for column in pair:
            src, tgt = [], []
            edge_per_trans = 3
            for c_id, c_df in data.groupby(column):  # 当column为"Source"时，c_id是每个分组的"Source"列的具体值
                c_df = c_df.sort_values(by="Time")  # 按照"Time"这一列的值从小到大排序
                df_len = len(c_df)  # c_df是2列数据框，df_len是c_df的行数
                sorted_idxs = c_df.index
                src.extend([sorted_idxs[i] for i in range(df_len)
                            for j in range(edge_per_trans) if i + j < df_len])
                tgt.extend([sorted_idxs[i+j] for i in range(df_len)
                            for j in range(edge_per_trans) if i + j < df_len])
            alls.extend(src)
            allt.extend(tgt)
        alls = np.array(alls)
        allt = np.array(allt)
        g = dgl.graph((alls, allt))

        # 3、将【类别特征】做编码映射处理，也是变相的更新data
        cal_list = ["Source", "Target", "Location", "Type"]# 对这几列的数据做编码映射处理
        for col in cal_list:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].apply(str).values)
        feat_data = data.drop("Labels", axis=1)
        labels = data["Labels"]
        ###
        feat_data.to_csv(prefix + "S-FFSD_feat_data.csv", index=None)
        labels.to_csv(prefix + "S-FFSD_label_data.csv", index=None)
        ###
        index = list(range(len(labels)))
        g.ndata['label'] = torch.from_numpy(
            labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(
            feat_data.to_numpy()).to(torch.float32)
        graph_path = prefix+"graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])
        from dgl.data.utils import load_graphs
        graph_list, _ = load_graphs(prefix + "graph-S-FFSD.bin")
        g = graph_list[0]  # 因为只保存了一个图，所以从列表中获取第一个图

        train_idx, test_idx, y_train, y_test = train_test_split(index, labels, stratify=labels, test_size=test_size/2,
                                                                random_state=2, shuffle=True)  # train_idx= {list: 62304},test_idx = {list: 15577},y_train = {Series: (62304,)},y_test = {Series: (15577,)}

    elif dataset == "yelp":
        cat_features = []
        data_file = loadmat(prefix + 'YelpChi.mat')
        labels = pd.DataFrame(data_file['label'].flatten())[0]
        feat_data = pd.DataFrame(data_file['features'].todense().A)
        # load the preprocessed adj_lists
        with open(prefix + 'yelp_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        file.close()
        index = list(range(len(labels)))
        train_idx, test_idx, y_train, y_test = train_test_split(index, labels, stratify=labels, test_size=test_size,
                                                                random_state=2, shuffle=True)
        src = []
        tgt = []
        for i in homo:
            for j in homo[i]:
                src.append(i)  # src是出发点
                tgt.append(j)  # tgt是被指向点
        src = np.array(src)
        tgt = np.array(tgt)
        g = dgl.graph((src, tgt))
        g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(
            feat_data.to_numpy()).to(torch.float32)
        graph_path = prefix + "graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])

    elif dataset == "amazon":
        cat_features = []
        data_file = loadmat(prefix + 'Amazon.mat')
        labels = pd.DataFrame(data_file['label'].flatten())[0]
        feat_data = pd.DataFrame(data_file['features'].todense().A)
        # load the preprocessed adj_lists
        with open(prefix + 'amz_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        file.close()
        index = list(range(3305, len(labels)))
        train_idx, test_idx, y_train, y_test = train_test_split(index, labels[3305:], stratify=labels[3305:],
                                                                test_size=test_size, random_state=2, shuffle=True)
        src = []
        tgt = []
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
        graph_path = prefix + "graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])

    return feat_data, labels, train_idx, test_idx, g, cat_features  # cat_features = {list: 3} ['Target', 'Location', 'Type' ]
