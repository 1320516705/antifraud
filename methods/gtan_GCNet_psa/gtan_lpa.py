import copy

# 调用时参数：load_lpa_subtensor(num_feat, cat_feat, labels,seeds, input_nodes, device)
# node_feat ={Tensor:(77881,126)};
# work_node_feat = {dict: 3} {'Target': tensor([ 0,1,0,...,68,37,108]),'Location':tensor([0, 1,0,...,0,0,10]),'Type':tensor([0, 1,0, ...,7,26,14])}
# labels = {Tensor:(77881,)};
# seeds ={Tensor:(128,)};
# input nodes = {Tensor: (2321,)};
def load_lpa_subtensor(node_feat, work_node_feat, labels, seeds, input_nodes, device):  # 此函数通过筛选和预处理，为图上的节点特征和标签数据创建了一个子集，适合于进行局部的标签传播算法迭代或图神经网络的小批量训练
    batch_inputs = node_feat[input_nodes].to(device)  # batch inputs ={Tensor:(2321,126)}
    batch_work_inputs = {i: work_node_feat[i][input_nodes].to(  # 特定特征集合（由键 i 定义）中选择了与 input_nodes 列表相匹配的行，形成一个新的张量
        device) for i in work_node_feat if i not in {"Labels"}}  # batch work inputs = {dict: 3} {'Target': tensor([238,8,0,...,15,0,0]),'Location':tensor([2,0,0,.12,12,0]),'Type': tensor([33,5,0,...,10, 0, 0])}
    # for i in batch_work_inputs:
    #    print(batch_work_inputs[i].shape)
    batch_labels = labels[seeds].to(device)  # batch labels = {Tensor: (128,)}
    train_labels = copy.deepcopy(labels)  # train_labels = {Tensor:(77881,)}
    propagate_labels = train_labels[input_nodes]  # propagate labels ={Tensor:(2321,)}
    propagate_labels[:seeds.shape[0]] = 2  # seeds = {Tensor:(128,)},表明将前128个节点的标签设置为2
    return batch_inputs, batch_work_inputs, batch_labels, propagate_labels.to(device)
