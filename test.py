# for fold, (trn_idx, val_idx) in enumerate(kfold.split(feat_df.iloc[train_idx], y_target)):
#     output_file.write(f'Training fold {fold + 1}\n')
#     trn_ind, val_ind = torch.from_numpy(np.array(train_idx)[trn_idx]).long().to(
#         device), torch.from_numpy(np.array(train_idx)[val_idx]).long().to(device)
#
#     train_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
#     train_dataloader = NodeDataLoader(graph,
#                                       trn_ind,
#                                       train_sampler,
#                                       device=device,
#                                       use_ddp=False,
#                                       batch_size=args['batch_size'],
#                                       shuffle=True,
#                                       drop_last=False,
#                                       num_workers=0
#                                       )
#     val_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
#     val_dataloader = NodeDataLoader(graph,
#                                     val_ind,
#                                     val_sampler,
#                                     use_ddp=False,
#                                     device=device,
#                                     batch_size=args['batch_size'],
#                                     shuffle=True,
#                                     drop_last=False,
#                                     num_workers=0,
#                                     )
#     # TODO
#     model = GraphAttnModel(in_feats=feat_df.shape[1],
#                            # 为什么要整除4？
#                            hidden_dim=args['hid_dim'] // 4,
#                            n_classes=2,
#                            heads=[4] * args['n_layers'],  # [4,4,4]
#                            activation=nn.PReLU(),
#                            n_layers=args['n_layers'],
#                            drop=args['dropout'],
#                            device=device,
#                            gated=args['gated'],
#                            ref_df=feat_df.iloc[train_idx],
#                            cat_features=cat_feat).to(device)
#     lr = args['lr'] * np.sqrt(args['batch_size'] / 1024)  # 0.00075
#     optimizer = optim.Adam(model.parameters(), lr=lr,
#                            weight_decay=args['wd'])
#     lr_scheduler = MultiStepLR(optimizer=optimizer, milestones=[
#         4000, 12000], gamma=0.3)
#
#     earlystoper = early_stopper(
#         patience=args['early_stopping'], verbose=True)
#     start_epoch, max_epochs = 0, 2000
#     for epoch in tqdm(range(start_epoch, args['max_epochs'])):
#         train_loss_list = []
#         train_acc_list = []
#         model.train()
#         for step, (input_nodes, seeds, blocks) in enumerate(
#                 tqdm(train_dataloader, desc='Training Batches', leave=False)):
#             batch_inputs, batch_work_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat, labels,
#                                                                                            seeds, input_nodes, device)
#             blocks = [block.to(device) for block in blocks]
#             train_batch_logits = model(
#                 blocks, batch_inputs, lpa_labels, batch_work_inputs)
#             mask = batch_labels == 2
#             train_batch_logits = train_batch_logits[~mask]
#             batch_labels = batch_labels[~mask]
#
#             train_loss = loss_fn(train_batch_logits, batch_labels)
#             # backward
#             optimizer.zero_grad()
#             train_loss.backward()
#             optimizer.step()
#             lr_scheduler.step()
#             train_loss_list.append(train_loss.cpu().detach().numpy())
#
#             # 新增
#             train_batch_pred = torch.sum(torch.argmax(train_batch_logits.clone().detach(), dim=1) == batch_labels) / \
#                                batch_labels.shape[0]
#             train_acc_list.append(train_batch_pred.detach().numpy())
#
#             if step % 10 == 0:
#                 tr_batch_pred = torch.sum(torch.argmax(train_batch_logits.clone(
#                 ).detach(), dim=1) == batch_labels) / batch_labels.shape[0]
#                 score = torch.softmax(train_batch_logits.clone().detach(), dim=1)[
#                         :, 1].cpu().numpy()
#
#
#
#         # 记录训练损失和准确率
#         train_losses.append(np.mean(train_loss_list))
#         train_accuracies.append(np.mean(train_acc_list))
#
#         # 以下省略验证集验证，验证集过程和训练集过程类似
#
# mask = y_target == 2
#     y_target[mask] = 0
#     my_ap = average_precision_score(y_target, torch.softmax(
#         oof_predictions, dim=1).cpu()[train_idx, 1])
#     # print("NN out of fold AP is:", my_ap)
#     output_file.write("NN out of fold AP is: {}\n".format(my_ap))
#     b_models, val_gnn_0, test_gnn_0 = earlystoper.best_model.to(
#         'cpu'), oof_predictions, test_predictions
#
#     test_score = torch.softmax(test_gnn_0, dim=1)[test_idx, 1].cpu().numpy()
#     y_target = labels[test_idx].cpu().numpy()
#     test_score1 = torch.argmax(test_gnn_0, dim=1)[test_idx].cpu().numpy()
#
#     mask = y_target != 2
#     test_score = test_score[mask]
#     y_target = y_target[mask]
#     test_score1 = test_score1[mask]
#
#     output_file.write("test AUC:{}\n".format(roc_auc_score(y_target, test_score)))
#     output_file.write("test f1:{}\n".format(f1_score(y_target, test_score1, average="macro")))
#     output_file.write("test AP:{}\n".format(average_precision_score(y_target, test_score)))


# h_src = feat
# h_dst = h_src[:graph.number_of_dst_nodes()]
# # Step 0. q, k, v
# q_src = self.lin_query(
#     h_src).view(-1, self._num_heads, self._out_feats)
# k_dst = self.lin_key(h_dst).view(-1, self._num_heads, self._out_feats)
# v_src = self.lin_value(
#     h_src).view(-1, self._num_heads, self._out_feats)
# # Assign features to nodes
# graph.srcdata.update({'ft': q_src, 'ft_v': v_src})
# graph.dstdata.update({'ft': k_dst})
# # Step 1. dot product
# graph.apply_edges(fn.u_dot_v('ft', 'ft', 'a'))
#
# # Step 2. edge softmax to compute attention scores
# graph.edata['sa'] = edge_softmax(
#     graph, graph.edata['a'] / self._out_feats ** 0.5)
# # Step 3. Broadcast softmax value to each edge, and aggregate dst node
# graph.update_all(fn.u_mul_e('ft_v', 'sa', 'attn'),
#                  fn.sum('attn', 'agg_u'))
# # output results to the destination nodes
# rst = graph.dstdata['agg_u'].reshape(-1, self._out_feats * self._num_heads)
# # self.skip_feat = nn.Linear(self._in_src_feats, self._out_feats*self._num_heads, bias=bias)
# skip_feat = self.skip_feat(feat[:graph.number_of_dst_nodes()])
# #  self.gate = nn.Linear(3*self._out_feats*self._num_heads, 1, bias=bias)
# gate = torch.sigmoid(self.gate(torch.concat([skip_feat, rst, skip_feat - rst], dim=-1)))
# rst = gate * skip_feat + (1 - gate) * rst
# # self.layer_norm = nn.LayerNorm(self._out_feats*self._num_heads)  # 64*4=256
# rst = self.layer_norm(rst)
# rst = self.activation(rst)
# return rst
