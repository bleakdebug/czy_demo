import torch
import torch.nn.functional as F
import torch.nn as nn


class SAAR_Loss(nn.Model):
    def __init__(self, args, model):
        super(PASSLEAF_Loss, self).__init__()
        self.args = args
        self.model = model

    def forward(self, graph, edge, triples):
        # bert模型
        graph_1 = {}
        for entity, neighbors in graph.items():
            if len(neighbors) <= 512:
                graph_1[entity] = self.Bert_graph(neighbors)
            else:
                # 超过512的序列要使用random.sample进行随机截取
                random_sample = random.sample(neighbors, 512)
                graph_1[entity] = self.Bert_graph(random_sample)
        # print(type(graph_1))
        # np.save('graph_dict.npy', graph_1)  # 保存

        # graph= np.load('graph_dict.npy', allow_pickle=True, encoding='bytes').tolist()
        y = list(graph_1.values())
        # print(y)
        # print(list(graph_1.values())[0])

        x = torch.tensor(y, dtype=torch.float32)  # 示例节点特征
        data = Data(x=x, edge_index=edge)
        x, edge_index = data.x, data.edge_index

        # 第一层GCN卷积
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # 第二层GCN卷积
        x = self.conv2(x, edge_index)
        # print(x)

        # 前向传播
        for entity in graph_1.keys():
            graph_1[entity] = x[entity]

        # np.save('graph_update.npy', graph)  # 保存

        # 构建图
        pre_score = []
        for h_id, r_id, t_id in triples:
            h = graph_1[h_id]
            r = torch.tensor(np.random.rand(60), dtype=torch.float32)
            t = graph_1[t_id]

            # 拼接向量
            concatenated_vector = torch.cat([h, r, t], dim=-1)
            # print(concatenated_vector.shape)

            input_sequence = concatenated_vector.view(1, -1)
            # print(input_sequence.dtype)
            # print(input_sequence.size(-1))
            # 前向传播

            # lstm = nn.LSTM(input_size=input_sequence.size(-1), hidden_size=128, num_layers=1, batch_first=True)
            output, (hidden_state, cell_state) = self.lstm(input_sequence)
            # print(output)

            # 输出更新后的向量
            score_distance = torch.sum(output)
            # print( score_distance)
            score_weight = torch.sigmoid(score_distance)
            pre_score.append(score_weight.item())
        pre_score = torch.tensor(pre_score, requires_grad=True)
        return pre_score