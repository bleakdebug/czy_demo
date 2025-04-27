from collections import defaultdict
from transformers import BertTokenizer, BertModel
import torch
#import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import torch.optim as optim
from .model import Model


class SAAR(Model):
    def __init__(self, num_nodes, input_dim, hidden_dim, output_dim, input_size, hidden_size):
        # 超参数
        super(SAAR, self).__init__()

        # GNN参数
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # LSTM参数
        self.input_size = input_size,
        self.hidden_size = hidden_size,
        # self.num_layers=num_layers,

        # bert模型配置
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # 或者选择其他预训练的BERT模型
        self.BertModel = BertModel.from_pretrained('bert-base-uncased')

        # GNN模型配置
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

        # LSTM模型配置
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    "2.Bert模型"

    # 加载BERT tokenizer和模型

    def Bert_graph(self, sentence):
        # 使用tokenizer将句子转换为token IDs
        # input_ids = tokenizer.encode(sentence, add_special_tokens=True)
        # 将token IDs转换为PyTorch张量
        if len(sentence) == 1:
            sentence.append(0)
            input_ids_tensor = torch.tensor([sentence])
        else:
            input_ids_tensor = torch.tensor([sentence])

        # 获取BERT模型的输出
        with torch.no_grad():
            outputs = self.BertModel(input_ids_tensor)

        # 获取CLS标记对应的隐藏状态
        cls_embedding = outputs.pooler_output

        # 将PyTorch张量转换为NumPy数组
        cls_embedding_np = cls_embedding.numpy()[0]
        return (cls_embedding_np)


    '''
    def get_score(self, graph, edge, triples):
        """
        测试阶段使用的函数，计算三元组的得分。

        Args:
            graph: 图数据，包含节点及其邻居信息。
            edge: 图的边索引。
            triples: 三元组数据，形状为 [batch_size, 3]。

        Returns:
            score: 三元组的得分。
        """
        # 调用 forward 函数计算得分
        with torch.no_grad():  # 测试阶段不需要计算梯度
            score = self.forward(graph, edge, triples)
        return score
    '''

    def score_func(self, head_emb, relation_emb, tail_emb, mode):
        """Calculating the score of triples.

        The formula for calculating the score is :math:`h^{\top} \operatorname{diag}(r) t`

        Args:
            head_emb: The head entity embedding.
            relation_emb: The relation embedding.
            tail_emb: The tail entity embedding.
            mode: Choose head-predict or tail-predict.

        Returns:
            score: The score of triples.
        """
        if mode == 'head-batch':
            score = head_emb * (relation_emb * tail_emb)
        else:
            score = (head_emb * relation_emb) * tail_emb

        score = score.sum(dim=-1)
        score = score.to(self.args.gpu)

        """ 1.Bounded rectifier """
        # shape = score.shape
        # tmp_max = torch.max(self.w * score + self.b, torch.zeros(shape, device=self.args.gpu))
        # score = torch.min(tmp_max, torch.ones(shape, device=self.args.gpu))

        """ 2.Logistic function"""
        score = torch.sigmoid(self.w * score + self.b)

        return score

    def forward(self, triples, negs=None, mode='single'):
        """The functions used in the training phase

        Args:
            triples: The triples ids, as (h, r, t, c), shape:[batch_size, 4].
            negs: Negative samples, defaults to None.
            mode: Choose head-predict or tail-predict, Defaults to 'single'.

        Returns:
            score: The score of triples.
        """
        triples = triples[:, :3].to(torch.int)
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, negs, mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)

        return score

    def get_score(self, batch, mode):
        """The functions used in the testing phase

        Args:
            batch: A batch of data.
            mode: Choose head-predict or tail-predict.

        Returns:
            score: The score of triples.
        """
        triples = batch["positive_sample"]
        triples = triples[:, :3].to(torch.int)
        triples = triples.to(self.args.gpu)

        head_emb, relation_emb, tail_emb = self.tri2emb(triples, mode=mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)

        return score