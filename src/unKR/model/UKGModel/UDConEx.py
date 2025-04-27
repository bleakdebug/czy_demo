import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .model import Model

'''
class UDConEx(nn.Module):
    def __init__(self, entity_size, relation_size, embed_dim, kernel_size, filter_num):
        super(UDConEx, self).__init__()
        
        # DistMult嵌入
        self.entity_embedding = nn.Embedding(entity_size, embed_dim)
        self.relation_embedding = nn.Embedding(relation_size, embed_dim)

        # Complex嵌入
        self.entity_embedding_real = nn.Embedding(entity_size, embed_dim)
        self.entity_embedding_imag = nn.Embedding(entity_size, embed_dim)
        self.relation_embedding_real = nn.Embedding(relation_size, embed_dim)
        self.relation_embedding_imag = nn.Embedding(relation_size, embed_dim)

        # 卷积神经网络
        self.conv1 = nn.Conv2d(1, filter_num, (kernel_size, embed_dim))
        self.fc1 = nn.Linear(filter_num, 1)

    def distmult_score(self, h, r, t):
        return torch.sum(h * r * t, dim=-1)

    def complex_score(self, h, r, t):
        real_part = torch.sum(h.real * r.real * t.real + h.imag * r.imag * t.imag, dim=-1)
        imag_part = torch.sum(h.real * r.imag * t.imag - h.imag * r.real * t.real, dim=-1)
        return real_part + imag_part

    def convE_score(self, h, r):
        h_r_concat = torch.cat([h.unsqueeze(1), r.unsqueeze(1)], dim=1)
        conv_output = self.conv1(h_r_concat)
        flattened = conv_output.view(conv_output.size(0), -1)
        return self.fc1(flattened)

    def forward(self, h_idx, r_idx, t_idx):
        # DistMult 嵌入
        h = self.entity_embedding(h_idx)
        r = self.relation_embedding(r_idx)
        t = self.entity_embedding(t_idx)

        # Complex 嵌入
        h_real = self.entity_embedding_real(h_idx)
        h_imag = self.entity_embedding_imag(h_idx)
        t_real = self.entity_embedding_real(t_idx)
        t_imag = self.entity_embedding_imag(t_idx)
        r_real = self.relation_embedding_real(r_idx)
        r_imag = self.relation_embedding_imag(r_idx)

        # 得分计算
        distmult_score = self.distmult_score(h, r, t)
        complex_score = self.complex_score(h_real, r_real, t_real)  # + Imaginary part
        convE_score = self.convE_score(h, r)

        # 综合得分
        final_score = distmult_score + convE_score * complex_score
        return final_score
'''


class UDConEx(Model):
    """
    A model that combines DistMult, ComplEx, and ConvE embeddings for Knowledge Graph Embedding.
    
    Attributes:
        entity_embedding: Entity embedding for DistMult and ComplEx.
        relation_embedding: Relation embedding for DistMult and ComplEx.
        entity_embedding_real: Real part of entity embedding for ComplEx.
        entity_embedding_imag: Imaginary part of entity embedding for ComplEx.
        relation_embedding_real: Real part of relation embedding for ComplEx.
        relation_embedding_imag: Imaginary part of relation embedding for ComplEx.
        conv1: Convolutional layer for ConvE model.
        fc1: Fully connected layer for ConvE model.
        w: Weight when calculating confidence scores.
        b: Bias when calculating confidence scores.
    """

    def __init__(self, args):
        super(UDConEx, self).__init__(args)
        self.args = args
        
        # Initialize embeddings and layers
        self.entity_embedding = None
        self.relation_embedding = None
        
        self.entity_embedding_real = None
        self.entity_embedding_imag = None
        self.relation_embedding_real = None
        self.relation_embedding_imag = None
        
        self.conv1 = nn.Conv2d(1, self.args.filter_num, (self.args.kernel_size, self.args.emb_dim))
        self.fc1 = nn.Linear(self.args.filter_num, 1)
        
        self.w = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.b = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.init_emb()

    def init_emb(self):
        """Initialize the entity and relation embeddings using Xavier uniform distribution."""
         # Initialize embeddings and layers
        self.entity_embedding = nn.Embedding(self.args.num_ent, self.args.emb_dim)
        self.relation_embedding = nn.Embedding(self.args.num_rel, self.args.emb_dim)
        
        self.entity_embedding_real = nn.Embedding(self.args.num_ent, self.args.emb_dim)
        self.entity_embedding_imag = nn.Embedding(self.args.num_ent, self.args.emb_dim)
        self.relation_embedding_real = nn.Embedding(self.args.num_rel, self.args.emb_dim)
        self.relation_embedding_imag = nn.Embedding(self.args.num_rel, self.args.emb_dim)
        
        nn.init.xavier_uniform_(self.entity_embedding.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding.weight.data)
        nn.init.xavier_uniform_(self.entity_embedding_real.weight.data)
        nn.init.xavier_uniform_(self.entity_embedding_imag.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding_real.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding_imag.weight.data)

    def distmult_score(self, h, r, t):
        return torch.sum(h * r * t, dim=-1)

    def complex_score(self, h, r, t):
        real_part = torch.sum(h.real * r.real * t.real + h.imag * r.imag * t.imag, dim=-1)
        imag_part = torch.sum(h.real * r.imag * t.imag - h.imag * r.real * t.real, dim=-1)
        return real_part + imag_part

    def convE_score(self, h, r):
        h_r_concat = torch.cat([h.unsqueeze(1), r.unsqueeze(1)], dim=1)
        conv_output = self.conv1(h_r_concat)
        flattened = conv_output.view(conv_output.size(0), -1)
        return self.fc1(flattened)

    def score_func(self, head_emb, relation_emb, tail_emb, mode):
        """Calculate the score of triples using DistMult, ComplEx, and ConvE."""
        distmult_score = self.distmult_score(head_emb, relation_emb, tail_emb)
        
        re_head, im_head = torch.chunk(head_emb, 2, dim=-1)
        re_relation, im_relation = torch.chunk(relation_emb, 2, dim=-1)
        re_tail, im_tail = torch.chunk(tail_emb, 2, dim=-1)
        complex_score = torch.sum(
            re_head * re_tail * re_relation
            + im_head * im_tail * re_relation
            + re_head * im_tail * im_relation
            - im_head * re_tail * im_relation,
            -1
        )
        #complex_score = self.complex_score(re_head, re_relation, re_tail)
        convE_score = self.convE_score(head_emb, relation_emb)

        # Final score combines DistMult, ComplEx, and ConvE contributions
        final_score = distmult_score + convE_score * complex_score
        score = torch.sigmoid(self.w * final_score + self.b)

        return score

    def forward(self, triples, negs=None, mode='single'):
        """Compute the score for a batch of triples during training."""
        triples = triples[:, :3].to(torch.int)
        
        # Convert triples to embeddings
        head_emb = self.entity_embedding(triples[:, 0])
        relation_emb = self.relation_embedding(triples[:, 1])
        tail_emb = self.entity_embedding(triples[:, 2])

        # Get the score using the score function
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)

        return score

    def get_score(self, batch, mode):
        """Compute the score for a batch of triples during testing."""
        triples = batch["positive_sample"]
        triples = triples[:, :3].to(torch.int)

        # Convert triples to embeddings
        head_emb = self.entity_embedding(triples[:, 0])
        relation_emb = self.relation_embedding(triples[:, 1])
        tail_emb = self.entity_embedding(triples[:, 2])

        # Get the score using the score function
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)

        return score
