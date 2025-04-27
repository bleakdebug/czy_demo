'''
# 映射函数：有界整流函数
def bounded_relu(score):
    return torch.clamp(score, 0, 1)

# 映射函数：Sigmoid函数
def sigmoid_mapping(score):
    return torch.sigmoid(score)

# 损失函数
def mse_loss(pred_score, true_score):
    return F.mse_loss(pred_score, true_score)

def compute_loss(model, positive_samples, negative_samples, true_scores_pos, true_scores_neg, neg_sample_size):
    # 正例损失
    pos_scores = model(*positive_samples)
    pos_pred = bounded_relu(pos_scores)
    pos_loss = mse_loss(pos_pred, true_scores_pos)
    
    # 负例损失
    neg_scores = model(*negative_samples)
    neg_pred = bounded_relu(neg_scores)
    neg_loss = mse_loss(neg_pred, true_scores_neg) / neg_sample_size
    
    # 总损失
    total_loss = pos_loss + neg_loss
    return total_loss
'''
import torch
import torch.nn.functional as F
import torch.nn as nn

class UDConEX_Loss(nn.Module):
        def __init__(self, args, model):
                super(UDConEX_Loss, self).__init__()
                self.args = args
                self.model = model

        def forward(self, pos_score, neg_score, pos_sample):
                # 提取正样本的置信度
                confidence = pos_sample[:, 3]  # 假设正样本的置信度在pos_sample的第四列

        # 映射函数：有界整流函数
                pos_score = torch.clamp(pos_score, 0, 1)
                neg_score = torch.clamp(neg_score, 0, 1)

        # 计算正例损失
                loss_pos = torch.sum((pos_score - confidence) ** 2)  # L_pos

        # 计算负例损失
                loss_neg = torch.sum(neg_score ** 2)  # L_neg

        # 总损失
                loss = loss_pos + loss_neg / neg_score.shape[1]
                return loss