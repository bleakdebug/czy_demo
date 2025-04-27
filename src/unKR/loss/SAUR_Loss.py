import torch
import torch.nn as nn
import torch.nn.functional as F

class SAUR_Loss(nn.Module):
    """The loss function of SAUR

    Attributes:
        args: Some pre-set parameters, etc
        model: The SAUR model for training.
    """
    def __init__(self, args, model):
        super(SAUR_Loss, self).__init__()
        self.args = args
        self.model = model

    def get_logic_loss(self, model, ids, args):
        """
        Calculates the logic loss for the model based on transitive and composite rule regularizations.

        Args:
            model: The SAUR model instance.
            ids: Tensor of triple ids.
            args: Model configuration parameters including regularization coefficients.

        Returns:
            The logic loss calculated from transitive and composite rule regularizations.
        """
        ids = ids[:, :3].to(torch.long)
        # transitive rule loss regularization
        transitive_coff = torch.tensor(args.transitive).to(args.gpu)
        if transitive_coff > 0:
            transitive_rule_reg = transitive_coff * model.transitive_rule_loss(ids)
        else:
            transitive_rule_reg = 0

        # composite rule loss regularization
        composite_coff = torch.tensor(args.composite).to(args.gpu)
        if composite_coff > 0:
            composition_rule_reg = composite_coff * model.composition_rule_loss(ids)
        else:
            composition_rule_reg = 0

        return (transitive_rule_reg + composition_rule_reg) / len(ids)

    def main_mse_loss(self, model, ids):
        """
        Computes the Mean Squared Error (MSE) loss for the model.

        Args:
            model: The SAUR model instance.
            ids: Tensor of triple ids(the confidence of negative samples is zero).

        Returns:
            mse: The MSE loss for the given triples.
        """
        criterion = nn.MSELoss(reduction='mean')
        prediction = model(ids[:, :3].to(torch.long), train=True).to(torch.float64)

        if ids.shape[1] > 3:
            truth = ids[:, 3]
        else:
            truth = torch.zeros(ids.shape[0], dtype=torch.float64).to(self.args.gpu)

        mse = criterion(prediction, truth)

        return mse

    def L2_regularization(self, model, ids, args):
        """
        Computes the L2 regularization loss for the model.

        Args:
            model: The SAUR model instance.
            ids: Tensor of triple ids.
            args: Model configuration parameters including regularization coefficients.

        Returns:
            L2_reg: The L2 regularization loss.
        """
        ids = ids[:, :3].to(torch.long)
        regularization = args.regularization
        device = args.gpu

        # regularization on entity embeddings
        ent_coff = torch.tensor(args.entity).to(device)
        ent_reg = ent_coff * (
            torch.norm(model.entity_embedding[ids[:, 0]], dim=1).mean() + \
            torch.norm(model.entity_embedding[ids[:, 2]], dim=1).mean()
        )

        # regularization on relation embeddings
        rel_coff = torch.tensor(args.relation).to(device)
        rel_reg = rel_coff * torch.norm(model.relation_embedding[ids[:, 1]], dim=1).mean()

        # regularization on attention weights
        att_coff = torch.tensor(args.attention).to(device)
        att_reg = att_coff * torch.norm(model.attention_weights, dim=1).mean()

        L2_reg = ent_reg + rel_reg + att_reg

        return L2_reg

    def forward(self, pos_scores, neg_scores, target=None, args=None):
        """
        计算损失函数值

        Args:
            pos_scores: 正样本的预测分数
            neg_scores: 负样本的预测分数
            target: 目标置信度值
            args: 模型参数配置（可选）

        Returns:
            loss: 计算的损失值
        """
        # 确定设备
        device = pos_scores.device
        
        # 创建默认目标值（如果未提供）
        if target is None:
            target = torch.ones_like(pos_scores)
        else:
            # 确保target在正确的设备上
            target = target.to(device)
            
        # 确保neg_scores在正确的设备上
        neg_scores = neg_scores.to(device)
        
        # 使用MSE损失函数
        pos_loss = F.mse_loss(pos_scores, target)
        neg_loss = F.mse_loss(neg_scores, torch.zeros_like(neg_scores))
        
        # 组合正负样本损失
        loss = pos_loss + neg_loss
        
        # 添加可选的正则化项
        if args is not None and hasattr(args, 'weight_decay') and args.weight_decay > 0:
            l2_reg = 0
            for param in self.model.parameters():
                l2_reg += torch.norm(param)
            loss += args.weight_decay * l2_reg
            
        return loss 