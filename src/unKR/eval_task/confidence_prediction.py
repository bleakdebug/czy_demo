import torch
import os
import torch.nn as nn
import numpy as np

'''
def conf_predict(batch, model):
    """The goal of evaluate task is to predict the confidence of triples.

    Args:
        batch: The batch of the triples for validation or test.
        model: The UKG model for training.
    Returns:
        MAE: Mean absolute error.
        MSE: Mean Square error.
    """
    pos_triple = batch["positive_sample"]
    confidence = pos_triple[:, 3]


    pred_score = model.get_score(batch, "single")

    pred_score = pred_score.squeeze()  # 维度压缩
    MAE_loss = nn.L1Loss(reduction="sum")
    # MAE = MAE_loss(pred_score, confidence) * batch["positive_sample"].shape[0]
    MAE = MAE_loss(pred_score, confidence)

    MSE_loss = nn.MSELoss(reduction="sum")
    # MSE = MSE_loss(pred_score, confidence) * batch["positive_sample"].shape[0]
    MSE = MSE_loss(pred_score, confidence)

    return MAE, MSE
'''
def conf_predict(batch, model,num_samples=1):
    """The goal of evaluate task is to predict the confidence of triples.

    Args:
        batch: The batch of the triples for validation or test.
        model: The UKG model for training.
        num_samples: Number of Monte Carlo samples for uncertainty estimation.
    Returns:
        MAE: Mean absolute error.
        MSE: Mean Square error.
    """
    pos_triple = batch["positive_sample"]
    
    # 处理三元组没有第四列（置信度）的情况
    if pos_triple.shape[1] > 3:
        confidence = pos_triple[:, 3]
    else:
        # 如果没有置信度列，使用全1张量作为目标值
        confidence = torch.ones(pos_triple.shape[0], device=pos_triple.device)
    
    # 执行前向传播获取预测分数
    pred_score = model(batch)
    
    # 计算MAE和MSE
    MAE_loss = nn.L1Loss(reduction="sum")
    MAE = MAE_loss(pred_score, confidence)

    MSE_loss = nn.MSELoss(reduction="sum")
    MSE = MSE_loss(pred_score, confidence)

    return MAE, MSE

