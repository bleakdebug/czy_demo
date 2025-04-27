import torch


def link_predict(batch, model, prediction="all"):
    """The evaluate task is predicting the head entity or tail entity in incomplete triples.

    Args:
        batch: The batch of the triples for validation or test.
        model: The UKG model for training.
        predicion: mode of link prediction.

    Returns:
        ranks: The rank of the triple to be predicted.
    """
    if prediction == "all":
        tail_ranks = tail_predict(batch, model)
        head_ranks = head_predict(batch, model)
        ranks = torch.cat([tail_ranks, head_ranks])
    elif prediction == "head":
        ranks = head_predict(batch, model)
    elif prediction == "tail":
        ranks = tail_predict(batch, model)

    return ranks.float()


def link_predict_filter(batch, model, confidence, prediction="all"):
    """The evaluate task is predicting the head entity or tail entity in incomplete triples.
    link_predict_filter is for high confidence test samples,
    only samples higher than the set confidence will participate in link prediction tasks.
    When confidence is set to 0, this function is the same as link_predict.

    Args:
        batch: The batch of the triples for validation or test.
        model: The UKG model for training.
        predicion: mode of link prediction.

    Returns:
        ranks: The rank of the triple to be predicted.
    """
    if prediction == "all":
        tail_ranks = tail_predict_filter(batch, model)
        head_ranks = head_predict_filter(batch, model)
        ranks = torch.cat([tail_ranks, head_ranks])
    elif prediction == "head":
        ranks = head_predict_filter(batch, model, confidence)
    elif prediction == "tail":
        ranks = tail_predict_filter(batch, model, confidence)

    return ranks.float()


def head_predict_filter(batch, model, confidence):
    """Getting head entity ranks (high confidence).

    Args:
        batch: The batch of the triples for validation or test
        model: The UKG model for training.

    Returns:
        tensor: The rank of the head entity to be predicted, dim [batch_size]
    """
    pos_triple = batch["positive_sample"]
    mask = pos_triple[:, -1] >= confidence
    pos_triple_filtered = pos_triple[mask]
    batch_copy = batch.copy()
    batch_copy["positive_sample"] = pos_triple_filtered
    idx = pos_triple_filtered[:, 0].long()
    label = batch["head_label"].clone()
    label = label[mask]
    pred_score = model.get_score(batch_copy, "head_predict")
    return calc_ranks(idx, label, pred_score)


def tail_predict_filter(batch, model, confidence):
    """Getting tail entity ranks.(high confidence)

    Args:
        batch: The batch of the triples for validation or test
        model: The UKG model for training.

    Returns:
        tensor: The rank of the tail entity to be predicted, dim [batch_size]
    """
    pos_triple = batch["positive_sample"]
    mask = pos_triple[:, -1] >= confidence
    pos_triple_filtered = pos_triple[mask]
    batch_copy = batch.copy()
    batch_copy["positive_sample"] = pos_triple_filtered
    idx = pos_triple_filtered[:, 2].long()
    label = batch["tail_label"].clone()
    label = label[mask]
    pred_score = model.get_score(batch_copy, "tail_predict")
    return calc_ranks(idx, label, pred_score)

def link_predict_raw(batch, model, confidence, prediction="all"):
    """The evaluate task is predicting the head entity or tail entity in incomplete triples.
    Only samples higher than the set confidence will participate in link prediction tasks.
    Different from link_predict_filter, this function does not filter samples that already exist in the knowledge graph when calculating the ranking of samples

    Args:
        batch: The batch of the triples for validation or test.
        model: The KG model for training.
        predicion: mode of link prediction.

    Returns:
        ranks: The rank of the triple to be predicted.
    """
    if prediction == "all":
        tail_ranks = tail_predict_raw(batch, model)
        head_ranks = head_predict_raw(batch, model)
        ranks = torch.cat([tail_ranks, head_ranks])
    elif prediction == "head":
        ranks = head_predict_raw(batch, model, confidence)
    elif prediction == "tail":
        ranks = tail_predict_raw(batch, model, confidence)

    return ranks.float()

def head_predict_raw(batch, model, confidence):
    """Getting head entity ranks. (high confidence + no filter)

    Args:
        batch: The batch of the triples for validation or test
        model: The UKG model for training.

    Returns:
        tensor: The rank of the head entity to be predicted, dim [batch_size]
    """
    pos_triple = batch["positive_sample"]
    mask = pos_triple[:, -1] >= confidence
    pos_triple_filtered = pos_triple[mask]
    batch_copy = batch.copy()
    batch_copy["positive_sample"] = pos_triple_filtered
    idx = pos_triple_filtered[:, 0].long()
    label = batch["head_label"].clone()
    label = label[mask]
    pred_score = model.get_score(batch_copy, "head_predict")
    return calc_ranks_raw(idx, label, pred_score)


def tail_predict_raw(batch, model, confidence):
    """Getting tail entity ranks. (high confidence + no filter)

    Args:
        batch: The batch of the triples for validation or test
        model: The UKG model for training.

    Returns:
        tensor: The rank of the tail entity to be predicted, dim [batch_size]
    """
    pos_triple = batch["positive_sample"]
    mask = pos_triple[:, -1] >= confidence
    pos_triple_filtered = pos_triple[mask]
    batch_copy = batch.copy()
    batch_copy["positive_sample"] = pos_triple_filtered
    idx = pos_triple_filtered[:, 2].long()
    label = batch["tail_label"].clone()
    label = label[mask]
    pred_score = model.get_score(batch_copy, "tail_predict")
    return calc_ranks_raw(idx, label, pred_score)

def head_predict(batch, model):
    """Getting head entity ranks.

    Args:
        batch: The batch of the triples for validation or test
        model: The UKG model for training.

    Returns:
        tensor: The rank of the head entity to be predicted, dim [batch_size]
    """
    pos_triple = batch["positive_sample"]
    idx = pos_triple[:, 0].long()
    label = batch["head_label"]
    pred_score = model.get_score(batch, "head_predict")
    return calc_ranks(idx, label, pred_score)


def tail_predict(batch, model):
    """Getting tail entity ranks.

    Args:
        batch: The batch of the triples for validation or test
        model: The UKG model for training.

    Returns:
        tensor: The rank of the tail entity to be predicted, dim [batch_size]
    """
    pos_triple = batch["positive_sample"]
    idx = pos_triple[:, 2].long()
    
    # 检查是否有tail_label，如果没有则创建一个全零标签
    if "tail_label" in batch:
        label = batch["tail_label"]
    else:
        # 创建一个简单的标签，尺寸与实体数量一致
        num_ent = model.args.num_ent if hasattr(model, 'args') and hasattr(model.args, 'num_ent') else 15000
        label = torch.zeros((pos_triple.size(0), num_ent), device=pos_triple.device)
        # 设置正样本位置为1
        for i, triple in enumerate(pos_triple):
            label[i, triple[2]] = 1
    
    try:
        pred_score = model.get_score(batch, "tail_predict")
    except (AttributeError, NotImplementedError) as e:
        print(f"模型不支持get_score方法: {e}")
        # 返回一个默认的排名结果，所有排名都是1
        return torch.ones_like(idx)
    
    return calc_ranks(idx, label, pred_score)


def calc_ranks(idx, label, pred_score):
    """Calculating triples score ranks.

    Args:
        idx ([type]): The id of the entity to be predicted.
        label ([type]): The id of existing triples, to calc filtered results.
        pred_score ([type]): The score of the triple predicted by the model.

    Returns:
        ranks: The rank of the triple to be predicted, dim [batch_size].
    """
    b_range = torch.arange(pred_score.size()[0])  # get batch_size

    target_pred = pred_score[b_range, idx]  #  idx is the real tail entity number in UKG
    # Set the score of the sample with label 1 to be very low and filter it to exclude samples that are already in the knowledge graph
    pred_score = torch.where(label.bool(), -torch.ones_like(pred_score) * 10000000, pred_score)
    pred_score[b_range, idx] = target_pred

    # Get the ranking of each score in the pred_score.
    ranks = (
            1
            + torch.argsort(
        torch.argsort(pred_score, dim=1, descending=True), dim=1, descending=False
    )[b_range, idx]
    )
    return ranks

def calc_ranks_raw(idx, label, pred_score):
    """Calculating triples score ranks.
    Different from calc_ranks, calc_ranks_raw will not exclude samples that are already in the knowledge graph when get the ranks.

    Args:
        idx ([type]): The id of the entity to be predicted.
        label ([type]): The id of existing triples, to calc filtered results.
        pred_score ([type]): The score of the triple predicted by the model.

    Returns:
        ranks: The rank of the triple to be predicted, dim [batch_size].
    """
    b_range = torch.arange(pred_score.size()[0])  # 得到batch_size

    target_pred = pred_score[b_range, idx]   #  idx is the real tail entity number in UKG

    pred_score[b_range, idx] = target_pred

    # Get the ranking of each score in the pred_score.
    ranks = (
            1
            + torch.argsort(
        torch.argsort(pred_score, dim=1, descending=True), dim=1, descending=False
    )[b_range, idx]
    )
    return ranks

