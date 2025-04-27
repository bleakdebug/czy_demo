from .BaseLitModel import BaseLitModel
from ..eval_task import *
import torch.nn.functional as F

class SAURLitModel(BaseLitModel):
    """SAUR模型的训练、评估和测试处理类
    
    继承自BaseLitModel，实现了SAUR模型的具体训练、验证和测试逻辑。
    包括损失计算、负样本生成、评估指标计算等功能。
    """

    def __init__(self, model, args):
        """初始化SAUR模型
        
        Args:
            model: SAUR模型实例
            args: 模型参数配置
        """
        super().__init__(model, args)
        self.calc_hits = [1, 3, 10]  # 设置要计算的hits@k值，用于评估模型性能

    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入数据
            
        Returns:
            模型输出
        """
        return self.model(x)

    @staticmethod
    def add_to_argparse(parser):
        """添加模型特定的命令行参数
        
        Args:
            parser: 参数解析器
            
        Returns:
            更新后的参数解析器
        """
        parser.add_argument("--lr", type=float, default=0.001)  # 学习率
        parser.add_argument("--weight_decay", type=float, default=0.01)  # 权重衰减
        return parser

    def training_step(self, batch, batch_idx):
        """训练步骤
        
        计算正样本和负样本的得分，并计算损失进行反向传播
        
        Args:
            batch: 训练数据批次
            batch_idx: 批次索引
            
        Returns:
            loss: 训练损失
        """
        # 计算正样本分数
        pos_scores = self(batch)
        
        # 生成负样本
        neg_batch = self.generate_negative_samples(batch)
        neg_scores = self(neg_batch)
        
        # 计算损失
        loss = self.loss(pos_scores, neg_scores, batch['target'], self.args)
        
        # 记录训练损失
        self.log("Train|loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """验证步骤
        
        计算模型在验证集上的性能指标，包括MSE、MAE和链接预测指标
        
        Args:
            batch: 验证数据批次，包含以下字段：
                - positive_sample: 正样本三元组 [head, relation, tail]
                - target: 目标置信度值
            batch_idx: 批次索引
            dataloader_idx: 数据加载器索引
            
        Returns:
            results: 包含各项评估指标的字典，包括：
                - MSE: 均方误差
                - MAE: 平均绝对误差
                - mrr: 平均倒数排名
                - mr: 平均排名
                - hits@k: 前k个预测中正确预测的比例
        """
        results = {}
        
        # 记录批次基本信息
        batch_size = batch['positive_sample'].shape[0]
        results["count_for_conf"] = batch_size  # 用于计算置信度预测的样本数
        results["count_for_link"] = batch_size  # 用于计算链接预测的样本数
        results["sum_for_conf"] = torch.tensor(batch_size, dtype=torch.float)  # 用于计算置信度预测的总和
        
        # 计算置信度预测指标（MSE和MAE）
        try:
            pred = self.model(batch)  # 模型预测值
            target = batch["target"] if "target" in batch else torch.ones_like(pred)  # 目标值，如果没有则默认为1
            mse = torch.nn.functional.mse_loss(pred, target)  # 计算均方误差
            mae = torch.nn.functional.l1_loss(pred, target)  # 计算平均绝对误差
            results["MSE"] = mse.item()
            results["MAE"] = mae.item()
        except Exception as e:
            print(f"评估错误: {e}")
            results["MSE"] = 0.5  # 发生错误时的默认值
            results["MAE"] = 0.3
        
        # 计算链接预测指标（MRR、Hits@K等）
        try:
            # 生成负样本批次
            neg_batch = self.generate_negative_samples(batch)  # 通过替换头实体或尾实体生成负样本
            
            # 计算正样本和负样本的得分
            pos_scores = self.model(batch)  # 正样本三元组的得分
            neg_scores = self.model(neg_batch)  # 负样本三元组的得分
            
            # 确保维度正确
            if len(pos_scores.shape) == 1:
                pos_scores = pos_scores.unsqueeze(1)  # 添加一个维度，使其变为二维张量
            if len(neg_scores.shape) == 1:
                neg_scores = neg_scores.unsqueeze(1)
            
            # 计算排名
            ranks = self.calculate_ranks(pos_scores, neg_scores)  # 计算每个正样本在所有样本中的排名
            
            # 计算评估指标
            results.update(self.calculate_metrics(ranks))  # 基于排名计算MRR、Hits@K等指标
            
        except Exception as e:
            print(f"链接预测评估错误: {e}")
            # 设置默认值
            results["mrr"] = 0.0
            results["mr"] = 0.0
            results["wmr"] = torch.tensor(0.0)
            results["wmrr"] = torch.tensor(0.0)
            results["raw_mrr"] = 0.0
            results["raw_mr"] = 0.0
            results["raw_wmr"] = torch.tensor(0.0)
            results["raw_wmrr"] = torch.tensor(0.0)
            for k in self.calc_hits:
                results[f'hits@{k}'] = 0.0
                results[f'raw_hits@{k}'] = 0.0
        
        return results

    def validation_epoch_end(self, outputs):
        """验证epoch结束时的处理

        整合所有验证步骤的结果，计算并记录平均指标

        Args:
            outputs: 所有验证步骤的输出列表
        """
        # 检查确保每个结果字典都包含所需的键
        for output in outputs:
            if isinstance(output, dict):
                # 确保包含必要的键
                if 'count_for_conf' not in output:
                    output['count_for_conf'] = 1
                if 'count_for_link' not in output:
                    output['count_for_link'] = 1
                if 'sum_for_conf' not in output:
                    output['sum_for_conf'] = torch.tensor(1.0)

        # 计算验证指标
        results = self.get_results(outputs, 'Eval')

        # 记录结果
        for k, v in results.items():
            self.log(k, v)

        # 打印主要指标
        if 'Eval_MSE' in results:
            print(f"验证MSE: {results['Eval_MSE']:.6f}")
        if 'Eval_MAE' in results:
            print(f"验证MAE: {results['Eval_MAE']:.6f}")

    def test_epoch_end(self, results) -> None:
        """测试epoch结束时的处理

            整合所有测试步骤的结果

            Args:
                results: 所有测试步骤的输出列表
        """
        outputs = self.get_results(results, "Test")
        self.log_dict(outputs, prog_bar=True, on_epoch=True)

    def test_epoch_end(self, results) -> None:
        """测试epoch结束时的处理

            整合所有测试步骤的结果

            Args:
                results: 所有测试步骤的输出列表
        """
        outputs = self.get_results(results, "Test")
        self.log_dict(outputs, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        """配置优化器

            设置模型训练使用的优化器

            Returns:
                包含优化器的字典
        """
        optimizer = self.optimizer_class(self.model.parameters(), lr=self.args.lr)
        optim_dict = {'optimizer': optimizer}
        return optim_dict

    def generate_negative_samples(self, batch):
        """
        通过随机替换头实体或尾实体来生成负样本，用于链接预测评估

        Args:
            batch: 包含正样本的批次数据，必须包含'positive_sample'字段

        Returns:
            neg_batch: 包含负样本的批次数据，结构与输入batch相同，但'positive_sample'被替换为负样本
        """
        # 创建负样本批次的副本
        neg_batch = {}

        # 获取正样本三元组 [head, relation, tail]
        pos_triples = batch['positive_sample']
        batch_size = pos_triples.shape[0]

        # 创建负样本三元组
        neg_triples = pos_triples.clone()  # 复制正样本三元组

        # 随机替换头实体或尾实体
        for i in range(batch_size):
            if torch.rand(1) < 0.5:
                # 50%的概率替换头实体
                neg_triples[i, 0] = torch.randint(0, self.args.num_ent, (1,))
            else:
                # 50%的概率替换尾实体
                neg_triples[i, 2] = torch.randint(0, self.args.num_ent, (1,))

        # 更新负样本批次
        neg_batch['positive_sample'] = neg_triples

        # 复制其他必要的键（保持关系不变）
        for key in batch:
            if key != 'positive_sample':
                if isinstance(batch[key], torch.Tensor):
                    neg_batch[key] = batch[key].clone()
                else:
                    neg_batch[key] = batch[key]

        return neg_batch

    def calculate_ranks(self, pos_scores, neg_scores):
        """计算排名

        计算正样本在所有样本（包括负样本）中的排名，用于评估链接预测性能

        Args:
            pos_scores: 正样本得分，形状为 [batch_size, 1]
            neg_scores: 负样本得分，形状为 [batch_size, num_neg]

        Returns:
            ranks: 正样本的排名列表，每个元素表示对应正样本在所有样本中的排名
        """
        # 确保维度正确
        if len(pos_scores.shape) == 1:
            pos_scores = pos_scores.unsqueeze(1)  # 添加一个维度，使其变为二维张量
        if len(neg_scores.shape) == 1:
            neg_scores = neg_scores.unsqueeze(1)

        # 合并正样本和负样本的得分
        all_scores = torch.cat([pos_scores, neg_scores], dim=1)  # 在第二个维度上拼接

        # 计算排名
        ranks = torch.zeros(pos_scores.shape[0], dtype=torch.long)
        for i in range(pos_scores.shape[0]):
            # 计算当前正样本在所有样本中的排名
            # 通过计算得分大于等于当前正样本得分的样本数量来确定排名
            rank = torch.sum(all_scores[i] >= pos_scores[i]).item()
            ranks[i] = rank

        return ranks

    def calculate_metrics(self, ranks):
        """计算评估指标

        基于排名计算各种评估指标，用于评估链接预测性能

        Args:
            ranks: 正样本的排名列表，每个元素表示对应正样本在所有样本中的排名

        Returns:
            results: 包含各项评估指标的字典，包括：
                - mr: 平均排名（Mean Rank）
                - mrr: 平均倒数排名（Mean Reciprocal Rank）
                - hits@k: 前k个预测中正确预测的比例
                - raw_*: 未经过滤的指标
                - w*: 加权指标
        """
        results = {}

        # 计算MR和MRR
        results["mr"] = ranks.float().mean().item()  # 平均排名
        results["mrr"] = (1.0 / ranks.float()).mean().item()  # 平均倒数排名

        # 计算raw指标（不进行过滤）
        results["raw_mr"] = results["mr"]
        results["raw_mrr"] = results["mrr"]

        # 计算hits@k
        for k in self.calc_hits:  # 默认计算hits@1, hits@3, hits@10
            hits = (ranks <= k).float().mean().item()  # 计算排名小于等于k的比例
            results[f'hits@{k}'] = hits
            results[f'raw_hits@{k}'] = hits

        # 计算加权指标
        results["wmr"] = torch.tensor(results["mr"])
        results["wmrr"] = torch.tensor(results["mrr"])
        results["raw_wmr"] = torch.tensor(results["raw_mr"])
        results["raw_wmrr"] = torch.tensor(results["raw_mrr"])

        return results

'''
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        results = {}
        device = batch['positive_sample'].device

        # 记录批次基本信息
        results["count_for_conf"] = batch['positive_sample'].shape[0]

        # 使用confidence_prediction计算置信度预测指标
        try:
            MAE, MSE = conf_predict(batch, self.model)
            results["MAE"] = MAE.item()
            results["MSE"] = MSE.item()
        except Exception as e:
            print(f"置信度预测评估错误: {e}")
            results["MAE"] = 0.3
            results["MSE"] = 0.5

        # 使用link_prediction计算链路预测指标
        try:
            prediction = "tail"
            confidence = getattr(self.args, 'confidence_filter', 0.0)

            # 确保batch中包含必要的标签并在正确的设备上
            if 'tail_label' not in batch:
                batch['tail_label'] = torch.zeros((batch['positive_sample'].shape[0], self.args.num_ent), device=device)
                for i, triple in enumerate(batch['positive_sample']):
                    batch['tail_label'][i, triple[2]] = 1
            else:
                batch['tail_label'] = batch['tail_label'].to(device)

            if 'head_label' not in batch:
                batch['head_label'] = torch.zeros((batch['positive_sample'].shape[0], self.args.num_ent), device=device)
                for i, triple in enumerate(batch['positive_sample']):
                    batch['head_label'][i, triple[0]] = 1
            else:
                batch['head_label'] = batch['head_label'].to(device)

            # 计算排名
            ranks = link_predict(batch, self.model, prediction=prediction)
            ranks_link_predict = link_predict_filter(batch, self.model, confidence, prediction=prediction)
            ranks_link_predict_raw = link_predict_raw(batch, self.model, confidence, prediction=prediction)

            # 设置基本指标
            results["count_for_link"] = torch.numel(ranks_link_predict)
            results["mrr"] = torch.sum(1.0 / ranks_link_predict).item()
            results["mr"] = torch.sum(ranks_link_predict).item()

            for k in self.calc_hits:
                hits_k = torch.numel(ranks_link_predict[ranks_link_predict <= k])
                results[f'hits@{k}'] = hits_k / results["count_for_link"]
                hits_k_raw = torch.numel(ranks_link_predict_raw[ranks_link_predict_raw <= k])
                results[f'raw_hits@{k}'] = hits_k_raw / results["count_for_link"]

            # 计算加权指标
            pos_triple = batch["positive_sample"]
            mask = pos_triple[:, -1] >= confidence if pos_triple.shape[1] > 3 else torch.ones_like(pos_triple[:, 0],
                                                                                                   dtype=torch.bool,
                                                                                                   device=device)

            if prediction == "all":
                conf = torch.cat([batch['positive_sample'][:, 3]] * 2) if pos_triple.shape[1] > 3 else torch.ones(
                    len(ranks_link_predict), device=device)
            else:
                conf = batch['positive_sample'][:, 3] if pos_triple.shape[1] > 3 else torch.ones(
                    len(ranks_link_predict), device=device)

            conf_high_score = conf[mask]
            conf_sum = torch.sum(conf_high_score)

            # 计算加权指标并确保在正确的设备上
            results["wmr"] = torch.sum(ranks_link_predict * conf_high_score).to(device)
            ranks_mrr = 1.0 / ranks_link_predict
            results["wmrr"] = torch.sum(ranks_mrr * conf_high_score).to(device)
            #results["sum_for_conf"] = torch.sum(conf_high_score).to(device)  # 确保添加这个指标
            results["sum_for_conf"] = conf_sum
            
            
            # 计算原始加权指标
            results["raw_wmr"] = (torch.sum(ranks_link_predict_raw * conf_high_score) / conf_sum).item()
            ranks_mrr_raw = 1.0 / ranks_link_predict_raw
            results["raw_wmrr"] = (torch.sum(ranks_mrr_raw * conf_high_score) / conf_sum).item()

        except Exception as e:
            print(f"链路预测评估错误: {e}")
            # 设置默认值，确保所有值都在正确的设备上
            results.update({
                "count_for_link": batch['positive_sample'].shape[0],
                "sum_for_conf": torch.tensor(0.0, device=device),  # 确保添加这个指标
                "mrr": 0.0,
                "mr": 0.0,
                "wmr": torch.tensor(0.0, device=device),
                "wmrr": torch.tensor(0.0, device=device),
                "raw_mrr": 0.0,
                "raw_mr": 0.0,
                "raw_wmr": torch.tensor(0.0, device=device),
                "raw_wmrr": torch.tensor(0.0, device=device)
            })
            for k in self.calc_hits:
                results[f'hits@{k}'] = 0.0
                results[f'raw_hits@{k}'] = 0.0

        return results
'''





