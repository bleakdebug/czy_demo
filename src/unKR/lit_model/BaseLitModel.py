import argparse
import pytorch_lightning as pl
import torch
import torch.nn as nn
from collections import defaultdict as ddict
from unKR import loss
import numpy as np
import matplotlib.pyplot as plt


class Config(dict):
    def __getattr__(self, name):
        return self.get(name)

    def __setattr__(self, name, val):
        self[name] = val


class BaseLitModel(pl.LightningModule):
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """

    def __init__(self, model, args: argparse.Namespace = None, src_list=None, dst_list=None, rel_list=None):
        super().__init__()
        self.train_epoch = 0
        self.model = model
        self.pool = []  # Sample pools
        self.entity = set()  # Entity pools
        self.relation = set()  # Relation pools
        self.semiM = 0  # Maximum number of semi-supervised samples
        self.alpha = 0.02
        self.training_set = []
        self.training_set_triples = []
        self.training_set_triples_list = []
        self.hr2t_train = ddict(set)
        self.rt2h_train = ddict(set)
        self.h2rt_train = ddict(set)
        self.t2rh_train = ddict(set)
        self.args = args
        self.neg_per_positive = args.num_neg
        optim_name = args.optim_name
        self.optimizer_class = getattr(torch.optim, optim_name)
        loss_name = args.loss_name
        self.loss_class = getattr(loss, loss_name)
        self.loss = self.loss_class(args, model)
        ''''''
        self.dropout = nn.Dropout(p=0.5)
        ''''''


    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--lr", type=float, default=0.1)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        return parser

    def configure_optimizers(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        raise NotImplementedError

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        raise NotImplementedError

    def get_results(self, results, mode):
        """Summarize the results of each batch and calculate the final result of the epoch
        Args:
            results ([type]): The results of each batch
            mode ([type]): Eval or Test
        Returns:
            dict: The final result of the epoch
        """
        conf_list = ["MAE", "MSE"]
        ''''''
        #conf_list = ["MAE", "MSE", "pred_mean","pred_std"]
        ''''''
        pred_mean_list = ["pred_mean"]
        pred_std_list = ["pred_std"]
        link_list = ["mrr", "mr", "hits@10", "hits@5", "hits@3", "hits@1"]
        link_list_raw = ["raw_mrr", "raw_mr", "raw_hits@10", "raw_hits@5", "raw_hits@3", "raw_hits@1"]

        outputs = ddict(float)
        count_for_conf = np.array([o["count_for_conf"] for o in results]).sum()
        count_for_link = np.array([o["count_for_link"] for o in results]).sum()
        conf_sum = np.array([o["sum_for_conf"].cpu() for o in results]).sum()

        for metric in list(results[0].keys())[1:]:
            if metric in conf_list:
                final_metric = "_".join([mode, metric])
                outputs[final_metric] = np.around(np.array([o[metric] for o in results]).sum() / count_for_conf,
                                                  decimals=6).item()

            elif metric in pred_mean_list:

                first_tensor_shape = None
                for o in results:
                    if torch.is_tensor(o[metric]):
                        first_tensor_shape = o[metric].shape
                        break

                # 过滤掉形状不匹配的 Tensor
                filtered_tensors = [
                    o[metric] for o in results
                    if torch.is_tensor(o[metric]) and o[metric].shape == first_tensor_shape
                ]

                # 计算平均值
                #outputs[final_metric] = torch.stack(filtered_tensors).sum(dim=0) /
                pred_mean_output = torch.stack(filtered_tensors).sum(dim=0) / count_for_conf
                if(mode=="Test"):
                    torch.set_printoptions(precision=20)
                    print(pred_mean_output)
                    # 子图 1：pred_mean 的变化
                    pred_mean_output = pred_mean_output.cpu()
                    plt.subplot(1, 2, 1)
                    plt.plot(pred_mean_output, label="Predicted Mean", color="blue")
                    plt.xlabel("Batch Index")
                    plt.ylabel("Predicted Confidence")
                    plt.title("Predicted Mean Over Batches")
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plt.show()

                    '''误差条图'''
                    '''
                    pred_mean = pred_mean_output.cpu().numpy().flatten()

                    # 将 pred_std 展平
                    # pred_std = pred_std.flatten()
                    pred_std = pred_std_output.flatten()[:len(pred_mean)]

                    # 创建 x 轴数据（批次索引）
                    batch_indices = np.arange(len(pred_mean))

                    # 绘制误差条图
                    plt.figure(figsize=(10, 6))

                    plt.errorbar(batch_indices, pred_mean, yerr=pred_std, fmt='o', color='blue', ecolor='lightblue',
                                 capsize=5, label='Predicted Mean ± Std')

                    # 设置图形属性
                    plt.xlabel("Batch Index")
                    plt.ylabel("Predicted Confidence")
                    plt.title("Predicted Confidence with Error Bars")
                    plt.legend()
                    plt.grid(True)
                    plt.show()
                    '''
                #outputs[final_metric] = round(outputs[final_metric], 6)

            elif metric in pred_std_list:
                # 获取第一个 numpy 数组的形状作为参考
                first_array_shape = None
                for o in results:
                    if isinstance(o[metric], np.ndarray):
                        first_array_shape = o[metric].shape
                        break

                # 过滤掉形状不匹配的 numpy 数组
                filtered_arrays = [
                    o[metric] for o in results
                    if isinstance(o[metric], np.ndarray) and o[metric].shape == first_array_shape
                ]

                # 计算平均值

                #outputs[final_metric] = np.sum(filtered_arrays, axis=0) / count_for_conf
                pred_std_output = np.sum(filtered_arrays, axis=0) / count_for_conf
                if(mode=="Test"):
                    print(pred_std_output)
                    plt.subplot(1, 2, 2)
                    plt.plot(pred_std_output, label="Predicted Std", color="red")
                    plt.xlabel("Batch Index")
                    plt.ylabel("Uncertainty (Std)")
                    plt.title("Predicted Uncertainty Over Batches")
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plt.show()

                    '''误差条图'''

                    pred_mean = pred_mean_output.cpu().numpy().flatten()

                    # 将pred_std 展平
                    # pred_std = pred_std.flatten()
                    pred_std = pred_std_output.flatten()[:len(pred_mean)]

                    # 创建 x 轴数据（批次索引）
                    batch_indices = np.arange(len(pred_mean))

                    #绘制误差条图
                    plt.figure(figsize=(10, 6))

                    plt.errorbar(batch_indices, pred_mean, yerr=pred_std, fmt='o', color='blue', ecolor='lightblue',
                             capsize=5, label='Predicted Mean ± Std')

                    # 设置图形属性
                    plt.xlabel("Batch Index")
                    plt.ylabel("Predicted Confidence")
                    plt.title("Predicted Confidence with Error Bars")
                    plt.legend()
                    plt.grid(True)
                    plt.show()
                #outputs[final_metric] = np.round(outputs[final_metric], 6)

            # elif metric in link_list_high_score_filtered:
            #     final_metric = "_".join([mode, metric])
            #     outputs[final_metric] = np.around(np.array([o[metric] for o in results]).sum() / count_for_link, decimals=3).item()
            elif metric in link_list_raw:
                final_metric = "_".join([mode, metric])
                outputs[final_metric] = np.around(np.array([o[metric] for o in results]).sum() / count_for_link,
                                                  decimals=6).item()
            else:
                outputs["_".join([mode, 'wmr'])] = np.around(
                    np.array([o['wmr'].cpu() for o in results]).sum() / conf_sum, decimals=6).item()
                outputs["_".join([mode, 'wmrr'])] = np.around(
                    np.array([o['wmrr'].cpu() for o in results]).sum() / conf_sum, decimals=6).item()
                outputs["_".join([mode, 'raw_wmr'])] = np.around(
                    np.array([o['raw_wmr'].cpu() for o in results]).sum() / conf_sum, decimals=6).item()
                outputs["_".join([mode, 'raw_wmrr'])] = np.around(
                    np.array([o['raw_wmrr'].cpu() for o in results]).sum() / conf_sum, decimals=6).item()

        return outputs

    def convert_semi_samples_batch(self, neg_samples, neg_scores, pos_samples, mode, device):
        # Calculate the total number of negative samples
        num_neg_samples = len(pos_samples) * len(neg_samples[0])
        # Duplicate the positive sample so that it can be modified
        semi_samples_formatted = pos_samples.repeat(1, len(neg_samples[0])).view(-1, 4).to(device)
        # Select whether to replace the head or tail entities based on the mode
        if mode == "head-batch":
            neg_samples_formatted = neg_samples.view(-1).long().cpu().numpy()
            semi_samples_formatted[:, 0] = torch.tensor(neg_samples_formatted, dtype=torch.double)
        elif mode == "tail-batch":
            neg_samples_formatted = neg_samples.view(-1).long().cpu().numpy()
            semi_samples_formatted[:, 2] = torch.tensor(neg_samples_formatted, dtype=torch.double)
        else:
            raise ValueError("Invalid mode. Use 'head-batch' or 'tail-batch'.")
        # Expands the score to the same shape as the negative sample size
        neg_scores_expanded = neg_scores.view(-1).to(device)
        # Add the score as a new fourth element
        semi_samples_formatted = torch.cat([semi_samples_formatted[:, 0:3], neg_scores_expanded.unsqueeze(1)], dim=1)
        return semi_samples_formatted

    def convert_neg_samples_batch(self, pos_samples, neg_samples, mode):
        # Obtain the number of positive and negative samples
        num_pos_samples = len(pos_samples)
        num_neg_samples_per_pos = len(neg_samples[0])
        num_neg_samples = num_pos_samples * num_neg_samples_per_pos
        # Replicate positive samples at one time
        pos_samples_formatted = pos_samples.repeat(1, num_neg_samples_per_pos).view(-1, 4)
        # Choose whether to replace the head or tail entity based on the mode
        if mode == "head-batch" or mode == "head_predict":
            neg_samples_formatted = neg_samples.view(-1).long()
            pos_samples_formatted[:, 0] = neg_samples_formatted.float()
        elif mode == "tail-batch" or mode == "tail_predict":
            neg_samples_formatted = neg_samples.view(-1).long()
            pos_samples_formatted[:, 2] = neg_samples_formatted.float()
        else:
            raise ValueError("Invalid mode. Use 'head-batch', 'tail-batch', 'head_predict', or 'tail_predict'.")
        # The score is set to 0
        pos_samples_formatted[:, 3] = 0.0
        return pos_samples_formatted

    def select_random_elements(self, tensor, n):
        # Gets the length of the tensor
        num_elements = len(tensor)
        # Generate random indexes
        random_indices = torch.randperm(num_elements)[:n]
        # Elements are selected based on a random index
        selected_elements = tensor[random_indices]

        return selected_elements

    def head_batch(self, h, r, t, neg_size=None):
        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.corrupt_head(t, r, num_max=(neg_size - neg_cur_size) * 2)
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]

    def corrupt_head(self, t, r, num_max=1):
        tmp = torch.randint(low=0, high=self.args.num_ent, size=(num_max,)).numpy()
        if not self.args.filter_flag:
            return tmp
        mask = np.in1d(tmp, self.rt2h_train[(r, t)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    def tail_batch(self, h, r, t, neg_size=None):
        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.corrupt_tail(h, r, num_max=(neg_size - neg_cur_size) * 2)
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]

    def corrupt_tail(self, h, r, num_max=1):
        tmp = torch.randint(low=0, high=self.args.num_ent, size=(num_max,)).numpy()
        if not self.args.filter_flag:
            return tmp
        mask = np.in1d(tmp, self.hr2t_train[(h, r)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg
