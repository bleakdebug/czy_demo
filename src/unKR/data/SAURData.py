import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
from collections import defaultdict
import numpy as np
import pytorch_lightning as pl
from unKR.data.Sampler import RandomSampler


class SAURDataset(Dataset):
    def __init__(self, triples=None, targets=None, edge_index=None, graph=None, args=None, train_sampler=None, test_sampler=None):
        super().__init__()
        self.triples = triples
        self.targets = targets
        self.edge_index = edge_index
        self.graph = graph
        self.args = args
        
        # 如果提供了args和采样器，可以用于初始化其他属性
        if args:
            self.train_sampler = train_sampler if train_sampler else RandomSampler(args)
            self.test_sampler = test_sampler if test_sampler else RandomSampler(args)
            self.num_ent = args.num_ent if hasattr(args, 'num_ent') else 15000
            self.num_rel = args.num_rel if hasattr(args, 'num_rel') else 36
        
    def traverse_triplets(self, triplets):
        """根据三元组构建图
        
        为每个实体创建其邻居实体列表
        
        Args:
            triplets: 三元组列表 [h, r, t]
            
        Returns:
            graph: 包含每个实体邻居的字典
        """
        graph = defaultdict(list)
        # 构建图
        for entity1, relation, entity2 in triplets:
            # 将实体ID转换为整数
            if isinstance(entity1, torch.Tensor):
                entity1 = entity1.item()
            if isinstance(entity2, torch.Tensor):
                entity2 = entity2.item()
            
            graph[entity1].append(entity2)
            graph[entity2].append(entity1)
        return graph
        
    def setup(self, stage=None):
        # 从TSV文件加载数据
        try:
            train_df = pd.read_csv(f"{self.args.data_path}/train.tsv", sep='\t', header=None)
            valid_df = pd.read_csv(f"{self.args.data_path}/valid.tsv", sep='\t', header=None)
            test_df = pd.read_csv(f"{self.args.data_path}/test.tsv", sep='\t', header=None)
            
            # 应用数据比例限制
            ratio = getattr(self.args, 'ratio', 0.018)
            train_size = int(len(train_df) * ratio)
            valid_size = int(len(valid_df) * ratio)
            test_size = int(len(test_df) * ratio)
            
            # 随机选择指定比例的数据
            train_df = train_df.sample(n=train_size, random_state=42)
            valid_df = valid_df.sample(n=valid_size, random_state=42)
            test_df = test_df.sample(n=test_size, random_state=42)
            
            print(f"应用比例 {ratio} 后，训练集: {len(train_df)}，验证集: {len(valid_df)}，测试集: {len(test_df)}")
        except Exception as e:
            print(f"数据加载失败: {e}")
            # 创建空DataFrame作为备用
            train_df = pd.DataFrame(columns=[0, 1, 2, 3])
            valid_df = pd.DataFrame(columns=[0, 1, 2, 3])
            test_df = pd.DataFrame(columns=[0, 1, 2, 3])
            
        # 获取设备信息
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        # 转换为张量
        train_triples = torch.tensor(train_df.iloc[:, :3].values, dtype=torch.long, device=device)
        if 3 in train_df.columns:
            train_conf = torch.tensor(train_df[3].values, dtype=torch.float, device=device)
        else:
            train_conf = torch.ones(len(train_df), device=device)
            
        valid_triples = torch.tensor(valid_df.iloc[:, :3].values, dtype=torch.long, device=device)
        if 3 in valid_df.columns:
            valid_conf = torch.tensor(valid_df[3].values, dtype=torch.float, device=device)
        else:
            valid_conf = torch.ones(len(valid_df), device=device)
            
        test_triples = torch.tensor(test_df.iloc[:, :3].values, dtype=torch.long, device=device)
        if 3 in test_df.columns:
            test_conf = torch.tensor(test_df[3].values, dtype=torch.float, device=device)
        else:
            test_conf = torch.ones(len(test_df), device=device)
        
        # 构建图结构
        if self.graph is None:
            # 合并所有三元组用于构建完整图结构
            all_triples = pd.concat([train_df, valid_df, test_df]).iloc[:, :3].values
            self.graph = self.traverse_triplets(all_triples)
            print(f"已构建图结构，包含 {len(self.graph)} 个实体")
        
        # 创建数据集 - 为每个数据集创建匹配大小的边索引
        # 训练集
        train_edge_index = torch.ones((len(train_triples), 2), dtype=torch.long, device=device)
        for i in range(len(train_triples)):
            train_edge_index[i, 0] = train_triples[i, 0]  # 头实体
            train_edge_index[i, 1] = train_triples[i, 2]  # 尾实体
        self.train_dataset = TensorDataset(train_triples, train_conf, train_edge_index)
        
        # 验证集
        valid_edge_index = torch.ones((len(valid_triples), 2), dtype=torch.long, device=device)
        for i in range(len(valid_triples)):
            valid_edge_index[i, 0] = valid_triples[i, 0]  # 头实体
            valid_edge_index[i, 1] = valid_triples[i, 2]  # 尾实体
        self.valid_dataset = TensorDataset(valid_triples, valid_conf, valid_edge_index)
        
        # 测试集
        test_edge_index = torch.ones((len(test_triples), 2), dtype=torch.long, device=device)
        for i in range(len(test_triples)):
            test_edge_index[i, 0] = test_triples[i, 0]  # 头实体
            test_edge_index[i, 1] = test_triples[i, 2]  # 尾实体
        self.test_dataset = TensorDataset(test_triples, test_conf, test_edge_index)
        
    def __len__(self):
        if self.triples is not None:
            return len(self.triples)
        elif hasattr(self, 'train_dataset'):
            return len(self.train_dataset)
        return 0
        
    def __getitem__(self, idx):
        if self.triples is not None and self.targets is not None and self.edge_index is not None:
            return {
                'positive_sample': self.triples[idx],
                'target': self.targets[idx],
                'edge_index': self.edge_index[idx],
                'graph': self.graph
            }
        elif hasattr(self, 'train_dataset'):
            return {
                'positive_sample': self.train_dataset[idx][0],
                'target': self.train_dataset[idx][1],
                'edge_index': self.train_dataset[idx][2],
                'graph': self.graph
            }
        # 默认情况
        return {
            'positive_sample': torch.zeros(3, dtype=torch.long),
            'target': torch.tensor(0.0),
            'edge_index': torch.zeros(2, dtype=torch.long),
            'graph': self.graph if self.graph else {}
        }
'''
class SAURDataset(Dataset):
    def __init__(self, triples=None, targets=None, edge_index=None, graph=None, args=None, train_sampler=None,
                 test_sampler=None):
        super().__init__()
        # 保持数据在CPU上
        self.triples = triples
        self.targets = targets
        self.edge_index = edge_index
        self.graph = graph
        self.args = args

        # 如果提供了args和采样器，可以用于初始化其他属性
        if args:
            self.train_sampler = train_sampler if train_sampler else RandomSampler(args)
            self.test_sampler = test_sampler if test_sampler else RandomSampler(args)
            self.num_ent = args.num_ent if hasattr(args, 'num_ent') else 15000
            self.num_rel = args.num_rel if hasattr(args, 'num_rel') else 36

    def __len__(self):
        if self.triples is not None:
            return len(self.triples)
        elif hasattr(self, 'train_dataset'):
            return len(self.train_dataset)
        return 0

    def __getitem__(self, idx):
        if self.triples is not None and self.targets is not None and self.edge_index is not None:
            # 创建基本数据字典，保持在CPU上
            batch = {
                'positive_sample': self.triples[idx],
                'target': self.targets[idx],
                'edge_index': self.edge_index[idx],
                'graph': self.graph
            }

            # 创建头实体和尾实体的标签矩阵，保持在CPU上
            head_label = torch.zeros(self.num_ent)
            tail_label = torch.zeros(self.num_ent)

            # 获取当前三元组的头尾实体
            if isinstance(self.triples[idx], torch.Tensor):
                head_idx = self.triples[idx][0].item()
                tail_idx = self.triples[idx][2].item()
            else:
                head_idx = self.triples[idx][0]
                tail_idx = self.triples[idx][2]

            # 设置正样本的位置为1
            head_label[head_idx] = 1
            tail_label[tail_idx] = 1

            # 添加标签到batch
            batch['head_label'] = head_label
            batch['tail_label'] = tail_label

            return batch

        elif hasattr(self, 'train_dataset'):
            # 如果是从训练数据集获取
            triple = self.train_dataset[idx][0]
            target = self.train_dataset[idx][1]
            edge_index = self.train_dataset[idx][2]

            # 创建头实体和尾实体的标签
            head_label = torch.zeros(self.num_ent)
            tail_label = torch.zeros(self.num_ent)
            head_label[triple[0]] = 1
            tail_label[triple[2]] = 1

            return {
                'positive_sample': triple,
                'target': target,
                'edge_index': edge_index,
                'graph': self.graph,
                'head_label': head_label,
                'tail_label': tail_label
            }

        # 默认返回值
        return {
            'positive_sample': torch.zeros(3, dtype=torch.long),
            'target': torch.tensor(0.0),
            'edge_index': torch.zeros(2, dtype=torch.long),
            'graph': self.graph if self.graph else {},
            'head_label': torch.zeros(self.num_ent),
            'tail_label': torch.zeros(self.num_ent)
        }

    def setup(self, stage=None):
        """从TSV文件加载数据"""
        try:
            # 加载数据
            train_df = pd.read_csv(f"{self.args.data_path}/train.tsv", sep='\t', header=None)
            valid_df = pd.read_csv(f"{self.args.data_path}/valid.tsv", sep='\t', header=None)
            test_df = pd.read_csv(f"{self.args.data_path}/test.tsv", sep='\t', header=None)

            # 应用数据比例限制
            ratio = getattr(self.args, 'ratio', 0.018)
            train_size = int(len(train_df) * ratio)
            valid_size = int(len(valid_df) * ratio)
            test_size = int(len(test_df) * ratio)

            # 转换为张量，保持在CPU上
            self.train_triples = torch.tensor(train_df.iloc[:, :3].values, dtype=torch.long)
            self.valid_triples = torch.tensor(valid_df.iloc[:, :3].values, dtype=torch.long)
            self.test_triples = torch.tensor(test_df.iloc[:, :3].values, dtype=torch.long)

            # 处理置信度值，保持在CPU上
            self.train_targets = torch.tensor(train_df[3].values, dtype=torch.float) if len(
                train_df.columns) > 3 else torch.ones(len(train_df))
            self.valid_targets = torch.tensor(valid_df[3].values, dtype=torch.float) if len(
                valid_df.columns) > 3 else torch.ones(len(valid_df))
            self.test_targets = torch.tensor(test_df[3].values, dtype=torch.float) if len(
                test_df.columns) > 3 else torch.ones(len(test_df))

            # 创建边索引，保持在CPU上
            self.train_edge_index = self._create_edge_index(self.train_triples)
            self.valid_edge_index = self._create_edge_index(self.valid_triples)
            self.test_edge_index = self._create_edge_index(self.test_triples)

            # 构建图结构
            if self.graph is None:
                self.graph = self.build_graph()

            print(f"数据加载完成 - 训练集: {len(train_df)}，验证集: {len(valid_df)}，测试集: {len(test_df)}")

        except Exception as e:
            print(f"数据加载错误: {e}")
            # 创建空数据，保持在CPU上
            self.train_triples = torch.tensor([], dtype=torch.long)
            self.valid_triples = torch.tensor([], dtype=torch.long)
            self.test_triples = torch.tensor([], dtype=torch.long)
            self.train_targets = torch.tensor([])
            self.valid_targets = torch.tensor([])
            self.test_targets = torch.tensor([])
            self.train_edge_index = torch.zeros((2, 0), dtype=torch.long)
            self.valid_edge_index = torch.zeros((2, 0), dtype=torch.long)
            self.test_edge_index = torch.zeros((2, 0), dtype=torch.long)
            self.graph = {}

    def _create_edge_index(self, triples):
        """创建边索引"""
        edge_index = torch.zeros((2, len(triples)), dtype=torch.long)
        edge_index[0] = triples[:, 0]  # 头实体
        edge_index[1] = triples[:, 2]  # 尾实体
        return edge_index
'''
class SAURDataModule(pl.LightningDataModule):
    def __init__(self, args, train_sampler, test_sampler):
        super().__init__()
        self.args = args
        self.train_sampler = train_sampler
        self.test_sampler = test_sampler
        self.data_path = args.data_path
        
    def setup(self, stage=None):
        """加载和预处理数据"""
        try:
            # 加载训练集
            train_data = pd.read_csv(f"{self.data_path}/train.tsv", sep='\t', header=None)
            # 应用数据比例限制
            ratio = getattr(self.args, 'ratio', 0.018)
            train_size = int(len(train_data) * ratio)
            train_data = train_data.sample(n=train_size, random_state=42)
            
            # 检查数据类型，如果包含字符串，创建实体和关系的映射
            if train_data.iloc[:, :3].dtypes.apply(lambda x: x == 'object').any():
                print("检测到字符串类型的实体或关系，创建ID映射...")
                # 提取所有实体和关系
                entities = set()
                relations = set()
                
                # 从训练集提取
                for _, row in train_data.iterrows():
                    entities.add(row[0])  # 头实体
                    relations.add(row[1])  # 关系
                    entities.add(row[2])  # 尾实体
                
                # 从验证集提取
                valid_data = pd.read_csv(f"{self.data_path}/valid.tsv", sep='\t', header=None)
                valid_size = int(len(valid_data) * ratio)
                valid_data = valid_data.sample(n=valid_size, random_state=42)
                
                for _, row in valid_data.iterrows():
                    entities.add(row[0])
                    relations.add(row[1])
                    entities.add(row[2])
                
                # 从测试集提取
                test_data = pd.read_csv(f"{self.data_path}/test.tsv", sep='\t', header=None)
                test_size = int(len(test_data) * ratio)
                test_data = test_data.sample(n=test_size, random_state=42)
                
                for _, row in test_data.iterrows():
                    entities.add(row[0])
                    relations.add(row[1])
                    entities.add(row[2])
                
                # 创建映射
                self.entity_to_id = {entity: idx for idx, entity in enumerate(entities)}
                self.relation_to_id = {relation: idx for idx, relation in enumerate(relations)}
                self.id_to_entity = {idx: entity for entity, idx in self.entity_to_id.items()}
                self.id_to_relation = {idx: relation for relation, idx in self.relation_to_id.items()}
                
                print(f"创建了 {len(self.entity_to_id)} 个实体和 {len(self.relation_to_id)} 个关系的ID映射")
                
                # 转换训练集
                train_triples_array = np.zeros((len(train_data), 3), dtype=np.int64)
                for i, (_, row) in enumerate(train_data.iterrows()):
                    train_triples_array[i, 0] = self.entity_to_id[row[0]]
                    train_triples_array[i, 1] = self.relation_to_id[row[1]]
                    train_triples_array[i, 2] = self.entity_to_id[row[2]]
                
                # 转换验证集
                valid_triples_array = np.zeros((len(valid_data), 3), dtype=np.int64)
                for i, (_, row) in enumerate(valid_data.iterrows()):
                    valid_triples_array[i, 0] = self.entity_to_id[row[0]]
                    valid_triples_array[i, 1] = self.relation_to_id[row[1]]
                    valid_triples_array[i, 2] = self.entity_to_id[row[2]]
                
                # 转换测试集
                test_triples_array = np.zeros((len(test_data), 3), dtype=np.int64)
                for i, (_, row) in enumerate(test_data.iterrows()):
                    test_triples_array[i, 0] = self.entity_to_id[row[0]]
                    test_triples_array[i, 1] = self.relation_to_id[row[1]]
                    test_triples_array[i, 2] = self.entity_to_id[row[2]]
                
                # 更新num_ent和num_rel
                self.args.num_ent = len(self.entity_to_id)
                self.args.num_rel = len(self.relation_to_id)
                
                # 确保采样器使用正确的实体数量
                self.train_sampler.num_ent = self.args.num_ent
                self.test_sampler.num_ent = self.args.num_ent
                
                print(f"更新后的实体数量: {self.args.num_ent}, 关系数量: {self.args.num_rel}")
                print(f"更新采样器实体数量: {self.train_sampler.num_ent}")
                
            else:
                # 如果已经是数值型，直接转换
                train_triples_array = train_data.iloc[:, :3].values.astype(np.int64)
                
                # 加载验证集
                valid_data = pd.read_csv(f"{self.data_path}/valid.tsv", sep='\t', header=None)
                valid_size = int(len(valid_data) * ratio)
                valid_data = valid_data.sample(n=valid_size, random_state=42)
                valid_triples_array = valid_data.iloc[:, :3].values.astype(np.int64)
                
                # 加载测试集
                test_data = pd.read_csv(f"{self.data_path}/test.tsv", sep='\t', header=None)
                test_size = int(len(test_data) * ratio)
                test_data = test_data.sample(n=test_size, random_state=42)
                test_triples_array = test_data.iloc[:, :3].values.astype(np.int64)
            
            self.train_triples = torch.tensor(train_triples_array)
            
            # 如果有第4列（置信度），也加载它
            if train_data.shape[1] > 3:
                self.train_targets = torch.tensor(train_data.iloc[:, 3].values, dtype=torch.float32)
            else:
                # 如果没有置信度列，创建全1向量
                self.train_targets = torch.ones(len(train_data), dtype=torch.float32)
                
            # 创建匹配大小的边索引
            self.train_edge_index = torch.zeros((len(self.train_triples), 2), dtype=torch.long)
            for i in range(len(self.train_triples)):
                self.train_edge_index[i, 0] = self.train_triples[i, 0]  # 头实体
                self.train_edge_index[i, 1] = self.train_triples[i, 2]  # 尾实体
            
            self.valid_triples = torch.tensor(valid_triples_array)
            
            if valid_data.shape[1] > 3:
                self.valid_targets = torch.tensor(valid_data.iloc[:, 3].values, dtype=torch.float32)
            else:
                self.valid_targets = torch.ones(len(valid_data), dtype=torch.float32)
                
            # 创建匹配大小的边索引
            self.valid_edge_index = torch.zeros((len(self.valid_triples), 2), dtype=torch.long)
            for i in range(len(self.valid_triples)):
                self.valid_edge_index[i, 0] = self.valid_triples[i, 0]  # 头实体
                self.valid_edge_index[i, 1] = self.valid_triples[i, 2]  # 尾实体
            
            self.test_triples = torch.tensor(test_triples_array)
            
            if test_data.shape[1] > 3:
                self.test_targets = torch.tensor(test_data.iloc[:, 3].values, dtype=torch.float32)
            else:
                self.test_targets = torch.ones(len(test_data), dtype=torch.float32)
                
            # 创建匹配大小的边索引
            self.test_edge_index = torch.zeros((len(self.test_triples), 2), dtype=torch.long)
            for i in range(len(self.test_triples)):
                self.test_edge_index[i, 0] = self.test_triples[i, 0]  # 头实体
                self.test_edge_index[i, 1] = self.test_triples[i, 2]  # 尾实体
            
            # 构建图
            self.graph = self.build_graph()
            print(f"应用比例 {ratio} 后，训练集: {len(train_data)}，验证集: {len(valid_data)}，测试集: {len(test_data)}")
            
        except Exception as e:
            print(f"数据加载错误: {str(e)}")
            raise e
        
    def build_graph(self):
        """构建图结构"""
        graph = defaultdict(list)
        try:
            # 如果存在实体ID映射，使用映射后的ID构建图
            if hasattr(self, 'entity_to_id'):
                print("使用实体ID映射构建图...")
                # 尝试从数据目录加载data.tsv
                try:
                    all_data = pd.read_csv(f"{self.data_path}/data.tsv", sep='\t', header=None)
                    print(f"从{self.data_path}/data.tsv加载数据构建图")
                except:
                    # 如果data.tsv不存在，合并训练、验证和测试集
                    print(f"无法找到data.tsv，使用训练、验证和测试集合并构建图")
                    train_data = pd.read_csv(f"{self.data_path}/train.tsv", sep='\t', header=None)
                    valid_data = pd.read_csv(f"{self.data_path}/valid.tsv", sep='\t', header=None)
                    test_data = pd.read_csv(f"{self.data_path}/test.tsv", sep='\t', header=None)
                    all_data = pd.concat([train_data, valid_data, test_data])
                
                # 使用映射的ID构建图
                for _, row in all_data.iterrows():
                    if row[0] in self.entity_to_id and row[2] in self.entity_to_id:
                        entity1 = self.entity_to_id[row[0]]
                        entity2 = self.entity_to_id[row[2]]
                        graph[entity1].append(entity2)
                        graph[entity2].append(entity1)
                
                print(f"图构建完成，共{len(graph)}个实体")
                return graph
            
            # 否则尝试直接加载数值型数据
            # 首先尝试从数据目录加载data.tsv
            try:
                all_data = pd.read_csv(f"{self.data_path}/data.tsv", sep='\t', header=None)
                print(f"从{self.data_path}/data.tsv加载数据构建图")
            except:
                # 尝试其他可能的路径
                try:
                    all_data = pd.read_csv(f"dataset/cn15k/data.tsv", sep='\t', header=None)
                    print(f"从dataset/cn15k/data.tsv加载数据构建图")
                except:
                    # 如果data.tsv不存在，合并训练、验证和测试集
                    print(f"无法找到data.tsv，使用训练、验证和测试集合并构建图")
                    train_data = pd.read_csv(f"{self.data_path}/train.tsv", sep='\t', header=None)
                    valid_data = pd.read_csv(f"{self.data_path}/valid.tsv", sep='\t', header=None)
                    test_data = pd.read_csv(f"{self.data_path}/test.tsv", sep='\t', header=None)
                    all_data = pd.concat([train_data, valid_data, test_data])
                
            # 检查数据类型，判断是否为数值型
            if all_data.iloc[:, :3].dtypes.apply(lambda x: x != 'object').all():
                for _, row in all_data.iterrows():
                    entity1, _, entity2 = row[:3]
                    # 确保实体ID是整数
                    entity1 = int(entity1)
                    entity2 = int(entity2)
                    graph[entity1].append(entity2)
                    graph[entity2].append(entity1)
                    
                print(f"图构建完成，共{len(graph)}个实体")
            else:
                print("数据包含非数值类型，无法直接构建图。请先创建实体ID映射。")
                
        except Exception as e:
            print(f"图构建错误: {str(e)}")
            print("创建一个空图结构...")
            
        return graph
        
    def train_dataloader(self):
        """创建并返回训练数据加载器"""
        train_dataset = SAURDataset(
            triples=self.train_triples,
            targets=self.train_targets,
            edge_index=self.train_edge_index,
            graph=self.graph,
            args=self.args,
            train_sampler=self.train_sampler,
            test_sampler=self.test_sampler
        )
        
        # 如果我们创建了实体和关系映射，确保sampler知道实体数量
        if hasattr(self, 'entity_to_id'):
            # 更新sampler中的实体数量
            self.train_sampler.num_ent = len(self.entity_to_id)
            self.test_sampler.num_ent = len(self.entity_to_id)
            print(f"更新采样器的实体数量: {self.train_sampler.num_ent}")
        
        return DataLoader(
            train_dataset,
            batch_size=self.args.train_bs,
            shuffle=self.args.shuffle,
            num_workers=self.args.num_workers,
            pin_memory=torch.cuda.is_available()
            #pin_memory=True
        )
        
    def val_dataloader(self):
        """创建并返回验证数据加载器"""
        valid_dataset = SAURDataset(
            triples=self.valid_triples,
            targets=self.valid_targets,
            edge_index=self.valid_edge_index,
            graph=self.graph,
            args=self.args,
            train_sampler=self.train_sampler,
            test_sampler=self.test_sampler
        )
        
        return DataLoader(
            valid_dataset,
            batch_size=self.args.eval_bs,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=torch.cuda.is_available()
            #pin_memory=True
        )
        
    def test_dataloader(self):
        """创建并返回测试数据加载器"""
        test_dataset = SAURDataset(
            triples=self.test_triples,
            targets=self.test_targets,
            edge_index=self.test_edge_index,
            graph=self.graph,
            args=self.args,
            train_sampler=self.train_sampler,
            test_sampler=self.test_sampler
        )
        
        return DataLoader(
            test_dataset,
            batch_size=self.args.eval_bs,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=torch.cuda.is_available()
            #pin_memory=True
        )