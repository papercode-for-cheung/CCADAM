import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.inits import glorot
from torch_scatter import scatter
from torch_geometric.utils import degree
from torch_geometric.utils import softmax, degree

class SubstructureInteraction(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        # 用于计算子结构相互作用强度的 MLP
        self.interaction_mlp = nn.Sequential(
            nn.Linear(self.in_channels * 2, self.in_channels),
            nn.PReLU(),
            nn.Linear(self.in_channels, 1)
        )

    def forward(self, z_1, z_2):
        """
        计算子结构间的相互作用强度，并选择正负样例
        :param z_1: 第一个图的子结构特征 (num_substructures_1, in_channels)
        :param z_2: 第二个图的子结构特征 (num_substructures_2, in_channels)
        :return:
            - pos_pairs: 正样例对，形状为 (2, num_pos_pairs)
            - neg_pairs: 负样例对，形状为 (2, num_neg_pairs)
            - interaction_scores: 子结构间的相互作用强度矩阵 (num_substructures_1, num_substructures_2)
        """
        # 1. 计算子结构间的相互作用强度
        interaction_scores = self.compute_interaction_scores(z_1, z_2)  # (num_substructures_1, num_substructures_2)

        # 2. 选择作用关系最强的两个子结构对作为正样例
        pos_pairs = self.select_positive_pairs(interaction_scores)

        # 3. 其他子结构对作为负样例
        neg_pairs = self.select_negative_pairs(interaction_scores, pos_pairs)

        return pos_pairs, neg_pairs, interaction_scores

    def compute_interaction_scores(self, z_1, z_2):
        """
        计算子结构间的相互作用强度
        :param z_1: 第一个图的子结构特征 (num_substructures_1, in_channels)
        :param z_2: 第二个图的子结构特征 (num_substructures_2, in_channels)
        :return: 相互作用强度矩阵 (num_substructures_1, num_substructures_2)
        """
        num_substructures_1 = z_1.size(0)
        num_substructures_2 = z_2.size(0)

        # 扩展 z_1 和 z_2 以计算所有子结构对的相互作用强度
        z_1_expanded = z_1.unsqueeze(1).expand(-1, num_substructures_2, -1)  # (num_substructures_1, num_substructures_2, in_channels)
        z_2_expanded = z_2.unsqueeze(0).expand(num_substructures_1, -1, -1)  # (num_substructures_1, num_substructures_2, in_channels)

        # 拼接特征并计算相互作用强度
        interaction_input = torch.cat([z_1_expanded, z_2_expanded], dim=-1)  # (num_substructures_1, num_substructures_2, in_channels * 2)
        interaction_scores = self.interaction_mlp(interaction_input).squeeze(-1)  # (num_substructures_1, num_substructures_2)

        return interaction_scores

    def select_positive_pairs(self, interaction_scores):
        """
        选择作用关系最强的两个子结构对作为正样例
        :param interaction_scores: 相互作用强度矩阵 (num_substructures_1, num_substructures_2)
        :return: 正样例对，形状为 (2, num_pos_pairs)
        """
        # 找到相互作用强度最大的两个子结构对
        topk_values, topk_indices = torch.topk(interaction_scores.flatten(), k=2)
        pos_pairs = torch.stack([
            topk_indices // interaction_scores.size(1),  # 子结构 1 的索引
            topk_indices % interaction_scores.size(1)     # 子结构 2 的索引
        ], dim=0)  # (2, num_pos_pairs)

        return pos_pairs

    def select_negative_pairs(self, interaction_scores, pos_pairs):
        """
        选择其他子结构对作为负样例
        :param interaction_scores: 相互作用强度矩阵 (num_substructures_1, num_substructures_2)
        :param pos_pairs: 正样例对，形状为 (2, num_pos_pairs)
        :return: 负样例对，形状为 (2, num_neg_pairs)
        """
        num_substructures_1 = interaction_scores.size(0)
        num_substructures_2 = interaction_scores.size(1)

        device = interaction_scores.device

        # 生成所有可能的子结构对
        all_pairs = torch.cartesian_prod(
            torch.arange(num_substructures_1, device=device),
            torch.arange(num_substructures_2, device=device)
        ).t()  # (2, num_substructures_1 * num_substructures_2)

        # 排除正样例对
        pos_mask = (all_pairs.unsqueeze(-1) == pos_pairs.unsqueeze(1)).all(dim=0).any(dim=-1)
        neg_pairs = all_pairs[:, ~pos_mask]  # (2, num_neg_pairs)

        return neg_pairs


class Interactions(nn.Module):
    def __init__(self, in_channels, temperature=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.temperature = temperature  # 温度参数，用于对比学习

        # 子结构相互作用模块
        self.substructure_interaction = SubstructureInteraction(in_channels)

        # 投影头，用于对比学习
        self.projection_head = nn.Sequential(
            nn.Linear(self.in_channels, self.in_channels),
            nn.PReLU(),
            nn.Linear(self.in_channels, self.in_channels)
        )

        # 用于处理节点特征的 MLP
        self.mlp = nn.Sequential(
            nn.PReLU(),
            nn.Linear(self.in_channels * 2, self.in_channels * 2),
            nn.PReLU(),
            nn.Linear(self.in_channels * 2, self.in_channels),
            nn.PReLU(),
            nn.Linear(self.in_channels, self.in_channels),
        )

        # 可学习参数
        self.a = nn.Parameter(torch.zeros(in_channels))
        glorot(self.a.view(1, -1))

    def forward(self, x_1, batch_1, x_2, batch_2, inter):
        # 1. 计算全局特征（Readout）
        x_1_readout = global_add_pool(x_1, batch_1)  # (batch_size, in_channels)
        x_2_readout = global_add_pool(x_2, batch_2)  # (batch_size, in_channels)

        # 2. 使用投影头将全局特征映射到对比学习空间
        z_1 = self.projection_head(x_1_readout)  # (batch_size, in_channels)
        z_2 = self.projection_head(x_2_readout)  # (batch_size, in_channels)

        # 3. 计算子结构间的相互作用强度，并选择正负样例
        pos_pairs, neg_pairs, interaction_scores = self.substructure_interaction(z_1, z_2)

        # 4. 计算对比学习损失
        contrastive_loss = self.contrastive_loss(z_1, z_2, pos_pairs, neg_pairs)
        similarity_matrix = F.cosine_similarity(x_1.unsqueeze(1), x_2.unsqueeze(0), dim=-1)
        attention_weights_1_to_2 = F.softmax(similarity_matrix / self.temperature, dim=-1)
        attention_weights_2_to_1 = F.softmax(similarity_matrix.t() / self.temperature, dim=-1)
        x_1 = torch.matmul(attention_weights_1_to_2, x_2)  # (num_nodes_1, in_channels)
        x_2 = torch.matmul(attention_weights_2_to_1, x_1)

        # 5. 保留原始代码中对节点特征的处理逻辑
        d_1 = degree(batch_1, dtype=batch_1.dtype)
        d_2 = degree(batch_2, dtype=batch_2.dtype)
        # x_1_readout = x_1_readout.repeat_interleave(d_2, dim=0)
        # x_2_readout = x_2_readout.repeat_interleave(d_1, dim=0)

        # 6. 处理节点特征（与原始代码一致）
        # x_1_score = torch.cat([x_1, x_2_readout], dim=1)
        # x_2_score = torch.cat([x_2, x_1_readout], dim=1)
        # x_1 = x_1 * torch.sigmoid(self.mlp(x_1_score))
        # x_2 = x_2 * torch.sigmoid(self.mlp(x_2_score))
        #logit = contrastive_loss.expand(batch_1.max() + 1)
        #logit = contrastive_loss
        s_1 = torch.cumsum(d_1, dim=0)
        s_2 = torch.cumsum(d_2, dim=0)
        ind_1 = torch.cat(
            [torch.arange(i, device=d_1.device).repeat_interleave(j) + (s_1[e - 1] if e else 0) for e, (i, j) in
             enumerate(zip(d_1, d_2))])
        ind_2 = torch.cat([torch.arange(j, device=d_1.device).repeat(i) + (s_2[e - 1] if e else 0) for e, (i, j) in
                           enumerate(zip(d_1, d_2))])
        x_1 = x_1[ind_1]
        x_2 = x_2[ind_2]
        size_1_2 = torch.mul(d_1, d_2)
        inputs = torch.cat((x_1, x_2), 1)
        ans_SSI = (self.a * self.mlp(inputs))
        ans_SSI = (ans_SSI * inter.repeat_interleave(size_1_2, dim=0)).sum(-1)

        batch_ans = torch.arange(inter.shape[0], device=inputs.device).repeat_interleave(size_1_2, dim=0)
        ans = scatter(ans_SSI, batch_ans, reduce='sum', dim=0)

        logit = ans
        # 7. 返回与原始代码一致的输出格式
        return logit, x_1, x_2,contrastive_loss,contrastive_loss.expand(batch_1.max() + 1)

    def contrastive_loss(self, z_1, z_2, pos_pairs, neg_pairs):
        """
        对比学习模块
        :param z_1: 第一个图的子结构特征 (num_substructures_1, in_channels)
        :param z_2: 第二个图的子结构特征 (num_substructures_2, in_channels)
        :param pos_pairs: 正样例对，形状为 (2, num_pos_pairs)
        :param neg_pairs: 负样例对，形状为 (2, num_neg_pairs)
        :return: 对比学习损失
        """
        # 1. 计算正样例对的相似度
        pos_similarity = self.compute_similarity(z_1, z_2, pos_pairs)

        # 2. 计算负样例对的相似度
        neg_similarity = self.compute_similarity(z_1, z_2, neg_pairs)

        # 3. 计算对比学习损失
        pos_loss = -torch.log(torch.sigmoid(pos_similarity / self.temperature)).mean()
        neg_loss = -torch.log(1 - torch.sigmoid(neg_similarity / self.temperature)).mean()
        loss = pos_loss + neg_loss
        return loss

    def compute_similarity(self, z_1, z_2, pairs):
        """
        计算子结构对的相似度
        :param z_1: 第一个图的子结构特征 (num_substructures_1, in_channels)
        :param z_2: 第二个图的子结构特征 (num_substructures_2, in_channels)
        :param pairs: 子结构对，形状为 (2, num_pairs)
        :return: 相似度，形状为 (num_pairs,)
        """
        z_1_selected = z_1[pairs[0]]  # (num_pairs, in_channels)
        z_2_selected = z_2[pairs[1]]  # (num_pairs, in_channels)
        similarity = F.cosine_similarity(z_1_selected, z_2_selected, dim=-1)  # (num_pairs,)
        return similarity