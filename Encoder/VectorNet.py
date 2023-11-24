#-*- coding: utf-8 -*-
"""
VectorNet-Decoder    

分层架构
- Polyline Subgraphs  
    -- Node Feature (ds,de,ai,j) (start_pt, end_pt, attribute features, Polyline index)
         ↓
- Global Interaction Graph  

"""
import os
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader, DataListLoader, Batch, Data

from Models.GlobalGraph import GlobalGraph, SelfAttentionFCLayer  
from Models.SubGraph import SubGraph 
from Models.MLP import MLP 


# TODO 参数固化了，需要灵活传参
# Backbone --- 特征提取器，不接MLP Predictor 
class VectorNetBackbone(nn.Module):
    """
    hierarchical GNN with trajectory prediction MLP
    """
    def __init__(self,
                 in_channels=8,
                 num_subgraph_layres=3,
                 num_global_graph_layer=1,
                 subgraph_width=64,
                 global_graph_width=64,
                 aux_mlp_width=64,
                 with_aux: bool = False,
                 device=torch.device("cpu")):
        super(VectorNetBackbone, self).__init__()
        # some params
        self.num_subgraph_layres = num_subgraph_layres
        self.global_graph_width = global_graph_width

        self.device = device

        # subgraph feature extractor
        self.subgraph = SubGraph(in_channels, num_subgraph_layres, subgraph_width)

        # global graph
        self.global_graph = GlobalGraph(self.subgraph.out_channels + 2,
                                        self.global_graph_width,
                                        num_global_layers=num_global_graph_layer)

        # auxiliary recoverey mlp 
        # 辅助任务
        self.with_aux = with_aux
        if self.with_aux:
            self.aux_mlp = nn.Sequential(
                MLP(self.global_graph_width, aux_mlp_width, aux_mlp_width),
                nn.Linear(aux_mlp_width, self.subgraph.out_channels)
            )

    def forward(self, data):
        """
        args:
            data (Data): [x, y, cluster, edge_index, valid_len]
        """
        batch_size = data.num_graphs
        time_step_len = data.time_step_len[0].int()
        valid_lens = data.valid_len

        id_embedding = data.identifier

        sub_graph_out = self.subgraph(data)

        if self.training and self.with_aux:
            randoms = 1 + torch.rand((batch_size,), device=self.device) * (valid_lens - 2) + \
                      time_step_len * torch.arange(batch_size, device=self.device)
            # mask_polyline_indices = [torch.randint(1, valid_lens[i] - 1) + i * time_step_len for i in range(batch_size)]
            mask_polyline_indices = randoms.long()
            aux_gt = sub_graph_out[mask_polyline_indices]
            sub_graph_out[mask_polyline_indices] = 0.0

        # reconstruct the batch global interaction graph data
        x = torch.cat([sub_graph_out, id_embedding], dim=1).view(batch_size, -1, self.subgraph.out_channels + 2)
        valid_lens = data.valid_len

        if self.training:
            # mask out the features for a random subset of polyline nodes
            # for one batch, we mask the same polyline features

            # global_graph_out = self.global_graph(sub_graph_out, batch_size=data.num_graphs)
            global_graph_out = self.global_graph(x, valid_lens=valid_lens)

            if self.with_aux:
                aux_in = global_graph_out.view(-1, self.global_graph_width)[mask_polyline_indices]
                aux_out = self.aux_mlp(aux_in)

                return global_graph_out, aux_out, aux_gt

            return global_graph_out, None, None

        else:
            # global_graph_out = self.global_graph(sub_graph_out, batch_size=data.num_graphs)
            global_graph_out = self.global_graph(x, valid_lens=valid_lens)

            return global_graph_out, None, None







class VectorNet(nn.Module):
    r"""
    sub-graph --- Polyline graph 
        - encode a polyline as a single vector  
        - 抽取Instance Feature（比如每辆车和行人的轨迹特征，lane的特征等）
    
    global-interaction graph --- 不同Node之间的交互关系
        - 长距离交互信息 
        - 相较于CNN，具有排除干扰信息能力（没有edge与ego agent相连)
        - 聚合能力更强, 全局视角
    """