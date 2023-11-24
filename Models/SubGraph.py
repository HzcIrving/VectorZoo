import torch 
import torch as th 
import torch.nn as nn
import torch.nn.functional as F 
import torchsnooper as snooper  

from MLP import MLP  

# TODO 
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, max_pool, avg_pool
from torch_geometric.utils import add_self_loops, remove_self_loops

""" 
- Polyline Subgraph 
    - Self-Attn Mode <DenseTNT> 
    - Graph Mode (torchgeometry) ... 
- Cross Attention 
""" 


# TODO  
class AttnSubgraph(nn.Module): 
    pass  


# @snooper.snoop() # debug mode 
class SubGraph(nn.Module):
    """
    Subgraph -- 
        计算Polyline中所有向量的子图，并获得Polyline级特征 
    
    原论文，第l层的方式: 
    Input Node Feature v(l) -> Node Encoder (MLP) -> Emb v(l) -> ------------------------------> Concat(Relational Operator)--->output v(l+1) -> ... output v(l+2) ... 
                                                              ↓                                     ↑
                                                              -> MaxPooling Aggregator(Emb v(l)) --->
    
    Aggregator --- 聚合所有邻居节点的信息 
    """  
    def __init__(self, in_chs, num_subgraph_layers=3, hidden_unit=64):
        super(SubGraph, self).__init__() 
        self.num_subgraph_layers = num_subgraph_layers # Layer Nums 
        self.hidden_unit = hidden_unit  
        self.out_channels = hidden_unit # Polyline-level-Feature 
        
        self.layer_seq = nn.Sequential()   
        for i in range(self.num_subgraph_layers):
            self.layer_seq.add_module(f"subgraph_layer{i}",MLP(in_chs, hidden_unit))  # f表示格式化字符串，支持format 
            in_chs = hidden_unit * 2  
        
        self.Linear = nn.Linear(hidden_unit*2, hidden_unit) 
    
    def forward(self, sub_data):
        """ 
        Poly Vector Set in 【torch_geometric.data.Data】 Format  
        args: 
            sub_data (Data): [x, y, cluster, edge_index, valid_len] 
            每一个子图的valid_len是对齐的 
            
        关于【torch_geometric.data.Data】
        - x : node feature matrix , shape (num_nodes, num_node_features ) --- (N, NF)  
        - y : GT labels    
        - edge-index: original (2, num_edges) or (num_edges, 2) --- 2:代表源节点、目标节点，为了计算邻接矩阵Adjacent Matrix 
        """
        x = sub_data.x   
        
        # cluster长整型转换 
        # ---------------------------------------
        sub_data.cluster = sub_data.cluster.long()  
        sub_data.edge_index = sub_data.edge_index.long()  
        
        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, MLP):
                x = layer(x) # (Node_nums, Node_features) -> (Node_nums, hidden_unit) 
                sub_data.x = x # Update node_feature 
                agg_data = max_pool(sub_data.cluster, sub_data)

                # 基于max_pool进行聚合 
                x = torch.cat([x, agg_data.x[sub_data.cluster]], dim=-1)

        x = self.linear(x)
        sub_data.x = x
        out_data = max_pool(sub_data.cluster, sub_data)
        x = out_data.x

        assert x.shape[0] % int(sub_data.time_step_len[0]) == 0

        # p --- 2范数 
        # dim = 1 --- 在feature维度上Norm 
        return F.normalize(x, p=2.0, dim=1)  



# TODO  需要修改 
# 测试代码 
import unittest 
import torch_geometric as tg 
import torch_geometric.data as tgd 
class SubGraphTest(unittest.TestCase):
    
    def setUp(self):
        self.num_subgraph_layers = 3
        self.hidden_unit = 64
        self.in_chs = 10
        
        self.edge_feature = 6 
        
        self.sub_data = tgd.Data( 
            # 3节点图 
            # 
            x=torch.randn(3, self.in_chs),  # (3,10) # node_nums 
            y=torch.LongTensor([0,1,2]), # 3类
            cluster=torch.randn(3, self.edge_feature),
            edge_index=torch.tensor([[0, 1, 0],
                                    [1, 2, 0]]),
            valie_len = 9
        )
        
        print("x:",self.sub_data.x.shape)
        print("y:",self.sub_data.y.shape) 
        # print("cluster:",self.sub_data.cluster.shape)
        print("edge_index:",self.sub_data.edge_index.shape) 
        
        # 10, 3, 64
        self.subgraph = SubGraph(self.in_chs, self.num_subgraph_layers, self.hidden_unit)
        
    def test_subgraph_forward(self):
        output = self.subgraph(self.sub_data) 
        print(output.shape)
        # self.assertEqual(output[0].shape, (10, self.subgraph.out_channels))
        # self.assertEqual(output[1].shape, (10, self.subgraph.out_channels)) 

# if __name__ == "__main__":
#     unittest.main()
    
