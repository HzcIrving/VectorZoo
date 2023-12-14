import sys 
sys.path.append(".")
import torch 
import torch.nn as nn  
import torch.nn.functional as F 

import numpy as np 

from torch_geometric.data import Data  
from torch_geometric.nn import MessagePassing, max_pool, avg_pool  # Message Aggregator  
from torch_geometric.utils import add_self_loops, remove_self_loops  


from Models.Layers.MLP import MLP

""" 
SubGraph for each cluster 
--- 聚合各个Polyline，分块的子图  

原论文中构建Polyline Subgraphs 
1). 考虑一个Polyline P机器所有节点 v_i, ... v_P，单层的subgraph propagation: 
    v_i^{l+1}  = φ_rel(g_enc(v_i^{l}), φ_{agg}(g_enc(v_j^l))) (聚合个体，和邻居的特征信息，每个邻居的特征信息哪些是最重要的，需要学加权参数W)
    a/. φ_rel --- concat 
    b/. g_enc --- 原论文用的FC + LayerNorm + ReLU  
    c/. φ_{agg} --- 原论文用的maxpooling  

2). 考虑Poly-level Feature，也是通过Maxpooling来聚合多节点信息  
    P = φ_{agg}(v_i^{L_p})
"""  

# TODO 将VectorNet Encoder BasicModule 换成GRU?  or GCN ? 
class SubGraph(nn.Module):
    """ 
    计算all vectors in a polyline, 
    获得 a polyline-level features ... 
    """ 
    def __init__(self, in_chs, num_subgraph_layers=3, hidden_units=64, out_chs=64) -> None:
        super().__init__() 
        
        self.num_subgraph_layers = num_subgraph_layers 
        self.in_chs = in_chs 
        self.hidden_units = hidden_units 
        self.out_chs = out_chs 
        
        # Sequential() 模式 
        self.layer_seq = nn.Sequential()
        for i in range(self.num_subgraph_layers):
            self.layer_seq.add_module(
                f'glp_{i}', MLP(in_channel=in_chs, 
                                hidden=hidden_units,
                                out_channel=out_chs)
            ) 
            
            in_chs = hidden_units * 2 # 逐步扩大 
        
        # MLP中包含了shortcut 
        # for name, layer in self.layer_seq.named_modules():
        #     # print(name)
        #     print(f"name:{name}, layer:{layer}")  
        
        self.linear = nn.Linear(hidden_units*2, hidden_units) 
    
    def forward(self, sub_data):
        """ 
        sub_data get from argoverseLoader   
        ---------------------------------------
        Polyline Vector Set in torch_geometric.data.Data format  
        
        Args: 
            - sub_data(Data): [x, y, cluster, edge_index, valid_len] 
                - x:(N_nodes, 2) Frame中的节点数 (时序obs_len Node + lane_node)
                - y:(N_nodes,)  
                - cluster: (N_nodes, ) 标记类每个Node的类id 
                - edge_index: (2, N_edges)  
                - valid_len: (n_clusters, 2) 2(x,y入口)
        """
        # x 所有bs在一起
        x = sub_data.x  
        sub_data.cluster = sub_data.cluster.long() 
        sub_data.edge_index = sub_data.edge_index.long()  
        
        for name, layer in self.layer_seq.named_modules(): 
            if isinstance(layer, MLP):
                x = layer(x) 
                sub_data.x = x 
                agg_data = max_pool(sub_data.cluster, sub_data)
                
                x = torch.cat([x, agg_data.x[sub_data.cluster]], dim=-1) 
            
        x = self.linear(x) 
        sub_data.x = x  
        out_data = max_pool(sub_data.cluster, sub_data)
        x = out_data.x  
        
        assert x.shape[0] % int(sub_data.time_step_len[0]) == 0  
        
        return F.normalize(x, p=2.0, dim=1) # L2 Norm           
        
        

# if __name__ == "__main__":
#     SubGraph(32)