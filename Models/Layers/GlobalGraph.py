from operator import index
import sys
sys.path.append(".") 
sys.path.append("./Models")

import math 
import numpy as np 

import torch 
import torch.nn as nn 
import torch.nn.functional as F    


from torch_geometric.nn import MessagePassing, max_pool  
from torch_geometric.utils import add_self_loops, degree   
from torch_geometric.data import Data  

# 图注意力机制 
from torch_geometric.nn import GATConv, GATv2Conv, TransformerConv, SuperGATConv

""" 
由SubGraph获得多个Polyline的Node Features - [p1, ..., p_p]  

Global Interaction Graph:
 {p_i^{l+1}} = GNN({p_i^{l}}, A) 
 - A: Adjacancy Matrix  
 - 全连接图假设   

Self-attn Operation 
 - GNN(P) = softmax(PQPK^T)PV  
 - P是node feature matrix 

"""

class GlobalGraph(nn.Module):
    """ 
    Global Graph that compute global information 
    """ 
    def __init__(self, in_chs, global_graph_width, 
                        num_global_layers = 1, 
                        need_scale = False, 
                        with_norm = False, 
                        message_passing = True) -> None:
        super().__init__()  
        
        self.in_chs = in_chs 
        self.global_graph_width = global_graph_width 
        
        self.layers = nn.Sequential() 
        
        for i in range(num_global_layers): 
            if message_passing: 
                self.layers.add_module(
                    f'glp_{i}', SelfAttnGNN(in_chs, self.global_graph_width, need_scale)
                )
            else: # TODO  normal attn 
                self.layers.add_module(
                    f'glp_{i}', SelfAttnFC 
                ) 
            
            in_chs = self.global_graph_width 
            
    def forward(self, global_data, **kwargs): 
        # kwargs： 关键字参数 
        x = global_data.x 
        edge_index = global_data.edge_index   
        
        # number of valid polylines (trajs + other road geometries)
        valid_lens = global_data.valid_lens   
        
        # global 最大的polyline tps 
        time_step_len = int(global_data.time_step_len[0])  
        
        
        for name, layer in self.layers.named_modules():
            if isinstance(layer, SelfAttnGNN):
                x = layer(x, edge_index, valid_lens)
            
            elif isinstance(layer, SelfAttnFC):
                x = layer(x, valid_lens, **kwargs) 
            
            elif isinstance(layer, GATv2Conv):
                x = layer(x, edge_index) # GNN(x, A) 
            
            elif isinstance(layer, TransformerConv):
                x = layer(x, edge_index)  
                
        return x          
    

class SelfAttnGNN(MessagePassing): 
    """ 
    
    Self-Attn Layer, No Scale Factor d_k 
    """
    
    def __init__(self, 
                 in_chs,
                 global_graph_width, 
                 need_scale = False, 
                 with_norm = False
                 ):
        
        # 消息聚合模式 self.aggregate 形式为add 
        super(SelfAttnGNN, self).__init__(aggr='add')  
        
         
        self.in_chs = in_chs 
        self.with_norm = with_norm 
        self.global_graph_width = global_graph_width  
        
        self.q_lin = nn.Linear(in_chs, global_graph_width)  
        self.k_lin = nn.Linear(in_chs, global_graph_width) 
        self.v_lin = nn.Linear(in_chs, global_graph_width) 
        
        self.scale_factor_d = 1 + int(np.sqrt(self.in_channels)) if need_scale else 1 
    
    def forward(self, x, edge_index, valid_len): 
        # x:[node, 2] 
        # edge_index: []
        
        # Attn 
        query = self.q_lin(x) 
        key = self.k_lin(x) 
        value = self.v_lin(x) 
        
        # GNN(P) = softmax(PQPK^T)PV   
        scores = torch.bmm(query, key.transpose(1,2))  
        attn_weights = self.masked_softmax(scores, valid_len) 
        x = torch.bmm(attn_weights, value)  
        
        x = x.view(-1, self.global_graph_width)  # bs*n_nodes, emb
        
        # 信息传递  
        # 包括三个流程 
        # 1. self.message --- 进行邻居信息变换 （e.g.提特征）
        # 2. self.aggregate --- 进行邻居信息聚合 (e.g. add, pooling等) 
        # 3. self.update --- 进行节点信息更新, x_i^k -> x_i^{k+1} 
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
        
    
    def message(self, x_j): 
        
        # 邻居信息变换 
        
        return x_j  
    
    @staticmethod 
    def masked_softmax(X, valid_len): 
        """ 
        Masked Softmax for attn scores 
        Args: 
            X: 3D Tensor (Bs, n_nodes, 2)  
            valid_len: 1D Tensor or 2D Tensor 
        """ 
        
        if valid_len is None:
            return nn.functional.softmax(X, dim=-1) 
        
        else:
            shape = X.shape   
            # if valid_len.dim() == 1: # repeats Node dim
            if valid_len.shape[0] != shape[0]:
                valid_len = torch.repeat_interleave(valid_len, repeats=shape[1], dim=0) 
            else:
                valid_len = valid_len.reshape(-1)  
                
            # Fill Masked Elements with a large negative, whose exp is 0 ---> decrease the influence of masked elements 
            mask = torch.ones_like(X, dtype=torch.bool) 
            for batch_id, cnt in enumerate(valid_len): 
                # :cnt，表示正常部分，cnt~max_len，是填充部分
                mask[batch_id, :cnt] = False    
                
            # True处填充  --- cnt:Max_Len部分 
            X_masked = X.masked_fill(mask, -1e12)   
            return nn.functional.softmax(X_masked, dim=-1) 
        

if __name__ == "__main__":
    # test 
    data = Data(x=torch.tensor([[[1.0, 1.0, 1.0],[7.0, 2.0, 3.5]]]), 
                edge_index=torch.tensor([[0, 1],[1, 0]]), 
                valid_lens=torch.tensor([2]))   # 1, 2, 3 ->  (Bs, N, feature)  
    
    print(data.x.size()) # 1, 2, 3  
    
    layer = SelfAttnGNN(in_chs=3, global_graph_width=32)  
    
    for k, v in layer.state_dict().items():
        print("k:", k, "v:",v.size()) 
        if k.endswith('weight'): # 32, 3 
            v = torch.ones(v.size()) 
        elif k.endswith('bias'):
            v = torch.ones(v.size()) 
        
        
    y = layer(data.x, data.edge_index, data.valid_lens) 
    print(y.shape) 
    
    