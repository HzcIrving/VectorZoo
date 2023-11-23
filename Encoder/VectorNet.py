#-*- coding: utf-8 -*-
"""
VectorNet-Decoder    

分层架构
- Polyline Subgraphs  
    -- Node Feature (ds,de,ai,j) (start_pt, end_pt, attribute features, Polyline index)
         ↓
- Global Interaction Graph  

"""
import torch 
import torch.nn as nn 






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