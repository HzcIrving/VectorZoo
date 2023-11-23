#-*- coding: utf-8 -*- 
# Basic Model Repo  

import torch 
import torch as th 
import torch.nn as nn 
import torch.nn.functional as F  

class MLP(nn.Module):
    def __init__(self, hidden_size, out_features=None):
        super(MLP, self).__init__()
        if out_features is None:
            out_features = hidden_size  
        self.linear = nn.Linear(hidden_size, out_features)
        self.ln = nn.LayerNorm(out_features)        
        self.relu = nn.ReLU() 
    
    def forward(self, x):
        x = self.linear(x)
        x = self.ln(x)
        x = self.relu(x)
        return x

