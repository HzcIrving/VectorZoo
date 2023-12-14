import sys 
sys.path.append(".") 

import os 
import numpy as np 
from tqdm import tqdm 
import matplotlib.pyplot as plt 


import torch
import torch.nn as nn
from torch.optim import Adam, AdamW 

# Graph Data Loader 
from torch_geometric.data import DataLoader
from torch_geometric.nn import DataParallel


# Data Processing Module
from argoverse.evaluation.eval_forecasting import get_displacement_errors_and_miss_rate
from argoverse.evaluation.competition_util import generate_forecasting_h5  

# Apex --- DataParallel 
# from apex import amp  
# from apex.parallel import DistributedDataParallel   


# relative import 
from Trainer.Trainer import Trainer, TrainingSettings  

# from Models.TNT import TNT   
# from



TNTParams = TrainingSettings()  
TNTParams.new_params = 1
print(TNTParams.new_params)

class TNTTrainer(Trainer):
    """
    TNTTrainer 
    """ 
    def __init__(self, model, optimizer, criterion, device, config):
        pass 


