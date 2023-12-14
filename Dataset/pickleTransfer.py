# 版本不对齐时（比如3.8生成的pickle，3.7生成报错ValueError: unsupported pickle protocol: 5） 
import os 
import pickle5 as pickle
# import pickle
import pandas as pd 

import sys 
sys.path.append(".")

# 预处理后的中间数据存放路径   
# /media/irvinghe/CodingSpace/ArgDataset/TNT-Trajectory-Prediction/interm_data_small/train_intermediate
INTERMEDIATE_DATA_DIR = "../TNT-Trajectory-Prediction/interm_data_small" 

dataset_input_path = os.path.join(INTERMEDIATE_DATA_DIR, "train_intermediate/features_1.pkl") 
print(dataset_input_path)

# with open(dataset_input_path, 'rb') as f:
#     config = pickle.load(f) 

# config = pickle.load(dataset_input_path)
raw_data = pd.read_pickle(dataset_input_path) 


