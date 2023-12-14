from fileinput import filename
from logging import warn
import os
import argparse
from os.path import join as pjoin
import copy
import sys
from wsgiref.headers import tspecials
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy import sparse

import warnings 
import sys 
sys.path.append(".") # pwd: /home/..VectorZoo/


# import torch
from torch.utils.data import Dataset, DataLoader

from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.visualization.visualize_sequences import viz_sequence
from argoverse.utils.mpl_plotting_utils import visualize_centerline 

# from core.util.preprocessor.base import Preprocessor
# from core.util.cubic_spline import Spline2D   
from Utils.cubicSpline import Spline2D 

from shapely.geometry.polygon import Polygon 

warnings.filterwarnings("ignore")


class ArgoversePreprocessor(Dataset): 
    """  
    数据预处理，利用Dataloader，getitem并不返回，将数据转存为.pkl
    - 对原始数据进行特征工程
    - data
    
    original datatree
        - TNT-Traj-Pred 
            - dataset 
                - raw_data 
                    - train 
                        - .csv 
                    - val 
                        - .csv 
                    - test_obs 
                        - .csv 
                - interm_data (2D) 
    
    realistic datatree 
        - TNT-Trajectory-Predicition 
            - train
                - train 
                    - data 
                        - .csv 
            - test 
                - test_obs
                    - data
                        - .csv 
            - val 
                - val 
                    - data 
                        - .csv  
            - interm_data 
    """
    
    def __init__(self, 
                 root_dir, 
                 split="train",  # test/val/train
                 algo="tnt",
                 obs_horizon = 20, # 20 * 0.1 = 2s 观测 
                 obs_range = 100, # -50 ~ 50 
                 pred_horizon = 30, # 预测 30 * 0.1 = 2s 
                 normalized = True, 
                 save_dir = None 
                 ):
        super().__init__() 
        
        # LANE_WIDTH   & COLOR_DICT 固有属性 
        self.LANE_WIDTH = {'MIA': 3.84, 'PIT': 3.97}
        self.COLOR_DICT = {"AGENT": "#d33e4c", "OTHERS": "#d3e8ef", "AV": "#007672"} 
        
        # ---- init params ----
        self.root_dir = root_dir
        self.algo = algo 
        self.obs_horizon = obs_horizon 
        self.obs_range = obs_range 
        self.pred_horizon = pred_horizon
        
        self.split = split
        self.normalized = normalized  
        
        # MAP初始化 
        self.am = ArgoverseMap() 
        
        # 'test_obs' or 'train' or 'val'  
        self.loader_dir = pjoin(self.root_dir,split) # /train/ 
        self.loader_dir = pjoin(self.loader_dir, self.split+"_obs"+"/data" if self.split == "test" else self.split+"/data") 
        
        assert os.path.exists(self.loader_dir), "【Warning!】data path error ... "
        self.loader = ArgoverseForecastingLoader(self.loader_dir)
        # self.loader = ArgoverseForecastingLoader(pjoin(self.root_dir, self.split+"_obs" if split == "test" else split)) # pjoin == os.path.join 
        
        # 保存路径
        self.save_dir = save_dir 
        print("数据预处理器实例化成功") # debug  
        
    def __getitem__(self, idx): 
        """Key-Value"""
        f_path = self.loader.seq_list[idx]    # /media/.../idx.csv
        seq = self.loader.get(f_path) 
        path, seq_f_name_ext = os.path.split(f_path)   # path =  /media/.../train/train/data, seq_f_name_ext = 'idx.csv'
        seq_f_name, ext = os.path.splitext(seq_f_name_ext) # idx, .csv
        
        df = copy.deepcopy(seq.seq_df)    
        
        return self.process_and_save(df, seq_id=seq_f_name, dir_ = self.save_dir)
        
    # Main function of preprocess 
    def process(self, dataframe:pd.DataFrame, seq_id, map_feat = True, viz=False): 
        """
        dataframe : data frame 
        seq_id: sequence id 
        map_feat : 地图特征数据   
        
        return: 
            DataFrame (处理后数据)
        """
        # 基本属性  
        # 输出: 
            # data: dict() 
                # key1 = city 
                # key2 = trajs (AV\AGENTS\OTHERS的轨迹)
                # key3 = steps (AV\AGENTS\OTHERS的时间步)
        data = self.read_argo_data(dataframe) 
        
        # features属性  
        """
        # data['orig'] = orig --- AGENT原点(obs的last point) (2,)
        # data['theta'] = theta --- 当前AGENT坐标系, obs的last pt和pred的第一个pt的方向为正方向  
        # data['rot'] = rot --- WorldFrame -> AgentFrame旋转矩阵 

        # data['feats'] = feats --- 观测范围内, N(agents) x obs_horizon x 3
        # data['has_obss'] = has_obss --- Mask N(agents) x obs_horizon x 1

        # data['has_preds'] = has_preds --- Mask N(agents) x pred_horizon 
        # data['gt_preds'] = gt_preds --- 预测GT N x pred_horizon x 2
        # data['tar_candts'] = tar_candts --- 基于CubicSpline，采样后的候选点(candidate sampling process )
        # data['gt_candts'] = tar_candts_gt --- 在tar_candts中与gt最接近的target sample 
        # data['gt_tar_offset'] = tar_offse_gt --- 最接近的与gt的offset

        # data['ref_ctr_lines'] = splines         # the reference candidate centerlines Spline for prediction 
        # data['ref_cetr_idx'] = ref_idx          # the idx of the closest reference centerlines  
        """
        data = self.get_obj_feats(data) 
        
        # # graphs    
        """
        # graph = dict() 
        # graph['ctrs'] = np.concatenate(ctrs, 0)  #(num_nodes, 2) 中心点坐标
        # graph['num_nodes'] = num_nodes  # 节点数
        # graph['feats'] = np.concatenate(feats, 0) #(num_nodes, 2) 方向
        # graph['turn'] = np.concatenate(turn, 0) #(num_nodes, 2) 2: (0,1) 右转, (1,0) 左转, (0,0) 直行
        # graph['control'] = np.concatenate(control, 0) #(num_nodes, ) 红绿灯控制
        # graph['intersect'] = np.concatenate(intersect, 0)  #(num_nodes, ) 路口
        # graph['lane_idcs'] = np.concatenate(lane_idcs, 0)  #(lane_idcs) lane idx 
        """
        data['graph'] = self.get_lane_graph(data) 
        data['seq_id'] = seq_id 
        
        # if vis , viz & check 
        if viz:
            self.visualize_data(data)
        
        # seq_id 
        return  pd.DataFrame([[data[key] for key in data.keys()]], columns = [key for key in data.keys()]) 
        
        
    def process_and_save(self, dataframe:pd.DataFrame, seq_id, dir_=None, map_feat = True):
        """预处理 & 保存

        Args:
            dataframe (pd.DataFrame): data frame 
            seq_id (str): seq_id
            dir_ : the dir 保存.pkl 
            map_feat :  bool 是否输出map feature
        """ 
        df_processed = self.process(dataframe, seq_id, map_feat,viz=False)  
        
        # 保存 
        self.save(df_processed, seq_id, dir_)  
        
        return []
         
    
    def __len__(self):
        return len(self.loader)  
    
    @staticmethod
    def read_argo_data(df: pd.DataFrame):  # DONE 
        """argoverse数据读取
        - 读取的是一个pd.DataFrame, 比如 data/1.csv  
        
        Returns:
            data dict() ['city', 'trajs', 'steps']
        """
        city = df["CITY_NAME"].values[0] 
        # print(city)  
        
        # TIMESTAMP|TRACK_ID|OBJECT_TYPE(AD,AGENT,OTHERS)|X,Y|CITY_NAME---------------------------------
        # 对其单个Scenario的时间戳，解决5s场景数据中，其他车辆OTHERS并不全在第一帧出现，最后一帧消失，可能在任意时刻
        # 出现、也可能在任意时刻消失的问题 
        after_deduplicate = np.unique(df['TIMESTAMP'].values)
        agt_ts = np.sort(after_deduplicate)   
        mapping = dict() 
        for i, ts in enumerate(agt_ts): 
            mapping[ts] = i # 地图时间错与索引mapping
        # timesteps 
        # len = len(df) e.g. 339 
        # 数值 [0~49] 代表50个时间步  
        # AGENT、AV固定在50个
        # 其他OTHERS 在0~50，最大50个时间步 
        # e.g. [0,0,0,0,1,1,1,2,2,...,49,49] Len = 339
        steps = [mapping[x] for x in df['TIMESTAMP'].values] 
        steps = np.asarray(steps, np.int64)  # (len)     
        # -----------------------------------------------------------------------------------------------
        
        # trajs 
        traj_x = df.X.to_numpy().reshape(-1,1) 
        traj_y = df.Y.to_numpy().reshape(-1,1) 
        trajs = np.concatenate((traj_x, traj_y), 1) # (len,2)  
        
        # Trackid & object_type process 
        # 定义一个函数，用于根据TRACK_ID和OBJECT_TYPE对df进行分组 
        # objs [(track_id, object_type): Int64Index(....序列号(frame号))]
        objs = df.groupby(['TRACK_ID', 'OBJECT_TYPE']).groups  # hash
        
        # Keys:[("Track-id", "object_type(Agent/Av/Other)")]  list  
        keys = list(objs.keys())  
        
        # x[1] --- object_type [av, agent, other, other , ....]
        obj_type = [x[1] for x in keys] # AV/AGENTS/OTHERS   
        
        # - for AGENTS  
        agt_idx = obj_type.index('AGENT') # list-- index of all AGENTS  
        
        # keys[agt_idx] --- ("Track-id", KEY) 
        idcs = objs[keys[agt_idx]] # ALL AGENTS   依据索引拿到其出现的frame 
        
        # 得到这个.csv seq的agt_traj和agt_timestep
        agt_traj = trajs[idcs] # agents traj 
        agt_step = steps[idcs] # agents time-step   
        
        # 删除感兴趣AGENT对应的部分，剩下(AV,OTHERS) 
        del keys[agt_idx] 
        
        ctx_trajs, ctx_steps = [], [] 
        # import pdb; pdb.set_trace()
        # AV & OTHERS
        for key in keys: 
            idcs = objs[key] 
            ctx_trajs.append(trajs[idcs]) 
            ctx_steps.append(steps[idcs]) 
        
        # 处理后的dict数据
        # 将结构化的单个Scenario数据保存
        data = dict() 
        data['city'] = city  
        data['trajs'] = [agt_traj] + ctx_trajs  # AGENT Trajs, AV Trajs, OTHERS Trajs
        data['steps'] = [agt_step] + ctx_steps  # AV(50,), AGENTS(50,) OTHERS(0~50,)
        
        return data
    
    #TODO 总结道路中心线特征提取过程
    def get_obj_feats(self, data):
        """
        输入: data(dict()), src: from read_argo_data output  
        - 包括 -----
    
    
        输出：补充对象属性 
        data['orig'] = orig
        data['theta'] = theta
        data['rot'] = rot

        data['feats'] = feats
        data['has_obss'] = has_obss

        data['has_preds'] = has_preds
        data['gt_preds'] = gt_preds
        data['tar_candts'] = tar_candts
        data['gt_candts'] = tar_candts_gt
        data['gt_tar_offset'] = tar_offse_gt

        data['ref_ctr_lines'] = splines         # the reference candidate centerlines Spline for prediction
        data['ref_cetr_idx'] = ref_idx          # the idx of the closest reference centerlines

        """   
        # data['trajs'][0] --- agent_traj
        # data['trajs'][1] --- av_traj 
        # data['trajs'][2:] --- others_trajs 
        # 获取POI Agent的最后一个point的信息，用于统一坐标系
        orig = data['trajs'][0][self.obs_horizon-1].copy().astype(np.float32)  
        
        # TODO:坐标系归一化理论 
        # 计算旋转矩阵
        # 以最后一个obs的方向为参考系
        if self.normalized:  # 如果需要坐标系归一化   
            
            # 车道方向 --- base是traj的Agent的最后一个观测Obs
            pre, conf = self.am.get_lane_direction(data['trajs'][0][self.obs_horizon-1], data['city'])
            
            # 置信度不够高的化，就基于Rule-based指定，这个Rule是依据obs的最后一个点以及倒数第4个点的朝向作为x正方向
            if conf <= 0.1:
                pre = (orig - data['trajs'][0][self.obs_horizon-4]) / 2.0 
                
            theta = - np.arctan2(pre[1], pre[0]) + np.pi / 2 
            
            # 旋转矩阵
            rot = np.asarray([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]], np.float32)
        else:
            # if not normalized, do not rotate. # 不需要坐标系归一化
            theta = None
            rot = np.asarray([
                [1.0, 0.0],
                [0.0, 1.0]], np.float32) 
            
        # get the target candidates and candidate gt 
        # 0~19 for obs (AGENT) 
        # shape(20,2)
        agt_traj_obs = data['trajs'][0][0: self.obs_horizon].copy().astype(np.float32) 
        
        # 20~49 for pred (AGENT) 
        # shape(30,2)
        agt_traj_fut = data['trajs'][0][self.obs_horizon:self.obs_horizon+self.pred_horizon].copy().astype(np.float32) # GT  
        
        # 获取候选的中心线 (全局坐标)
        # s ---- e (start - end) 
        # List[Array(shape:)] len(List) = num选的候选点
        ctr_line_candts = self.am.get_candidate_centerlines_for_traj(agt_traj_obs, data['city'], viz=False)   
        
        
        # 统一坐标系 Global->Local---------------------------------------------------------------
        # 将agent future traj 变换到相对于agent obs的last point的坐标系下  
        # 将center_line的候选车道线也变换到相对于agent obs的last point的坐标系下  
        # rotate the center lines and find the reference center line 
        agt_traj_fut = np.matmul(rot, (agt_traj_fut - orig.reshape(-1, 2)).T).T 
        for i, _ in enumerate(ctr_line_candts):
            ctr_line_candts[i] = np.matmul(rot, (ctr_line_candts[i] - orig.reshape(-1, 2)).T).T    
        
        # TODO: TNT的Target Candidate Sampling  
        # CublicSpline 2D 
        # target candidates after spline 2D operation  
        # TNT的target candidate sampling操作  
        # 这里的tar_candts需要与VectorNet提取的Feature进行连接 
        # output: (N, 2) N: line_pts 
        tar_candts = self.lane_candidate_sampling(ctr_line_candts,[0,0],viz=False) # Sampling Points 
        
        # Inference Phase 
        if self.split == "test":
            splines, ref_idx = None, None
            tar_candts_gt, tar_offse_gt = np.zeros((tar_candts.shape[0], 1)), np.zeros((1, 2))
            
        else: 
            # Training Phase使用参考Centerline和candidate的offset 
            # 1. 得到基于ctr_line_candts中筛选出的Ref的Centerline和其对应的Candidate中的索引 
            splines, ref_idx = self.get_ref_centerline(ctr_line_candts, agt_traj_fut,viz=False) 
            
            # 2. 得到Target Sampling中的Points中离GT终点最近的点的Onehot向量以及其偏移量 
            # tar_candt_gt与N与tar_candts一致，sampling pts的个数
            tar_candts_gt, tar_offse_gt = self.get_candidate_gt(tar_candts, agt_traj_fut[-1])

        feats, ctrs, has_obss, gt_preds, has_preds = [], [], [], [], []
        x_min, x_max, y_min, y_max = -self.obs_range, self.obs_range, -self.obs_range, self.obs_range 
        # Loop --- 
        # 1. traj (for agent, av, other1,...) 
        # 2. step (for agent ,av, other1,...)
        for traj, step in zip(data['trajs'], data['steps']):
            if self.obs_horizon-1 not in step: # Others 可能不一定在Range中, 排除不相关的Agents
                continue
        
            # normalize and rotate 
            # last point in obs   
            # 统一坐标系 
            traj_nd = np.matmul(rot, (traj - orig.reshape(-1, 2)).T).T

            # collect the future prediction ground truth
            gt_pred = np.zeros((self.pred_horizon, 2), np.float32)
            has_pred = np.zeros(self.pred_horizon, np.bool) 
            
            # 未来的数据index需要进行Mask 
            future_mask= np.logical_and(step >= self.obs_horizon, step < self.obs_horizon + self.pred_horizon) #(30,) 
            post_step = step[future_mask] - self.obs_horizon 
            post_traj = traj_nd[future_mask]
            gt_pred[post_step] = post_traj
            has_pred[post_step] =True

            # colect the observation
            obs_mask = step < self.obs_horizon #(20, )
            step_obs = step[obs_mask]
            traj_obs = traj_nd[obs_mask]
            idcs = step_obs.argsort()
            step_obs = step_obs[idcs]
            traj_obs = traj_obs[idcs] 
            
            # 有些OTHERS，在20个tps的观测周期中，前一段时间步并没有出现
            # OTHERS只Focus在最近的那个TPS中
            for i in range(len(step_obs)):
                if step_obs[i] == self.obs_horizon - len(step_obs) + i:
                    break 
                
            step_obs = step_obs[i:]
            traj_obs = traj_obs[i:] 
            
            # others ， 不考虑， 只考虑last observation frame
            if len(step_obs) <= 1:
                continue 
        
            feat = np.zeros((self.obs_horizon, 3), np.float32)
            has_obs = np.zeros(self.obs_horizon, np.bool)

            feat[step_obs, :2] = traj_obs
            feat[step_obs, 2] = 1.0
            has_obs[step_obs] = True

            # 超出范围的点，不考虑
            if feat[-1, 0] < x_min or feat[-1, 0] > x_max or feat[-1, 1] < y_min or feat[-1, 1] > y_max:
                continue
            
            # feats dimension 
            # (20, 3) # x, y, 1.0
            feats.append(feat)                  # displacement vectors
            has_obss.append(has_obs)
            gt_preds.append(gt_pred)
            has_preds.append(has_pred) 
        
        # numpy化 
        feats = np.asarray(feats, np.float32) # f32 shape(N<agents>,obs_horizon,3)
        has_obss = np.asarray(has_obss, np.bool) #Mask,shape(N<agents>,obs_horizon)
        gt_preds = np.asarray(gt_preds, np.float32) #shape(N<agents>,pred_horizon,2)
        has_preds = np.asarray(has_preds, np.bool)  #shape(pred_horizon,)
        
        # vis  
        # ref_idx 参考Ref_Centerline的idx 
        # gt_preds[0]: pred
        # feats[0]: obs
        # splines : 基于ctr_line_candts中筛选出的Ref的Centerline
        # ctr_line_candts : 候选centerline （从api获取，离散的初始base候选）
        # self.plot_ref_centerlines(ctr_line_candts, splines, feats[0], gt_preds[0], ref_idx)
        
        data['orig'] = orig
        data['theta'] = theta
        data['rot'] = rot

        data['feats'] = feats
        data['has_obss'] = has_obss

        data['has_preds'] = has_preds
        data['gt_preds'] = gt_preds
        data['tar_candts'] = tar_candts
        data['gt_candts'] = tar_candts_gt
        data['gt_tar_offset'] = tar_offse_gt

        data['ref_ctr_lines'] = splines         # the reference candidate centerlines Spline for prediction
        data['ref_cetr_idx'] = ref_idx          # the idx of the closest reference centerlines
        
        return data

    # 候选
    def lane_candidate_sampling(self, centerline_list, orig, distance=0.5, viz=False): 
        """
        TNT的工作流： 
        - 1. 输入local的centerline_list 
        - 2. 基于三次多项式 Spline2D 来对候选点进行Smooth Operation
        - 3. 得到优化后的候选的centerline_list  
        
        - Target candidate sampling采样距离在0.5m时为最佳 
            - 采样区域为centerline上选择
        
        - 折线 -> 平滑曲线
        Args:
            centerline_list (_type_): _description_
            orig (_type_): _description_
            distance (float, optional): _description_. Defaults to 0.5.
            viz (bool, optional): _description_. Defaults to True.
        """
        
        # print(type(centerline_list))  
        # print(len(centerline_list))
        # print(centerline_list[0].shape)  # 50,2 
        # print(centerline_list[1].shape)  
        
        # ------- 
        candidates = []  
        # 遍历每一条centerline 
        for land_id, line in enumerate(centerline_list): 
            sp = Spline2D(x=line[:,0], y=line[:,1]) # reso默认0.1  
            # TODO: x,y -> Frenet坐标系 
            # 在局部系 
            # s: line
            # d: lateral 
            s_o, d_o = sp.calc_frenet_position(orig[0], orig[0])  
            
            # distance 0.5m  --- 沿着path的方向  from s_o -> s_final
            s = np.arange(s_o, sp.s[-1], distance)  
            
            # TODO s -> new x,y 
            # s -> x, y 
            ix, iy = sp.calc_global_position_online(s) 
            
            candidates.append(np.stack([ix,iy],axis=1)) 
        
        candidates = np.concatenate(candidates)
        # print("before unique:", candidates.shape)  
        # plt.scatter(candidates[:, 0], candidates[:, 1], marker="*", c="g", alpha=1, s=6.0, zorder=15)
        # plt.show()
        
        candidates = np.unique(candidates, axis=0)    # 去除重复点（有些line是重复的）
        # print("after unique:", candidates.shape)  
        # plt.scatter(candidates[:, 0], candidates[:, 1], marker="*", c="g", alpha=1, s=6.0, zorder=15)
        # plt.show()
        
        if viz:
            fig = plt.figure(0, figsize=(8, 7))
            fig.clear()
            for centerline_coords in centerline_list:
                visualize_centerline(centerline_coords)
            plt.scatter(candidates[:, 0], candidates[:, 1], marker="*", c="g", alpha=1, s=6.0, zorder=15)
            plt.xlabel("Map X")
            plt.ylabel("Map Y")
            # plt.axis("off")
            plt.title("No. of lane candidates = {}; No. of target candidates = {};".format(len(centerline_list), len(candidates))) 
            plt.legend()
            plt.show()
        
        return candidates
    
    @staticmethod 
    def get_ref_centerline(cline_list, pred_gt, viz=False): 
        # 平滑操作 
        """
        输入: 
        - ctr_line_candts : List of candidate centerlines [Numpy array(50, 2), ...] 
        - agt_traj_fut(GT) : Ground Truth Numpy array (30, 2) (local coords) 
            - 相对于obs的最后一个pt
        
        输出: 
        - ref_centerlines : Spline2D优化后的参考线
        - line_idx : 是和GT[29] Final Position 之间的差距最小的(FDE)的中心参考线的索引
        """
        
        if len(cline_list) == 1:
            # 只有一个候选 
            return [Spline2D(x=cline_list[0][:,0], y=cline_list[0][:,1])], 0     
        else:
            line_idx = 0 
            ref_centerlines = [Spline2D(x=cline_list[i][:, 0], y=cline_list[i][:, 1]) for i in range(len(cline_list))]

            # search the closest point of the traj final position to each center line
            min_distances = []
            for line in ref_centerlines:
                xy = np.stack([line.x_fine, line.y_fine], axis=1)
                diff = xy - pred_gt[-1, :2]
                dis = np.hypot(diff[:, 0], diff[:, 1])
                min_distances.append(np.min(dis))
            line_idx = np.argmin(min_distances)
            
        
            if viz:
                fig = plt.figure(0, figsize=(8, 7))
                fig.clear()
                for centerline_coords in cline_list:
                    visualize_centerline(centerline_coords)
                plt.scatter(line.x_fine, line.y_fine, marker="*", c="g", alpha=1, s=6.0, zorder=15)
                plt.xlabel("Map X")
                plt.ylabel("Map Y")
                # plt.axis("off")
                # plt.title("No. of lane candidates = {}; No. of target candidates = {};".format(len(centerline_list), len(candidates)))  
                
                
                # plt.legend()
                plt.show()

            return ref_centerlines, line_idx 
    
    @staticmethod 
    def get_candidate_gt(target_candidate, gt_target):
        """
        从目标candidate中，找到与gt最接近的，并输出GT的One-Hot编码  
        类似于Teacher Forcing的思想，基于Ground Truth进行指导，在training phase使用

        Args:
            target_candidate (_type_): (N, 2) Candidates 
            gt_target (_type_): (1,2) Final Target (e.g. agt_traj_fut[-1])

        Returns:
            onehot, offset_xy 
        """ 
        # 计算目标点 和 GTTarget之间的欧式距离 , 并找出最小的点的索引， 在对应的One Hot向量上置1
        displacement = gt_target - target_candidate 
        gt_index = np.argmin(np.power(displacement[:,0],2) + np.power(displacement[:,1],2)) 
        
        # 类似于Mask 
        one_hot = np.zeros((target_candidate.shape[0], 1)) 
        one_hot[gt_index] = 1 # gt_index处 
        
        # delta_x, delta_y
        offset_xy = gt_target - target_candidate[gt_index] 
        
        return one_hot, offset_xy
    
    
    def save(self, dataframe:pd.DataFrame, file_name, dir_=None):
        """
        save the feature in the data sequence in a single csv files
        :param dataframe: DataFrame, the dataframe encoded
        :param set_name: str, the name of the folder name, exp: train, eval, test
        :param file_name: str, the name of csv file
        :param dir_: str, the directory to store the csv file (csv file) 
        :return: 
        """
        # 类型检测 
        if not isinstance(dataframe, pd.DataFrame):
            return  
        
        if not dir_:
            dir_ = os.path.join(os.path.split(self.root_dir)[0], "intermediate", self.split + "_intermediate", "raw")
        else:
            dir_ = os.path.join(dir_, self.split + "_intermediate", "raw") 
            
        if not os.path.exists(dir_):
            os.makedirs(dir_)
        
        fname = f"features_{file_name}.pkl" # to pickle 
        dataframe.to_pickle(pjoin(dir_, fname))  
        print(f"[Preprocess]: Saving data to {dir_} with name {fname} ... ", end='\r') # debug 
                    
    def get_lane_graph(self, data):
        """
        Rectangles
        基于Obs_range的矩形区域 
        ------------------------------------------------------------ 
        1. 依据obs_range，获取搜索半径
        2. 使用argoverse api获取半径内所有地图Lane  
        3. 坐标系norm，统一，旋转到与AGENT一致 
        4. 获取Lane Graph Node的几何、属性信息（几何、朝向、转向、红绿灯、路口属性）
        5. 构建车道拓扑关系 
        6. TODO 构建前驱后继关系 
        7. 构建图 
        """ 
        # 机器人的FOV 
        x_min,x_max,y_min,y_max = -self.obs_range, self.obs_range, -self.obs_range, self.obs_range
        radius = max(abs(x_min), abs(x_max))+max(abs(y_min), abs(y_max))
        radius_scale = 1.5
        
        # 1. 获取观测范围搜索半径内Lanes的id --------------------------------------------------
        lane_ids = self.am.get_lane_ids_in_xy_bbox(data['orig'][0], data['orig'][1], data['city'], radius*radius_scale) 
        lane_ids = copy.deepcopy(lane_ids) 
        lanes = dict()
        
        # 2. 坐标变换，保证lanes与agents的坐标系一致（last point方向） ----------------------------- 
        for lane_id in lane_ids:
            # 2.1 拿到lane_id对应的centerline
            lane = self.am.city_lane_centerlines_dict[data['city']][lane_id] 
            lane = copy.deepcopy(lane) 
            
            # vis 
            # visualize_centerline(lane.centerline)
            
            centerline = np.matmul(data['rot'], (lane.centerline - data['orig'].reshape(-1,2)).T).T 
            x, y = centerline[:, 0], centerline[:, 1] 
            
            # 2.2 范围筛选，丢弃超范围的数值
            if x.max() < x_min or x.min() > x_max or y.max() < y_min or y.min() > y_max:
                continue 
            
            # 2.3 基于原始Centerline来获得多边形 
            else:  
                polygon = self.am.get_lane_segment_polygon(lane_id, data['city'])
                polygon = copy.deepcopy(polygon) 
                lane.centerline = centerline  
                
                # polygon vis  
                # display()  
                # 统一坐标系 
                lane.polygon = np.matmul(data['rot'], (polygon[:,:2] - data['orig'].reshape(-1,2)).T).T  
                lanes[lane_id] = lane 
        
        # plt.show() 
        
        # 3. 计算Lane Graph Node的几何信息和属性信息 ------------------------------------------
        # lanes {} dict() Key: lane_id in obs_range, Value: lane property 
        lane_ids = list(lanes.keys())  
        # 几何、朝向、转向、红绿灯、路口属性
        ctrs, feats, turn, control, intersect = [], [], [], [], []
        for lane_id in lane_ids:
            lane = lanes[lane_id] 
            
            # 3.1 中心线
            ctrln = lane.centerline 
            # 3.2 段数
            num_segs = len(ctrln) - 1 # 点数 - 1  
            # 3.3 几何 --- 车道中心线坐标序列前后两个点中点
            ctrs.append(np.asarray((ctrln[:-1]+ctrln[1:])/2.0, np.float32))  
            # 3.4 朝向Orientation --- vi^end - vi^start
            feats.append(np.asarray(ctrln[1:]-ctrln[:-1], np.float32)) 
            # 3.5 车道转向 --- 左转、直行、右转、掉头 (左右转) 
            # TODO: 只考虑了左转右转的setting 
            x = np.zeros((num_segs, 2), np.float32)  #(num_segs, 2)
            if lane.turn_direction == "LEFT": 
                x[:, 0] = 1 # 左转 
            elif lane.turn_direction == "RIGHT":
                x[:, 1] = 1 # 右转 
            else:
                pass 
            turn.append(x)  
            
            control.append(lane.has_traffic_control * np.ones(num_segs, np.float32))  
            intersect.append(lane.is_intersection * np.ones(num_segs, np.float32))
        

        # 4. 构建车道拓扑关系， predecessors、successors、Left neighbors、right neighbors (前后左右)---------
        # o(red)） -- o(blue) -- o(blue) -- o(blue) -- o(blue) -- o(green) 
        # 0           1          2          3          4          5 
        # 如上所注释，假设当前预测范围内包含三个Lane（不同颜色不同类），这些Lane Graph共有5个Node，分别赋予他们唯一编号 0~5 
        lane_idcs = [] 
        count = 0 
        for i, ctr in enumerate(ctrs):
            lane_idcs.append(i * np.ones(len(ctr), np.int64))  # nodes , 每个Lane上按照中心点划分节点
            count += len(ctr) 
        num_nodes = count # 节点数
        
        # 5. TODO 构建前驱后继关系
        
        # 6. 图
        graph = dict() 
        graph['ctrs'] = np.concatenate(ctrs, 0)  #(num_nodes, 2) 中心点坐标
        graph['num_nodes'] = num_nodes  # 节点数
        graph['feats'] = np.concatenate(feats, 0) #(num_nodes, 2) 方向
        graph['turn'] = np.concatenate(turn, 0) #(num_nodes, 2) 2: (0,1) 右转, (1,0) 左转, (0,0) 直行
        graph['control'] = np.concatenate(control, 0) #(num_nodes, ) 红绿灯控制
        graph['intersect'] = np.concatenate(intersect, 0)  #(num_nodes, ) 路口
        graph['lane_idcs'] = np.concatenate(lane_idcs, 0)  #(lane_idcs) lane idx
        
        return graph               

    # Plot & Visualize Tools 
    def plot_ref_centerlines(self, cline_list, splines, obs, pred, ref_line_idx):
        print("CenterCandis Nums:{}, Shape:{}".format(len(cline_list), cline_list[0].shape))
        print("Obs Shape:{}".format(obs.shape))
        print("Pred Shape:{}".format(pred.shape))
        
        fig = plt.figure(0, figsize=(8,7))
        fig.clear() 
        
        for centerline_coords in cline_list:
            visualize_centerline(centerline_coords) 
            
        for i, spline in enumerate(splines):
            xy = np.stack([spline.x_fine, spline.y_fine], axis=1) 
            print("Splines.shape:{}".format(xy.shape))
            if i == ref_line_idx:
                plt.plot(xy[:, 0], xy[:, 1], "--", color="r", alpha=0.7, linewidth=1, zorder=12)
            else:
                plt.plot(xy[:, 0], xy[:, 1], "--", color="g", alpha=0.5, linewidth=1, zorder=10)

        self.plot_traj(obs, pred)

        plt.xlabel("Map X")
        plt.ylabel("Map Y")
        # plt.axis("off")
        plt.show()
        # plt.show(block=False)
        # plt.pause(0.5)
    def plot_traj(self, obs, pred, traj_id=0):
        assert len(obs) != 0, "ERROR: The input trasectory is empty!"
        traj_na = "t{}".format(traj_id) if traj_id else "traj" 
        
        if traj_id == 0: 
            obj_type = 'AGENT' 
        elif traj_id == 1:
            obj_type = 'AV' 
        else:
            obj_type = 'OTHERS'
        # obj_type = "AGENT" if traj_id == 0 else "OTHERS"  
        
        print("OBJ_TYPE:【{}】".format(obj_type))
        # plt.title("{},trajectory") 
        plt.plot(obs[:, 0], obs[:, 1], color=self.COLOR_DICT[obj_type], alpha=1, linewidth=1, zorder=15)
        plt.plot(pred[:, 0], pred[:, 1], "d-", color=self.COLOR_DICT[obj_type], alpha=1, linewidth=1, zorder=15)

        plt.text(obs[0, 0], obs[0, 1], "{}_s".format(traj_na))

        if len(pred) == 0:
            plt.text(obs[-1, 0], obs[-1, 1], "{}_e".format(traj_na))
        else:
            plt.text(pred[-1, 0], pred[-1, 1], "{}_e".format(traj_na))

    
    def visualize_data(self, data):
        """
        visualize the extracted data, and exam the data
        """
        fig = plt.figure(0, figsize=(8, 7))
        fig.clear()

        # visualize the centerlines -------------------------------
        lines_ctrs = data['graph']['ctrs'] # 中心信息
        lines_feats = data['graph']['feats'] # 方向 
        lane_idcs = data['graph']['lane_idcs'] # ids 
        for i in np.unique(lane_idcs):
            line_ctr = lines_ctrs[lane_idcs == i] # s + e / 2
            line_feat = lines_feats[lane_idcs == i] # e - s 
            line_str = (2.0 * line_ctr - line_feat) / 2.0 # (2*（s+e）/2 - e + s ) / 2 = s 坐标
            line_end = (2.0 * line_ctr[-1, :] + line_feat[-1, :]) / 2.0 # 同上 e 坐标 
            line = np.vstack([line_str, line_end.reshape(-1, 2)]) #  centerline的格式 array(start_pos, end_pos) # shape (start_pos.shape[0] + end_pos.shape[0], 2)
            visualize_centerline(line)

        # visualize the trajectory
        trajs = data['feats'][:, :, :2]  # (num_agents, obs_horizon, 3)
        has_obss = data['has_obss'] # (num_agents, obs_horizon)
        preds = data['gt_preds'] #(num_agents, pred_horizon, 2)
        has_preds = data['has_preds'] # (num_agents, pred_horizon)
        for i, [traj, has_obs, pred, has_pred] in enumerate(zip(trajs, has_obss, preds, has_preds)):
            self.plot_traj(traj[has_obs], pred[has_pred], i)

        plt.xlabel("Map X")
        plt.ylabel("Map Y")
        plt.axis("off")
        plt.show()
        # plt.show(block=False)
        # plt.pause(0.5)
        


if __name__ == "__main__": 
    
    parser = argparse.ArgumentParser() 
    parser.add_argument("-r", "--root", type=str, default="../TNT-Trajectory-Prediction/") 
    parser.add_argument("-s", "--small", type=bool, default=False) # debug True   
    
    args = parser.parse_args()  
    
    for split in ["train", "val", "test"]:
        ArgoProcess = ArgoversePreprocessor(args.root, split, save_dir=pjoin(args.root, "interm_data" if not args.small else "interm_data_small")) 
        loader = DataLoader(
            ArgoProcess,
            batch_size = 1 if sys.gettrace() else 16, # debug mode 1 else 16 
            num_workers = 0 if sys.gettrace() else 8, # debug mode 0 else 16  
            shuffle = False, 
            pin_memory = False, 
            drop_last = False 
        )  
        
        for i, data in enumerate(tqdm(loader)): 
            if args.small:
                if split == "train" and i >= 20:
                    break 
                elif split == "val" and i >= 10:
                    break 
                elif split == "test" and i >= 10:
                    break
    
    
    
    # test code 
    # root_dir = "../TNT-Trajectory-Prediction/" 
    # ArgoProcess = ArgoversePreprocessor(root_dir)  
    
    # # test data  
    # seq_path = f'../TNT-Trajectory-Prediction/train/train/data/2000.csv'  
    
    # test_data = ArgoProcess.loader.get(seq_path).seq_df
    # # data = ArgoProcess.read_argo_data(test_data)  
    # # data = ArgoProcess.get_obj_feats(data)
    # # data = ArgoProcess.get_lane_graph(data)  

    # ArgoProcess[5] 

    # # data = ArgoProcess.process(test_data, 2000) 
    
    # # import pdb; pdb.set_trace()
    # #
    # # test1 = np.random.rand(18,2) 
    # # test2 = np.random.rand(18,2) 
    # # test3 = np.vstack([test1,test2]) 
    # # print(test3.shape)