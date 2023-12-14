from copyreg import pickle
import sys 
sys.path.append(".")
import os
import os.path as osp
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

import gc
from copy import deepcopy, copy

import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader
# from torch.utils.data import DataLoader
# sys.path.append("core/dataloader")


def get_fc_edge_index(node_indices):
    """
    node_indices: np.array([indices]), the indices of nodes connecting with each other;
    return a tensor(2, edges), indicing edge_index 
    ------------------------------------------------------
    全连接图，边索引
    """ 
    # xx: (9,9), yy:(9,9)  
    # e.g. node_indics = [1,2,3]
    # xx:[[1,2,3],[1,2,3],[1,2,3]] yy:[[1,1,1],[2,2,2],[3,3,3]]
    xx, yy = np.meshgrid(node_indices, node_indices)
    xy = np.vstack(([xx.reshape(-1), yy.reshape(-1)])).astype(np.int64) # 生成edge_index(src, tar) 符合torch_geometric格式
    return xy


def get_traj_edge_index(node_indices):
    """
    generate the polyline graph for traj, each node are only directionally connected with the nodes in its future
    node_indices: np.array([indices]), the indices of nodes connecting with each other;
    return a tensor(2, edges), indicing edge_index
    """
    edge_index = np.empty((2, 0))
    for i in range(len(node_indices)):
        xx, yy = np.meshgrid(node_indices[i], node_indices[i:])
        edge_index = np.hstack([edge_index, np.vstack(([xx.reshape(-1), yy.reshape(-1)])).astype(np.int64)])
    return edge_index


class GraphData(Data):
    """
    override key `cluster` indicating which polyline_id is for the vector
    """

    def __inc__(self, key, value):
        if key == 'edge_index':
            return self.x.size(0)
        elif key == 'cluster':
            return int(self.cluster.max().item()) + 1
        else:
            return 0

# dataset loader which loads data into memory
class ArgoverseInMem(InMemoryDataset):
    """ 
    Raw DataTree(After Preprocessing) 
    ------------------------------------------
    "../TNT-Trajectory-Prediction/interm_data"  
    - train_intermediate 
        - raw 
            - feature_xxx.pkl
        - processed(after this .py)
            - data.pt 
            - pre_transform.pt 
            - pre_filter.pt 
    - val_intermediate
    - test_intermediate
    """
    def __init__(self, root, transform=None, pre_transform=None):
        super(ArgoverseInMem, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        gc.collect()

    @property
    def raw_file_names(self):
        return [file for file in os.listdir(self.raw_dir) if "features" in file and file.endswith(".pkl")]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        """ transform the raw data and store in GraphData """
        # loading the raw data 
        # 存放轨迹数量
        traj_lens = []  
        # 存放轨迹数量 + lane数量
        valid_lens = [] 
        # 候选轨迹数 
        candidate_lens = [] 
        
        for raw_path in tqdm(self.raw_paths, desc="Loading Raw Data..."): 
            #TODO: debug写死raw_paths  
            # raw_path = "../TNT-Trajectory-Prediction/interm_data_small/train_intermediate/features_1.pkl" 
            
            # 读取预处理的pickle文件 
            raw_data = pd.read_pickle(raw_path)
                
            # raw_data['feats'] 观测特征数据 
            # -- dim(traj_num, obs_horizon<20>, 3) 3:(x,y,1.0)
            traj_num = raw_data['feats'].values[0].shape[0] # features_1.pkl: 6条轨迹
            traj_lens.append(traj_num)

            # raw_data['graph'] 
            # 包括ctrs 中心点坐标(N,2) 中心线坐标序列前后两个点中点
            # num_nodes 节点数(N)
            # feats 方向(N,2) 
            # turns转向(N,2) --- (0,1) R, (1,0) L, (1,1) Straight 
            # control(N,) 红绿灯控制 --- bool 
            # intersect(N,) 路口 --- bool 
            # lan_idcs: lane_idx(N, ) --- features1.pkl基本1个lane 9个节点
            lane_num = raw_data['graph'].values[0]['lane_idcs'].max() + 1 # 车道数
            valid_lens.append(traj_num + lane_num) 

            # 候选点, (M, 2), M位候选点数
            candidate_num = raw_data['tar_candts'].values[0].shape[0]
            candidate_lens.append(candidate_num) 
            
            # debug 
            break  
        
        num_valid_len_max = np.max(valid_lens)
        num_candidate_max = np.max(candidate_lens)
        print("\n[Argoverse]: The maximum of valid length is {}.".format(num_valid_len_max))
        print("[Argoverse]: The maximum of no. of candidates is {}.".format(num_candidate_max))

        # 转换入口 --- from pkl -> GraphData
        # pad vectors to the largest polyline id and extend cluster, save the Data to disk
        data_list = []
        for ind, raw_path in enumerate(tqdm(self.raw_paths, desc="Transforming the data to GraphData...")):  
            
            #TODO: debug写死raw_paths --- 2 
            # raw_path = "../TNT-Trajectory-Prediction/interm_data_small/train_intermediate/features_1.pkl" 
            
            raw_data = pd.read_pickle(raw_path)

            # input data  
            # x:(N, 2) <n_nodes, 2> (时序轨迹node + lane node)   
            # cluster: (N, ) [0,0..,0, ..., 131, 131,...131] 共132(K)个聚类(for 1.pkl) 126+6 共132个
                # - K为number of valid polylines
            # edge_index: (2, X), 2 (src, tar) 19*19*6 + 126/9 * 126/9 
            # indentifier(K, 2) 每个类的身份标识
            x, cluster, edge_index, identifier = self._get_x(raw_data)  
            
            # y.shape:(pred_len*2, ) y[:2] 为Final与Initial的Offset，y[2:]为所有的差分offset
            y = self._get_y(raw_data)  
            
            # Graph Data Standart Format 
            graph_input = GraphData(
                x=torch.from_numpy(x).float(), # (N_nodes, 2) 
                y=torch.from_numpy(y).float(),  # (pred_len*2, )
                cluster=torch.from_numpy(cluster).short(), # (N_nodes, )
                edge_index=torch.from_numpy(edge_index).long(), # (2, X)
                identifier=torch.from_numpy(identifier).float(),    # the identify embedding of global graph completion

                traj_len=torch.tensor([traj_lens[ind]]).int(),            # number of traj polyline --- 
                valid_len=torch.tensor([valid_lens[ind]]).int(),          # number of valid polylines (traj polylines + )
                time_step_len=torch.tensor([num_valid_len_max]).int(),    # the maximum of no. of polyline

                candidate_len_max=torch.tensor([num_candidate_max]).int(),
                candidate_mask=[],
                candidate=torch.from_numpy(raw_data['tar_candts'].values[0]).float(), # shape(588, 2) 588位target sampling的候选点数
                candidate_gt=torch.from_numpy(raw_data['gt_candts'].values[0]).bool(), #  one_hot （588， 1） 候选点的GT的one hot
                offset_gt=torch.from_numpy(raw_data['gt_tar_offset'].values[0]).float(), # offset的gt  得到Target Sampling中的Points中离GT最近的偏移量 
                target_gt=torch.from_numpy(raw_data['gt_preds'].values[0][0][-1, :]).float(), # (traj_nums, pred_len, 2)

                orig=torch.from_numpy(raw_data['orig'].values[0]).float().unsqueeze(0), # (2,) 坐标系norm
                rot=torch.from_numpy(raw_data['rot'].values[0]).float().unsqueeze(0), # (2,2) 旋转坐标系(Local->Global) local是最后一个观测点
                seq_id=torch.tensor([int(raw_data['seq_id'])]).int() # 1 数据features 1.pkl
            )
            data_list.append(graph_input) 
            break

        # self.collate封装，将所有的数据打包成一个巨大的Data(Graph格式),并重构各个原始Data对象的切片字典slices
        # data: GraphData(candidate=[588,2], x=[1248,10], ...) 
        # slices:{'x': (0,N) -- size(2,)} 
        data, slices = self.collate(data_list) 
        
        # 将巨大的Data对象和切片字典slices存入文件（.pt）
        torch.save((data, slices), self.processed_paths[0])

    def get(self, idx):
        data = super(ArgoverseInMem, self).get(idx).clone()

        feature_len = data.x.shape[1]
        index_to_pad = data.time_step_len[0].item()
        valid_len = data.valid_len[0].item()

        # pad feature with zero nodes
        data.x = torch.cat([data.x, torch.zeros((index_to_pad - valid_len, feature_len), dtype=data.x.dtype)])
        data.cluster = torch.cat([data.cluster, torch.arange(valid_len, index_to_pad, dtype=data.cluster.dtype)]).long()
        data.identifier = torch.cat([data.identifier, torch.zeros((index_to_pad - valid_len, 2), dtype=data.identifier.dtype)])

        # pad candidate and candidate_gt
        num_cand_max = data.candidate_len_max[0].item()
        data.candidate_mask = torch.cat([torch.ones((len(data.candidate), 1)),
                                         torch.zeros((num_cand_max - len(data.candidate), 1))])
        data.candidate = torch.cat([data.candidate[:, :2], torch.zeros((num_cand_max - len(data.candidate), 2))])
        data.candidate_gt = torch.cat([data.candidate_gt,
                                       torch.zeros((num_cand_max - len(data.candidate_gt), 1), dtype=data.candidate_gt.dtype)])

        assert data.cluster.shape[0] == data.x.shape[0], "[ERROR]: Loader error!"

        return data

    @staticmethod
    def _get_x(data_seq):
        """
        feat: [xs, ys, vec_x, vec_y, step(timestamp), traffic_control, turn, is_intersection, polyline_id]; 
        ----------------------------------
        xs, ys: the control point of the vector, for trajectory, it's start point, for lane segment, it's the center point;
        vec_x, vec_y: the length of the vector in x, y coordinates;
        step: indicating the step of the trajectory, for the lane node, it's always 0;
        traffic_control: feature for lanes
        turn: twon binary indicator representing is the lane turning left or right;
        is_intersection: indicating whether the lane segment is in intersection;
        polyline_id: the polyline id of this node belonging to;
        """
        feats = np.empty((0, 10)) 
        edge_index = np.empty((2, 0), dtype=np.int64)
        identifier = np.empty((0, 2))

        # 1. get traj features 获得轨迹属性 
        # (Traj_Nums, 20(obs_len), 2)
        traj_feats = data_seq['feats'].values[0]  
        # (Traj_nums, 20) - bool 
        traj_has_obss = data_seq['has_obss'].values[0]
        
        # (20,1) 
        step = np.arange(0, traj_feats.shape[1]).reshape((-1, 1))
        traj_cnt = 0
        for _, [feat, has_obs] in enumerate(zip(traj_feats, traj_has_obss)): 
            # (20,2) 
            # has_obs True or False --- mask作用 
            # (obs_len-1, 2) e.g. 19,2
            xy_s = feat[has_obs][:-1, :2] 
            # vector向量 --- 差分 
            # (obs_len-1, 2)
            vec = feat[has_obs][1:, :2] - feat[has_obs][:-1, :2] 
            
            # 交通 --- agent trajs不具有该属性，0填充
            traffic_ctrl = np.zeros((len(xy_s), 1)) 
            # 交叉路口 --- agent trajs不具有该属性，0填充
            is_intersect = np.zeros((len(xy_s), 1)) 
            # 转向 --- agent trajs不具有该属性，0填充
            is_turn = np.zeros((len(xy_s), 2)) 
            # 编号 polyline (agent 轨迹)
            polyline_id = np.ones((len(xy_s), 1)) * traj_cnt 
            feats = np.vstack([feats, np.hstack([xy_s, vec, step[has_obs][:-1], traffic_ctrl, is_turn, is_intersect, polyline_id])])   # feats(K, 10) K属性数, shape(K, 10), 其中10(xs,ys,vecx,vecy,step,ctrl,intersect, turn*2, polyline_id) K(Traj_nums)
            
            traj_cnt += 1
        # Final trajs: (19*traj_nums, 10), e.g. 1.pkl -- 114 (6*19)
        # 2. get lane features --- 道路属性 统一道路与轨迹属性
        graph = data_seq['graph'].values[0]
        ctrs = graph['ctrs'] # (lane)n_nodes, 2 <中心点> 
        vec = graph['feats'] # n_nodes, 2
        traffic_ctrl = graph['control'].reshape(-1, 1) #(n_nodes, 2)
        is_turns = graph['turn'] # (n_nodes, 2)
        is_intersect = graph['intersect'].reshape(-1, 1)
        lane_idcs = graph['lane_idcs'].reshape(-1, 1) + traj_cnt # 编号，道路
        steps = np.zeros((len(lane_idcs), 1)) 
        # 3. 整合polyline(traj feature)特征，和道路特征，整合至维数(n_trajs*19 + n_nodes, 10)
        feats = np.vstack([feats, np.hstack([ctrs, vec, steps, traffic_ctrl, is_turns, is_intersect, lane_idcs])])
        # 4. get the cluster and construct subgraph edge_index 
        # 编号： 前n_trajs*19为Agent轨迹，后n_nodes为道路
        # 类数: n_trajs*19(每个agents都是一类) + lanes(多个节点用一个lane)
        cluster = copy(feats[:, -1].astype(np.int64))
        for cluster_idc in np.unique(cluster):
            [indices] = np.where(cluster == cluster_idc) # 0类的索引, cluster[indice[0]:indice[-1]+1] 是同一类， 内部子图subgraph是全联通图 
            identifier = np.vstack([identifier, np.min(feats[indices, :2], axis=0)]) # 进入该cluster的x,y信息 -- 类身份识别 
            if len(indices) <= 1:
                continue                # skip if only 1 node
            if cluster_idc < traj_cnt: # 前traj_cnt的Agent边属性 (2,X)
                # 这里对于agent来说，是构建其traj的subgraph，耦合了时序信息
                edge_index = np.hstack([edge_index, get_fc_edge_index(indices)]) # edge_index(2, len(indics)xlen(indics))
            else:
                edge_index = np.hstack([edge_index, get_fc_edge_index(indices)]) # traj_cnt~feats.shape[0]的路属性
        # e.g. 
        # traj_nums=agent_nums=6
        # lane_nums
        # feats:(1248, 10) 1248 = traj_nums * 19(轨迹长) + n_nodes 
        # cluster: (1248, ) 同上 n_cluster = traj_nums * 19 + lane_nums
        # edge_index  --- 边(2,12372) 12372 = agent_num(or traj_nums)*(obs_len*obs_len)+n_lanes*(9*9)
        # indentifier --- 类身份表示 (n_clusters, 2) (2：x,y入口)
        return feats, cluster, edge_index, identifier

    @staticmethod
    def _get_y(data_seq): 
        # POI Agent的观测obs  (20, 2)
        traj_obs = data_seq['feats'].values[0][0] 
        # POI Agent的预测真值 (30, 2)
        traj_fut = data_seq['gt_preds'].values[0][0]  
        # offset_fut: 1, 2 + 29, 2 -> 30, 2 (第一个1,2是起点和终点的偏移量，后面29,2是差分偏移量)
        offset_fut = np.vstack([traj_fut[0, :] - traj_obs[-1, :2], traj_fut[1:, :] - traj_fut[:-1, :]])
        return offset_fut.reshape(-1).astype(np.float32)


class ArgoverseInDisk(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(ArgoverseInDisk, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        gc.collect()

    @property
    def raw_file_names(self):
        return [file for file in os.listdir(self.raw_dir) if "features" in file and file.endswith(".pkl")] 

    @property
    def processed_file_names(self):
        return [file for file in os.listdir(self.processed_dir) if "data" in file and file.endswith(".pt")]

    def download(self):
        pass

    def len(self):
        return len(self.processed_file_names)

    def process(self):
        """ transform the raw data and store in GraphData """
        # loading the raw data
        traj_lens = []
        valid_lens = []
        candidate_lens = []
        for raw_path in tqdm(self.raw_paths, desc="Loading Raw Data..."):
            raw_data = pd.read_pickle(raw_path)

            # statistics
            traj_num = raw_data['feats'].values[0].shape[0]
            traj_lens.append(traj_num)

            lane_num = raw_data['graph'].values[0]['lane_idcs'].max() + 1
            valid_lens.append(traj_num + lane_num)

            candidate_num = raw_data['tar_candts'].values[0].shape[0]
            candidate_lens.append(candidate_num)
            
        num_valid_len_max = np.max(valid_lens)
        num_candidate_max = np.max(candidate_lens)
        print("\n[Argoverse]: The maximum of valid length is {}.".format(num_valid_len_max))
        print("[Argoverse]: The maximum of no. of candidates is {}.".format(num_candidate_max))

        # pad vectors to the largest polyline id and extend cluster, save the Data to disk
        for ind, raw_path in enumerate(tqdm(self.raw_paths, desc="Transforming the data to GraphData...")):
            file_name = osp.split(raw_path)[1]
            file_id = re.findall(r"\d+", file_name)[0]

            raw_data = pd.read_pickle(raw_path)
            # input data
            x, cluster, edge_index, identifier = self._get_x(raw_data)
            y = self._get_y(raw_data)
            graph_input = GraphData(
                x=torch.from_numpy(x).float(),
                y=torch.from_numpy(y).float(),
                cluster=torch.from_numpy(cluster).short(),
                edge_index=torch.from_numpy(edge_index).long(),
                identifier=torch.from_numpy(identifier).float(),    # the identify embedding of global graph completion

                traj_len=torch.tensor([traj_lens[ind]]).int(),            # number of traj polyline
                valid_len=torch.tensor([valid_lens[ind]]).int(),          # number of valid polyline
                time_step_len=torch.tensor([num_valid_len_max]).int(),    # the maximum of no. of polyline

                candidate_len_max=torch.tensor([num_candidate_max]).int(),
                candidate_mask=[],
                candidate=torch.from_numpy(raw_data['tar_candts'].values[0]).float(),
                candidate_gt=torch.from_numpy(raw_data['gt_candts'].values[0]).bool(),
                offset_gt=torch.from_numpy(raw_data['gt_tar_offset'].values[0]).float(),
                target_gt=torch.from_numpy(raw_data['gt_preds'].values[0][0][-1, :]).float(),

                orig=torch.from_numpy(raw_data['orig'].values[0]).float().unsqueeze(0),
                rot=torch.from_numpy(raw_data['rot'].values[0]).float().unsqueeze(0),
                seq_id=torch.tensor([int(raw_data['seq_id'])]).int()
            )

            # save the data into a single file
            torch.save(graph_input, osp.join(self.processed_dir, 'data_{}.pt'.format(file_id)))

    #TODO : 具体的Padding方法还得Debug一遍
    def get(self, idx: int):
        data = torch.load(osp.join(self.processed_dir, self.processed_file_names[idx]))

        feature_len = data.x.shape[1] # 2, x,y (for traj prediction )
        index_to_pad = data.time_step_len[0].item() # 某个X.pkl下，这一段数据中，最长的tp len，进行padding
        valid_len = data.valid_len[0].item()

        # pad feature with zero nodes
        data.x = torch.cat([data.x, torch.zeros((index_to_pad - valid_len, feature_len), dtype=data.x.dtype)])
        data.cluster = torch.cat([data.cluster, torch.arange(valid_len, index_to_pad)]).long()
        data.identifier = torch.cat([data.identifier, torch.zeros((index_to_pad - valid_len, 2), dtype=data.x.dtype)])

        # pad candidate and candidate_gt
        num_cand_max = data.candidate_len_max[0].item()
        data.candidate_mask = torch.cat([torch.ones((len(data.candidate), 1)),
                                         torch.zeros((num_cand_max - len(data.candidate), 1))])
        data.candidate = torch.cat([data.candidate, torch.zeros((num_cand_max - len(data.candidate), 2))])
        data.candidate_gt = torch.cat([data.candidate_gt, torch.zeros((num_cand_max - len(data.candidate_gt), 1))])

        return data

    @staticmethod
    def _get_x(data_seq):
        """
        feat: [xs, ys, vec_x, vec_y, step(timestamp), traffic_control, turn, is_intersection, polyline_id];
        xs, ys: the control point of the vector, for trajectory, it's start point, for lane segment, it's the center point;
        vec_x, vec_y: the length of the vector in x, y coordinates;
        step: indicating the step of the trajectory, for the lane node, it's always 0;
        traffic_control: feature for lanes
        turn: twon binary indicator representing is the lane turning left or right;
        is_intersection: indicating whether the lane segment is in intersection;
        polyline_id: the polyline id of this node belonging to;
        """
        feats = np.empty((0, 10))
        edge_index = np.empty((2, 0), dtype=np.int64)
        identifier = np.empty((0, 2))

        # get traj features
        traj_feats = data_seq['feats'].values[0]
        traj_has_obss = data_seq['has_obss'].values[0]
        step = np.arange(0, traj_feats.shape[1]).reshape((-1, 1))
        traj_cnt = 0
        for _, [feat, has_obs] in enumerate(zip(traj_feats, traj_has_obss)):
            xy_s = feat[has_obs][:-1, :2]
            vec = feat[has_obs][1:, :2] - feat[has_obs][:-1, :2]
            traffic_ctrl = np.zeros((len(xy_s), 1))
            is_intersect = np.zeros((len(xy_s), 1))
            is_turn = np.zeros((len(xy_s), 2))
            polyline_id = np.ones((len(xy_s), 1)) * traj_cnt
            feats = np.vstack([feats, np.hstack([xy_s, vec, step[has_obs][:-1], traffic_ctrl, is_turn, is_intersect, polyline_id])])
            traj_cnt += 1

        # get lane features
        graph = data_seq['graph'].values[0]
        ctrs = graph['ctrs']
        vec = graph['feats']
        traffic_ctrl = graph['control'].reshape(-1, 1)
        is_turns = graph['turn']
        is_intersect = graph['intersect'].reshape(-1, 1)
        lane_idcs = graph['lane_idcs'].reshape(-1, 1) + traj_cnt
        steps = np.zeros((len(lane_idcs), 1))
        feats = np.vstack([feats, np.hstack([ctrs, vec, steps, traffic_ctrl, is_turns, is_intersect, lane_idcs])])

        # get the cluster and construct subgraph edge_index
        cluster = copy(feats[:, -1].astype(np.int64))
        for cluster_idc in np.unique(cluster):
            [indices] = np.where(cluster == cluster_idc)
            identifier = np.vstack([identifier, np.min(feats[indices, :2], axis=0)])
            if len(indices) <= 1:
                continue                # skip if only 1 node
            if cluster_idc < traj_cnt:
                edge_index = np.hstack([edge_index, get_fc_edge_index(indices)])
            else:
                edge_index = np.hstack([edge_index, get_fc_edge_index(indices)])
        return feats, cluster, edge_index, identifier

    @staticmethod
    def _get_y(data_seq):
        traj_obs = data_seq['feats'].values[0][0]
        traj_fut = data_seq['gt_preds'].values[0][0]
        offset_fut = np.vstack([traj_fut[0, :] - traj_obs[-1, :2], traj_fut[1:, :] - traj_fut[:-1, :]])
        return offset_fut.reshape(-1).astype(np.float32)


if __name__ == "__main__":

    # INTERMEDIATE_DATA_DIR = "../../dataset/interm_data"
    # 预处理后的中间数据存放路径  
    INTERMEDIATE_DATA_DIR = "../TNT-Trajectory-Prediction/interm_data_small" 
    # train_intermediate  
    # val_intermediate 
    # test_intermediate 
    for folder in ["train", "val", "test"]:
    # for folder in ["test"]: 
        dataset_input_path = os.path.join(INTERMEDIATE_DATA_DIR, f"{folder}_intermediate")

        # dataset = Argoverse(dataset_input_path)
        dataset = ArgoverseInMem(dataset_input_path).shuffle()
        batch_iter = DataLoader(dataset, batch_size=16, num_workers=0, shuffle=True, pin_memory=False)
        for k in range(1):
            for i, data in enumerate(tqdm(batch_iter, total=len(batch_iter), bar_format="{l_bar}{r_bar}")):
                print(i)
                print(data)

            # print("{}".format(i))
            # candit_len = data.candidate_len_max[0]
            # print(candit_len)
            # target_candite = data.candidate[candit_gt.squeeze(0).bool()]
            # try:
            #     # loss = torch.nn.functional.binary_cross_entropy(candit_gt, candit_gt)
            #     target_candite = data.candidate[candit_gt.bool()]
            # except:
            #     print(torch.argmax())
            #     print(candit_gt)
            # # print("type: {}".format(type(candit_gt)))
            # print("max: {}".format(candit_gt.max()))
            # print("min: {}".format(candit_gt.min()))