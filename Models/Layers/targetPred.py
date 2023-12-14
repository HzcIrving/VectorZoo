# -*- coding: utf-8 -*- 
import sys 
sys.path.append(".")

from Models.Layers.MLP import MLP   
import torch 
import torch.nn as nn 
import torch.nn.functional as F  
import numpy as np 

from torch.distributions import Normal  

# TODO: 分类Loss + Offset的Huber Loss
# 分类GT, 回归GT分别依赖于 candidate_gt 和 offset_gt  

# Masked Softmax
from Models.Layers.maskedSoftmax import masked_softmax 

class TargetPred(nn.Module): 
    """ 
    对Agent Future Targets的未来目标分布建模p(T|x)

    - N discrete quantized locations with continuous offsets 
      T = {τ^n} = {(x^n, y^n) + (delta_x^n, delta_y^n)} 
    
    - Dist over targets: discrete-continuous factorization 
    p(τ^n|x) = π(τ^n) · Gauss(delta_x^n|μx) · Gauss(delta_y^n|μy) 
    
    """
    def __init__(self, 
                 in_channels:int, 
                 hidden_dim:int=64, 
                 m:int=50,
                 device=torch.device("cpu"), 
                 feat_num=2):  
        """
        feat_num:特征数，(x,y)即为2
        """
        super(TargetPred, self).__init__()  
        
        self.in_chs = in_channels 
        self.hidden_dim = hidden_dim 
        
        # 输出M个待选的目标点
        self.M = m # output candidate target  
        
        self.device = device  
        
        # π 
        # self.in_chs(feat_in_dim), 2(tar_candidate<x,y>)
        self.prob_mlp = nn.Sequential(
            MLP(self.in_chs+2, hidden_dim, hidden_dim), 
            nn.Linear(hidden_dim, 1)
        )
        
        # μ offset_mean: delta_x, delta_y
        self.mean_mlp = nn.Sequential(
            MLP(self.in_chs+2, hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, feat_num)
        ) 
        
    def forward(self, 
                feat_in: torch.Tensor, 
                tar_candidate: torch.Tensor, 
                candidate_mask = None):
        """
        feat_in: context encoder轨迹特征编码的特征输入 (bs,1,in_channels)
        tar_candidate: Agent的Target Candidate Sampling的候选目标点， 共N个 
            - 若Vehicle Agent: 从lane_centerline采样(数据集的preprocess过程进行) 
            - 若Ped Agent：从虚拟Grid中采样，Grid中心是行人 
        
        candidate_mask: the mask of valid target candidate 
        -----------------------------------------
        Returns:
            - target_candidate_probability: π  
            - offset_mean: delta_x, delta_y
        """  
        
        # dimension must be [batch size, 1, in_channels]
        assert feat_in.dim() == 3, "[TNT-TargetPred]: Error input feature dimension" 
        
        # N个候选目标 
        batch_size, n, _ = tar_candidate.size() 
        
        # stack the target candidates to the end of input feature
        feat_in_repeat = torch.cat([feat_in.repeat(1, n, 1), tar_candidate], dim=2)
        
        # 1) prob output --- 删除dim为1的维度 
        # compute probability for each candidate
        prob = self.prob_mlp(feat_in_repeat).squeeze(2)  
        #   - discrete distribution over location (softmax) 
        if not isinstance(candidate_mask, torch.Tensor):
            tar_candit_prob = F.softmax(prob, dim=-1) # (bs, n_tar)
        else: 
            tar_candit_prob = masked_softmax(prob, candidate_mask, dim=-1)  

        # 2) mean offset predict 
        tar_offset_mean = self.mean_mlp(feat_in_repeat)                                         # [batch_size, n_tar, 2]

        return tar_candit_prob, tar_offset_mean 
    
    def loss(self,
             feat_in:torch.Tensor, 
             tar_candidate:torch.Tensor, 
             gt_candidate:torch.Tensor, 
             gt_offset:torch.Tensor, 
             candidate_mask=None, 
             debug = False):
        """
        Loss: Crossentropy(分类) Loss + Huber回归Loss 
              L_cls(pi, u) + L_offset(ux, uy, delta_x^u, delta_y^u) 
        compute the loss for target prediction, classification gt is binary labels,
        only the closest candidate is labeled as 1 
        -----------------------------------------------
        :param feat_in: encoded feature for the target candidate, [batch_size, inchannels]
        :param tar_candidate: the target candidates for predicting the end position of the target agent, [batch_size, N, 2]
        :param candidate_gt: target prediction ground truth, classification gt and offset gt, [batch_size, N] (One Hot)
        :param offset_gt: the offset ground truth, [batch_size, 2]
        :param candidate_mask: the mask of valid target candidate 
        ------------------------------------------------
        :return:
        """
            
        # 1. 看tar_candidate和tar_candidate tar_num维数是否对齐 
        bs, n, _ = tar_candidate.size()   
        _, num_candidate = gt_candidate.size() 
        assert n == num_candidate, "[TNT-TargetPred]: Error, The num target candidate and the ground truth one-hot vector is not aligned: {} vs {};".format(n, num_candidate) 
        
        
        # 2. 分类cls loss 
        tar_candit_prob, tar_offset_mean = self.forward(feat_in, tar_candidate, candidate_mask) 
        
        n_candidate_loss = F.cross_entropy(tar_candit_prob.transpose(1, 2), gt_candidate.long(), reduction='sum')  
        # n_candidate_loss = F.cross_entropy(tar_candit_prob, gt_candidate.long(), reduction='sum')  
        
        # 3. 筛选TopM个目标点的indics
        _, indices = tar_candit_prob[:,:,1].topk(self.M, dim=1)  
        batch_idx = torch.vstack([torch.arange(0, bs, device=self.device) for _ in range(self.M)]).T
        
        # 4. 计算offset_loss 
        offset_loss = F.smooth_l1_loss(tar_offset_mean[gt_candidate.bool()], gt_offset, reduction='sum')   
        
        # TODO debug checking process
        if debug: 
            pass 
        
        return n_candidate_loss + offset_loss, tar_candidate[batch_idx, indices], tar_offset_mean[batch_idx, indices] 
    
    def inference(self, 
                  feat_in:torch.Tensor, 
                  tar_candidate:torch.Tensor, 
                  candidate_mask=None):
        """
        推理阶段， 只输出M预测目标的概率 
        predict the target end position of the target agent from the target candidates
        feat_in: encode编码轨迹特征, [bs, in_channels] 
        tar_candidate: 候选目标位置(x,y), [bs, N, 2] 
        candidate_mask: the mask of valid target candidate
        """
        return self.forward(feat_in, tar_candidate, candidate_mask)
        

if __name__ == "__main__":
    # 测试代码 
    bs = 16
    in_chs = 64 
    N = 1000 # 1000个Target candidate 
    layer = TargetPred(in_chs, hidden_dim=64, m=50) 
    print("total number of params: ", sum(p.numel() for p in layer.parameters())) 
    
    feat_tensor = torch.randn((bs, 1, in_chs)).float() 
    tar_candidate = torch.randn((bs, N, 2)).float() 
    gt_candit = torch.zeros((bs, N), dtype=torch.bool)  
    # 写一个掩码mask来屏蔽gt_candit 
    candidate_mask = torch.ones((bs, N), dtype=torch.bool) 
    for i in range(50):
        candidate_mask[np.random.randint(0,bs),np.random.randint(0,N)] = False 
        
        
    gt_candit[:, 5] = 1.0 # 每个都是第5个是GT 
    gt_offset = torch.randn((bs, 2)) # tar -> goal offset 
    
    # 测试forward 
    tar_pred, offset_pred = layer(feat_tensor, tar_candidate,candidate_mask)   
    print("shape of pred prob: ", tar_pred.size())
    print("shape of dx and dy: ", offset_pred.size())
    
    # 测试loss 
    loss = layer.loss(feat_tensor, tar_candidate, gt_candit, gt_offset, candidate_mask)
    
    # 测试inference 
    tar_candidate, offset = layer.inference(feat_tensor, tar_candidate, candidate_mask)
    print("shape of tar_candidate: ", tar_candidate.size())
    print("shape of offset: ", offset.size())