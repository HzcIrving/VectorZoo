#-*- coding: utf-8 -*- 
# Parent Trainer 

# 垃圾回收  
# gc.collect --- 释放内存
# gc.enable --- 启动垃圾回收 
import gc  

import os
from matplotlib import axis
from tqdm import tqdm

import json

import torch

import torch.distributed as dist
from torch.utils.data import distributed
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter 

# 需要安装torch_geometric 
# from torch_geometric.data import DataLoader, DataListLoader 

# Argoverse DataProcessing
# from argoverse.evaluation.eval_forecasting import get_displacement_errors_and_miss_rate

# Unified Training Configuration Setting 
class TrainingSettings: 
    loader = None # DataLoader 
    batch_size: int = 1
    num_workers: int = 1
    lr: float = 1e-4  # 初始化学习率 
    betas=(0.9, 0.999) # Adam Setting 
    weight_decay: float = 0.01 # Adam Setting (lr decay)
    warmup_epoch=30 # optimizatioin scheduler param
    with_cuda: bool = False # tag indicating whether using gpu for training
    cuda_device=None # device number 
    multi_gpu: bool = False # whether multi-gpus are using
    enable_log: bool = False
    log_freq: int = 2 # logging frequency in epoch
    save_folder: str = "" 
    verbose: bool = True # whether printing debug messages 
    
    
# TODO  抽象Trainer
class Trainer(object): 
    """
    接口说明 
    1. train(self, epoch):  
    2. eval(self, epoch): 
    3. test(self): 
    4. iteration(self, epoch, dataloader): 
    5. compute_loss(self, data): 
    6. write_log(self, name_str, data, epoch): 
    7. save(self, iter_epoch, loss): 
    8. save_model(self, prefix=""): 
    9. load(self, load_path, mode='c'): 
    10. compute_metric(self, miss_threshold=2.0): 
    """
    def __init__(self, 
                 trainset,  # 训练、评估、测试数据集 
                 evalset, 
                 testset,
                 train_cfg = TrainingSettings(),  # 基类参数配置 TODO 后续需要按照mmdetection进行完善 Config功能
                 ): 
        # Dataloader  
        self.trainset = trainset 
        self.evalset = evalset
        self.testset = testset  
        self.num_workers = train_cfg.num_workers 
        self.batch_size = train_cfg.batch_size
        
        # cuda 
        # determine cuda device id 
        self.multi_gpu = train_cfg.multi_gpu 
        self.cuda_id = train_cfg.cuda_device if train_cfg.with_cuda and train_cfg.cuda_device else 0
        self.device = torch.device("cuda:{}".format(self.cuda_id) if torch.cuda.is_available() and train_cfg.with_cuda else "cpu")
        torch.backends.cudnn.benchmark = True if torch.cuda.is_available() and train_cfg.with_cuda else False     # boost cudnn 
        if 'WORLD_SIZE' in os.environ and self.multi_gpu:
            self.multi_gpu = True if int(os.environ['WORLD_SIZE']) > 1 else False
        else:
            self.multi_gpu = False 
        torch.manual_seed(self.cuda_id)     # seed 
        if self.multi_gpu:
            torch.cuda.set_device(self.cuda_id)
            dist.init_process_group(backend='nccl', init_method='env://')

        # Multi GPU Training Setting  
        assert train_cfg.loader is not None, "loader is not be loaded correctly"  
        self.loader = train_cfg.loader
        if self.multi_gpu: 
            # datset sampler when training with distributed data parallel model
            self.train_sampler = distributed.DistributedSampler(
                self.trainset,
                num_replicas=int(os.environ['WORLD_SIZE']),
                rank=self.cuda_id
            )
            self.eval_sampler = distributed.DistributedSampler(
                self.evalset,
                num_replicas=int(os.environ['WORLD_SIZE']),
                rank=self.cuda_id
            )
            self.test_sampler = distributed.DistributedSampler(
                self.testset,
                num_replicas=int(os.environ['WORLD_SIZE']),
                rank=self.cuda_id
            )

            self.train_loader = self.loader(
                self.trainset,
                batch_size=self.batch_size,
                num_workers=0,
                pin_memory=True,
                shuffle=False,
                sampler=self.train_sampler
            )
            self.eval_loader = self.loader(self.evalset, batch_size=self.batch_size, num_workers=0, sampler=self.eval_sampler)
            # self.eval_loader = self.loader(self.evalset, batch_size=self.batch_size)
            self.test_loader = self.loader(self.testset, batch_size=self.batch_size, num_workers=0, sampler=self.test_sampler)
            # self.test_loader = self.loader(self.testset, batch_size=self.batch_size) 
        else: 
            self.train_loader = self.loader(
                self.trainset,
                batch_size=self.batch_size,
                num_workers=self.num_workers, # 线程数
                pin_memory=True, # 是否将数据加载到GPU内存中
                shuffle=True  # 是否在训练期间对数据进行随机打乱
            )
            self.eval_loader = self.loader(self.evalset, batch_size=self.batch_size, num_workers=self.num_workers)
            # self.eval_loader = self.loader(self.evalset, batch_size=self.batch_size)
            self.test_loader = self.loader(self.testset, batch_size=self.batch_size, num_workers=self.num_workers)
            # self.test_loader = self.loader(self.testset, batch_size=self.batch_size)

        # Model  
        # TNT/MLP ... 
        # -------------------------
        self.model = None  
        # ------------------------- 
        
        # Training Params Config  
        # -- 1. Optim 
        self.lr = train_cfg.lr
        self.betas = train_cfg.betas
        self.weight_decay = train_cfg.weight_decay
        self.optim = None
        self.optm_schedule = None  
        # -- 2. Warmup 
        self.warmup_epoch = train_cfg.warmup_epoch
        
        # Criterion and Metric (评价指标)  
        self.criterion = None
        self.min_eval_loss = None
        self.best_metric = None
        
        # Tensorboard Log  
        self.enable_log = train_cfg.enable_log
        self.save_folder = train_cfg.save_folder
        if not self.multi_gpu or (self.multi_gpu and self.cuda_id == 1):
            self.logger = SummaryWriter(log_dir=os.path.join(self.save_folder, "log"))
        self.log_freq = train_cfg.log_freq
        self.verbose = train_cfg.verbose 
        
        # 内存清理
        gc.enable()
        
    
    def train(self, epoch):
        gc.collect()
        self.model.train() # train 
        return self.iteration(epoch, self.train_loader)
    
    def eval(self, epoch):
        gc.collect()
        self.model.eval() # eval 
        return self.iteration(epoch, self.eval_loader) 
    
    def test(self): 
        raise NotImplementedError 
    
    def iteration(self, epoch, dataloader):
        raise NotImplementedError

    def compute_loss(self, data):
        raise NotImplementedError

    def write_log(self, name_str, data, epoch):
        if not self.enable_log:
            return
        self.logger.add_scalar(name_str, data, epoch) # tensorboard record 
    
    def save(self, iter_epoch, loss):
        """
        save current state of the training and update the minimum loss value
        :param save_folder: str, the destination folder to store the ckpt
        :param iter_epoch: int, ith epoch of current saving checkpoint
        :param loss: float, the loss of current saving state
        :return:
        """
        if self.multi_gpu and self.cuda_id != 1:
            return

        self.min_eval_loss = loss 
        # Save Folder Check 
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder, exist_ok=True) 
            
        # epoch info | optimizer info | min_eval_loss info 
        # dict() former  --> save 
        torch.save({
            "epoch": iter_epoch,
            "model_state_dict": self.model.state_dict() if not self.multi_gpu else self.model.module.state_dict(),
            # "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
            "min_eval_loss": loss
        }, os.path.join(self.save_folder, "checkpoint_iter{}.ckpt".format(iter_epoch))) 
        
        # debug info (Save flag)
        if self.verbose:
            print("【Trainer】: Saving checkpoint to {}...".format(self.save_folder))
        
        
    def save_model(self, prefix=""):
        """
        save current state of the model
        :param prefix: str, the prefix to the model file
        :return:
        """
        if self.multi_gpu and self.cuda_id != 1:
            return

        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder, exist_ok=True)

        # compute the metrics and save 
        # 性能指标
        metric = self.compute_metric()

        # skip model saving if the minADE is not better 
        # 只存最好的model  --- 依据性能指标
        if self.best_metric and isinstance(metric, dict):
            if metric["minADE"] >= self.best_metric["minADE"]:
                print("【Trainer】: Best minADE: {}; Current minADE: {}; Skip model saving...".format(
                    self.best_metric["minADE"],
                    metric["minADE"]))
                return

        # save best metric --- record成绩相关配置
        if self.verbose:
            print("【Trainer】: Best minADE: {}; Current minADE: {}; Saving model to {}...".format(
                self.best_metric["minADE"] if self.best_metric else "Inf",
                metric["minADE"],
                self.save_folder)) 
        self.best_metric = metric # record best score in metric
        metric_stored_file = os.path.join(self.save_folder, "{}_metrics.txt".format(prefix)) # Save_metrics 
        with open(metric_stored_file, 'a+') as f:
            f.write(json.dumps(self.best_metric))
            f.write("\n")

        # save model --- only save model 
        torch.save(
            self.model.state_dict() if not self.multi_gpu else self.model.module.state_dict(),
            # self.model.state_dict(),
            os.path.join(self.save_folder, "{}_{}.pth".format(prefix, type(self.model).__name__))
        )
    
    def load(self, load_path, mode='c'):
            """
            loading function to load the ckpt or model
            :param mode: str, "c" for checkpoint, or "m" for model 
                        "c" 是读save接口存的所有文件， "m"是只读模型文件
            :param load_path: str, the path of the file to be load
            :return:
            """
            if mode == 'c':
                # load ckpt
                ckpt = torch.load(load_path, map_location=self.device)
                try:
                    self.model.load_state_dict(ckpt["model_state_dict"])
                    self.optim.load_state_dict(ckpt["optimizer_state_dict"])
                    self.min_eval_loss = ckpt["min_eval_loss"]
                except:
                    raise Exception("[Trainer]: Error in loading the checkpoint file {}".format(load_path))
            elif mode == 'm':
                try:
                    self.model.load_state_dict(torch.load(load_path, map_location=self.device))
                except:
                    raise Exception("[Trainer]: Error in loading the model file {}".format(load_path))
            else:
                raise NotImplementedError 
        

    def compute_metric(self, miss_threshold=2.0):
        """
        Compute Test Dataset Metric 
        : miss_threshold: float, the threshold of the miss rate [m] e.g. MissRate@2m
        : return: dict, 
        """
        assert self.model, "[Trainer]: No valid model, metrics can't be computed!"
        assert self.testset, "[Trainer]: No test dataset, metrics can't be computed!"

        # 预测轨迹
        forecasting_trajs = {} 
        # 真实轨迹
        gt_trajs = {} 
        
        seq_id = 0 

        # K预测轨迹数(K=6) , horizon T 
        k = self.model.k if not self.multi_gpu else self.model.module.k
        horizon = self.model.horizon if not self.multi_gpu else self.model.module.horizon 
        
        self.model.eval() 
        with torch.no_grad():
            for data in tqdm(self.test_loader):
                bs = data.num_graphs  # 图
                # 转换成 (bs,T,2)的格式，并沿着轨迹进行累加，这么做因为预测的是offset偏移值
                gt = data.y.unsqueeze(1).view(bs, -1, 2).cumsum(axis=1).numpy() 
                
                # inference and transform dimension
                if self.multi_gpu:
                    out = self.model.module.inference(data.to(self.device))
                else:
                    out = self.model.inference(data.to(self.device)) 
                    
                pred_y = out.cpu().numpy() 
                
                # 记录Prediction & GT 
                for batch_id in range(bs):
                    forecasting_trajs[seq_id] = [pred_y_k for pred_y_k in pred_y[batch_id]]
                    gt_trajs[seq_id] = gt[batch_id]
                    seq_id += 1 # t -> T  
            
            TODO 
            # ADE/FDE Compute and miss rate compute 
            metric_results = get_displacement_errors_and_miss_rate( 
                forecasting_trajs,
                gt_trajs,
                k,
                horizon, # T
                miss_threshold # MissRate@2m
            )


if __name__ == "__main__":
    a = torch.ones(3,3,2) 
    b = a.cumsum(axis=1)
    print(b) 
                
        
        