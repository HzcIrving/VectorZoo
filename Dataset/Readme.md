## 1. Argoverse数据集

- Argoverse 1 Motion Forecasting数据集 
  - 324557 场景 
    - 每个场景5s长度，用于训练与验证
    - 采样频率10Hz
    - 每个场景都包含以10Hz频率采样的每个被跟踪物体的2DBEV中心 
  - 筛选了1000h的驾驶数据
  - 找到最具有挑战性的片段
    - 十字路口
    - 左转/右转
    - 变道

- BEV图中
  - 绿色 -- ego vehicle 
  - 红色 -- agent of interest 
  - 淡蓝色 -- other objects of interest  