
# 测试说明

## TODO 测试结论
- 终于发现，需要将GAMMA = 1，理由，迷
- UPDATE_GLOBAL_ITER=5 仿真不行，50最好
- 网络个数20，
- 好的结果
"""MAX_GLOBAL_EP=5000,
                UPDATE_GLOBAL_ITER=50,
                GAMMA=1,
                ENTROPY_BETA=0.01,
                LR_A=0.0001,
                LR_C=0.0001, )"""
## 调试历史
### 1
para = A3C.Para(env,  # 环境参数包括state_dim,action_dim,abound,step,reset
                a_constant=True,  # 动作是否是连续
                units_a=500,  # 双层网络，第一层的大小
                units_c=500,  # 双层网络，critic第一层的大小
                MAX_GLOBAL_EP=20000,  # 全局需要跑多少轮数
                UPDATE_GLOBAL_ITER=50,  # 多少代进行一次学习，调小一些学的比较快
                gamma=1,  # 奖励衰减率
                ENTROPY_BETA=0.01,  # 表征探索大小的量，越大结果越不确定
                LR_A=0.0001,  # Actor的学习率
                LR_C=0.0001,  # Crtic的学习率
                sigma_mul=0.01,
                MAX_EP_STEP=510,  # 控制一个回合的最长长度
                train=train_flag  # 表示训练
                )
  #### 失败
  
### 2 目前最有可能
 para = A3C.Para(env,  # 环境参数包括state_dim,action_dim,abound,step,reset
                a_constant=True,  # 动作是否是连续
                units_a=500,  # 双层网络，第一层的大小
                units_c=500,  # 双层网络，critic第一层的大小
                MAX_GLOBAL_EP=10000,  # 全局需要跑多少轮数
                UPDATE_GLOBAL_ITER=50,  # 多少代进行一次学习，调小一些学的比较快
                gamma=1,  # 奖励衰减率
                ENTROPY_BETA=0.01,  # 表征探索大小的量，越大结果越不确定
                LR_A=0.0001,  # Actor的学习率
                LR_C=0.001,  # Crtic的学习率
                sigma_mul=0.1,
                MAX_EP_STEP=510,  # 控制一个回合的最长长度
                train=train_flag  # 表示训练
                )
#### 第一次，结果还可以，但是末端精度还是不够     
#### 第二次，不稳定，效果不行  


### 3 gamma = 0.9
para = A3C.Para(env,  # 环境参数包括state_dim,action_dim,abound,step,reset
                a_constant=True,  # 动作是否是连续
                units_a=500,  # 双层网络，第一层的大小
                units_c=500,  # 双层网络，critic第一层的大小
                MAX_GLOBAL_EP=10000,  # 全局需要跑多少轮数
                UPDATE_GLOBAL_ITER=50,  # 多少代进行一次学习，调小一些学的比较快
                gamma=0.9,  # 奖励衰减率
                ENTROPY_BETA=0.01,  # 表征探索大小的量，越大结果越不确定
                LR_A=0.0001,  # Actor的学习率
                LR_C=0.001,  # Crtic的学习率
                sigma_mul=0.1,
                MAX_EP_STEP=510,  # 控制一个回合的最长长度
                train=train_flag  # 表示训练
                )
#### 第一次，结果不好，没有稳定的意向

### 4  LR_A=0.0001, LR_C=0.01,
para = A3C.Para(env,  # 环境参数包括state_dim,action_dim,abound,step,reset
                a_constant=True,  # 动作是否是连续
                units_a=500,  # 双层网络，第一层的大小
                units_c=500,  # 双层网络，critic第一层的大小
                MAX_GLOBAL_EP=10000,  # 全局需要跑多少轮数
                UPDATE_GLOBAL_ITER=50,  # 多少代进行一次学习，调小一些学的比较快
                gamma=0.9,  # 奖励衰减率
                ENTROPY_BETA=0.01,  # 表征探索大小的量，越大结果越不确定
                LR_A=0.0001,  # Actor的学习率
                LR_C=0.01,  # Crtic的学习率
                sigma_mul=0.1,
                MAX_EP_STEP=510,  # 控制一个回合的最长长度
                train=train_flag  # 表示训练
                )
 #### 失败           
                
## TODO 下一步工作
- 如何实现控制剖面的优化