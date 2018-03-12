
"""
DDPG is Actor-Critic based algorithm
Designer: Lin Cheng  17.08.2017
"""

########################### Package  Input  #################################
# import DDPG
from DDPG import DDPG
from SmallOptimalControl import SmallOptimalControl as Object_AI
import numpy as np
import matplotlib.pyplot as plt

############################ Hyper Parameters #################################

max_Episodes = 20000
max_Ep_Steps = 2000
rendering = False
############################ Object and Method  ####################################

env = Object_AI()

s_dim = env.state_dim
print("环境状态空间维度为", s_dim)
print('-----------------------------\t')
a_dim = env.action_dim
print("环境动作空间维度为", a_dim)
print('-----------------------------\t')
a_bound = env.a_bound
print("环境动作空间的上界为", a_bound)
print('-----------------------------\t')

RLmethod = DDPG(
            env.action_dim,  # 动作的维度
            env.state_dim,  # 状态的维度
            env.a_bound,  # 动作的上下限
            e_greedy_end=0.05,  # 最后的探索值 0.1倍幅值
            e_liner_times=1500*50,  # 探索值经历多少次学习变成e_end
            epilon_init=0.5,  # 表示1倍的幅值作为初始值
            LR_A=0.0001,  # Actor的学习率
            LR_C=0.001,  # Critic的学习率
            GAMMA=0.9,  # 衰减系数
            TAU=0.01,  # 软替代率，例如0.01表示学习eval网络0.01的值，和原网络0.99的值
            MEMORY_SIZE=30000,  # 记忆池容量
            BATCH_SIZE=1280,  # 批次数量
            units_a=100,  # Actor神经网络单元数
            units_c=100,  # Crtic神经网络单元数
            # train=True  # 训练的时候有探索
            train=False  # 训练的时候有探索
            )

###############################  training  ####################################

if RLmethod.train:
    for i in range(max_Episodes):
        state_now = env.reset_random()
        ep_reward = 0

        state_now_sequence = np.empty((0, 2))
        action_sequence = np.empty((0, 1))
        reward_sequence = np.empty((0, 1))
        state_next_sequence = np.empty((0, 2))
        done_sequence = np.empty((0, 1))

        for j in range(max_Ep_Steps):

            # Add exploration noise
            action = RLmethod.choose_action(state_now)
            state_next, reward, done, info = env.step(action)

            state_now_sequence = np.vstack((state_now_sequence, state_now))
            action_sequence = np.vstack((action_sequence, action))
            reward_sequence = np.vstack((reward_sequence, reward))
            state_next_sequence = np.vstack((state_next_sequence, state_next))
            done_sequence = np.vstack((done_sequence, done))

            RLmethod.learn()

            state_now = state_next
            ep_reward += reward

            if done:
                x_error = env.x - 1
                print('Episode:', i, ' Reward: %.5f' % ep_reward, 'Explore: %.5f' % RLmethod.epsilon,
                      'x_error : %.5f' % x_error)
                break

        TD_n = 20
        for kk in range(len(state_now_sequence[:, 0])):
            if kk + TD_n - 1 < len(state_now_sequence[:, 0]) - 1:
                state_now = state_now_sequence[kk, :]
                action = action_sequence[kk, 0]
                reward = np.sum(reward_sequence[kk: kk + TD_n, 0])
                state_next = state_next_sequence[kk + TD_n - 1, :]
                done = False
            else:
                state_now = state_now_sequence[kk, :]
                action = action_sequence[kk, 0]
                reward = np.sum(reward_sequence[kk:, 0])
                state_next = state_next_sequence[-1, :]
                done = True
            RLmethod.store_transition(state_now, action, reward, state_next, np.array([done * 1.0]))

        # if (var < 0.05) and (ep_reward > - 0.4):
        if (RLmethod.epsilon <= RLmethod.epsilon_end):
            RLmethod.net_save()
            break
else:
    plt.axis([-0.1, 1, -0.1, 1.2])
    plt.ion()
    plt.grid(color='g', linewidth='0.3', linestyle='--')

    # state_now = env.reset_stable()
    state_now = env.reset_random()
    ep_reward = 0
    state_now_sequence = np.empty((0, 2))
    action_sequence = np.empty((0, 1))
    reward_sequence = np.empty((0, 1))
    state_next_sequence = np.empty((0, 2))
    done_sequence = np.empty((0, 1))

    for j in range(max_Ep_Steps):

        # Add exploration noise
        action = RLmethod.choose_action(state_now)
        state_next, reward, done, info = env.step(action)


        plt.scatter(env.t, env.x, color='b')
        plt.scatter(env.t, action, color='r')
        plt.pause(0.001)

        state_now = state_next
        ep_reward += reward

        if done:
            x_error = env.x - 1
            print('Reward: %.5f' % ep_reward, 'x_error : %.5f' % x_error)
            plt.scatter(env.t, env.x, color='b')
            plt.scatter(env.t, action, color='r')
            plt.pause(100000000)
            break








