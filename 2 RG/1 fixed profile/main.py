"""
DDPG is Actor-Critic based algorithm
Designer: Lin Cheng  17.08.2017
"""

########################### Package  Input  #################################

import ReentryGuidance as RG
import numpy as np
import matplotlib.pyplot as plt

############################ Hyper Parameters #################################

max_Episodes = 20000
max_Ep_Steps = 20000
rendering = False
############################ Object and Method  ####################################

env = RG.ReentryGuidance()

weight = 0.4571

# 1 case1, 2 case2, 3 case3, 4 case4, 5 random
observation = env.reset(1)

ob_profile = np.empty((0, 6))
Q_dot_profile = np.empty((0,))
ny_profile = np.empty((0,))
q_profile = np.empty((0,))
tht_profile = np.empty((0,))


for j in range(max_Ep_Steps):

    observation_, reward, done, info = env.step(weight)

    # memorize the profile
    Q_dot_profile = np.hstack((Q_dot_profile, info['Q_dot']))
    ny_profile = np.hstack((ny_profile, info['ny']))
    q_profile = np.hstack((q_profile, info['q']))
    tht_profile = np.hstack((tht_profile, info['tht']))
    ob_profile = np.vstack((ob_profile, observation))

    observation = observation_

    if observation[2] < env.vf:
        break

# np.save('result_case1secant.npy', state_sequence)


r_profile = ob_profile[:, 0]
h_profile = r_profile - env.R0 * 1000
range_profile = ob_profile[:, 1]
v_profile = ob_profile[:, 2]
theta_profile = ob_profile[:, 3]


plt.figure(num=1)
plt.plot(range_profile * env.R0, h_profile / 1000)
plt.grid()

plt.figure(num=2)
plt.plot(v_profile, h_profile / 1000, color='blue', linewidth=1.0, linestyle='-')
plt.plot(env.vv_HV[0], env.H_up_HV[0], color='red', linewidth=1.0, linestyle='--')
plt.plot(env.vv_HV[0], env.H_down_HV[0], color='red', linewidth=1.0, linestyle='-')
plt.grid()


plt.figure(num=3)
plt.plot(v_profile, tht_profile, color='blue', linewidth=1.0, linestyle='-')
plt.plot(env.vv_HV[0], env.tht_up[0], color='red', linewidth=1.0, linestyle='--')
plt.plot(env.vv_HV[0], env.tht_down[0], color='red', linewidth=1.0, linestyle='-')
plt.grid()

plt.figure(num=4)
plt.plot(Q_dot_profile, color='blue', linewidth=1.0, linestyle='-')
plt.grid()

plt.figure(num=5)
plt.plot(ny_profile, color='blue', linewidth=1.0, linestyle='-')
plt.grid()

plt.figure(num=6)
plt.plot(q_profile, color='blue', linewidth=1.0, linestyle='-')
plt.grid()


plt.show()


