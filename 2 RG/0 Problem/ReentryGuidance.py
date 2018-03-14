import numpy as np
import scipy.io as sio
import scipy.interpolate as interploate
import math


class ReentryGuidance(object):

    def __init__(self):

        # 仿真数据
        self.delta_t = 1
        self.state_dim = 4
        self.action_dim = 1
        self.a_bound = np.array([0, 1])  # 0到1

        # 飞行器总体参数
        self.m0 = 907  # kg
        self.S = 0.35  # m2
        self.R0 = 6378  # Km
        self.g0 = 9.8  # m/s^2
        self.Rd = 0.1
        self.C1 = 11030
        self.Vc = (self.g0 * self.R0 * 1000) ** 0.5
        self.Tc = (self.R0 * 1000 / self.g0) ** 0.5
        self.beta = 7200
        self.rho0 = 1.225

        # 飞行器初始位置信息
        self.h0 = 100  # Km
        self.r0 = self.R0 + self.h0  # Km
        self.v0 = 7200  # m/s
        self.theta0 = -2 / 180 * math.pi  # 弧度
        self.chi0 = 55 / 180 * math.pi  # 弧度
        self.gama0 = 160 / 180 * math.pi  # 弧度
        self.phi0 = 5 / 180 * math.pi  # 弧度
        self.range0 = 0  # Km
        self.Q0 = 0
        self.Osci0 = 0
        self.X0_2 = np.hstack((self.r0 * 1000, self.range0, self.v0, self.theta0, self.Q0, self.Osci0))

        # 目标位置信息
        self.gamaT0 = 235 / 180 * math.pi  # 弧度
        self.phiT0 = 25 / 180 * math.pi  # 弧度
        self.vT0 = 0  # 弧度
        self.chiT0 = 0  # 弧度
        self.range_need = 8332  # Km

        # 约束条件
        self.hf = 20
        self.vf = 1800
        self.Q_dot_max_allow = 1200  # Kw/m2
        self.n_max_allow = 4  # g0
        self.q_max_allow = 200  # Kpa

        # 求解好的走廊
        self.Corridor = sio.loadmat('./Corridor.mat')
        self.vv_HV = self.Corridor['vv_HV']
        self.H_up_HV = self.Corridor['H_up_HV']
        self.H_down_HV = self.Corridor['H_down_HV']
        self.tht_up = self.Corridor['Compound_tht_up']
        self.tht_down = self.Corridor['Compound_tht_down']

    def reset(self, reset_flag):

        if reset_flag == 1:
            # case 1 割线热为52500  weight 0.4571
            self.state = self.X0_2
            ## state_extraction
            state_feature_now = self.State_Feature_extraction()
            return state_feature_now.copy()

        elif reset_flag == 2:
            # case 2  割线热为49468
            self.range0 = (self.range_need - 7429) / self.R0
            self.X0_2 = np.hstack((self.r0 * 1000, self.range0, self.v0, self.theta0, self.Q0, self.Osci0))
            self.state = self.X0_2
            ## state_extraction
            state_feature_now = self.State_Feature_extraction()
            return state_feature_now.copy()
        elif reset_flag == 3:
            # case 3  割线热为54278
            self.range0 = (self.range_need - 8929.1) / self.R0
            self.X0_2 = np.hstack((self.r0 * 1000, self.range0, self.v0, self.theta0, self.Q0, self.Osci0))
            self.state = self.X0_2
            ## state_extraction
            state_feature_now = self.State_Feature_extraction()
            return state_feature_now.copy()
        elif reset_flag == 4:
            # case 4  割线热为56015
            self.range0 = (self.range_need - 9562) / self.R0
            self.X0_2 = np.hstack((self.r0 * 1000, self.range0, self.v0, self.theta0, self.Q0, self.Osci0))
            self.state = self.X0_2
            ## state_extraction
            state_feature_now = self.State_Feature_extraction()
            return state_feature_now.copy()
        elif reset_flag == 5:
            delta_h = np.random.uniform(-5, 5, 1)
            delta_v = np.random.uniform(-30, 30, 1)
            self.range0 = np.random.uniform(-1500, 1000, 1) / self.R0
            self.X0_2 = np.hstack(
                ((self.r0 + delta_h) * 1000, self.range0, self.v0 + delta_v, self.theta0, self.Q0, self.Osci0))
            self.state = self.X0_2
            ## state_extraction
            state_feature_now = self.State_Feature_extraction()
            return state_feature_now.copy()

    def step(self, weight):

        # 积分方程, 单步积分
        k1, info = self.Move_equation2(weight)
        self.state = self.state + self.delta_t * k1

        # reward calculation
        reward, done = self.reward_cal(info)

        ## state_extraction
        state_feature_now = self.State_Feature_extraction()

        return state_feature_now.copy(), reward, done, info

    def reward_cal(self, info):

        ## path constraint calculation
        # Q_dot_exceed = np.max(np.array([0, info['Q_dot']-self.Q_dot_max_allow]))
        Q_dot_exceed = 0
        ny_exceed = np.max(np.array([0, info['ny'] - self.n_max_allow]))
        q_exceed = np.max(np.array([0, info['q'] - self.q_max_allow]))
        if np.max([Q_dot_exceed, ny_exceed, q_exceed]) > 0:
            print('chaole', 'Q_dot', info['Q_dot'], 'ny', info['ny'], 'q', info['q'])

        w_Q_dot = 1  # 1
        w_ny = 1  # 10
        w_q = 0.1  # 1
        ## reward calculation
        reward_sum = info['Q_dot'] * self.delta_t / 50000 + \
                     w_Q_dot * Q_dot_exceed + w_ny * ny_exceed + w_q * q_exceed

        if self.state[2] < self.vf:
            done = True
            range_error = abs((self.state[1] * self.R0) - self.range_need)
            hf_error = abs((self.state[0] / 1000 - self.R0) - self.hf)
            reward = - reward_sum - 10 * (range_error / 200) ** 2 - 0.1 * (hf_error) ** 2
        else:
            done = False
            reward = - reward_sum
        return reward, done

    def V2Alpha(self):
        v = self.state[2]
        v1 = 3100
        v2 = 4700
        alpha_max = 20
        alpha_CLCD_max = 8.5
        if v < v1:
            alpha = alpha_CLCD_max
        elif v < v2:
            alpha = alpha_CLCD_max + (alpha_max - alpha_CLCD_max) / (v2 - v1) * (v - v1)
        else:
            alpha = alpha_max

        return alpha

    def V2Tht(self, weight):
        v = self.state[2]

        if v > self.vv_HV.max():
            v = self.vv_HV.max()
        elif v < self.vv_HV.min():
            v = self.vv_HV.min()

        tht_design = (1 - weight) * self.tht_up[0] + weight * self.tht_down[0]

        # "nearest","zero"为阶梯插值
        # slinear 线性插值
        # "quadratic","cubic" 为2阶、3阶B样条曲线插值
        f = interploate.interp1d(self.vv_HV[0], tht_design, kind='quadratic')
        return f(v)

    def H2g(self):
        # h的单位为m
        r = self.state[0]
        h = r - self.R0 * 1000
        g = self.g0 * (self.R0 * 1000) ** 2 / (self.R0 * 1000 + h) ** 2
        return g

    def H2rho(self):
        # h的单位为m
        r = self.state[0]
        h = r - self.R0 * 1000
        rho = self.rho0 * math.exp(-h / self.beta)
        return rho

    def Move_equation2(self, weight):

        # state decompose
        r = self.state[0]
        v = self.state[2]
        theta = self.state[3]

        # environment calculation
        g = self.H2g()
        rho = self.H2rho()

        # control
        alpha = self.V2Alpha()
        tht = self.V2Tht(weight)
        Cl, Cd = self.AlphaMa2ClCd(alpha, v / 340)
        q = 0.5 * rho * v ** 2
        L = q * Cl * self.S
        D = q * Cd * self.S

        # 运动方程
        r_dot = v * math.sin(theta)
        range_dot = v * math.cos(theta) / r
        v_dot = -D / self.m0 - g * math.sin(theta)
        theta_dot = 1 / v * (L * math.cos(tht / 180 * math.pi) / self.m0 - (g - v ** 2 / r) * math.cos(theta))
        Q_dot = self.C1 / math.sqrt(self.Rd) * (rho / self.rho0) ** 0.5 * (v / self.Vc) ** 3.15
        Osci_dot = 0

        state_dot = np.hstack((r_dot, range_dot, v_dot, theta_dot, Q_dot, Osci_dot))

        # path constraint
        # Q_dot = Q_dot
        ny = math.sqrt((L / self.m0 / self.g0) ** 2 + (L / self.m0 / self.g0) ** 2)
        # q= q

        ## 万能的info
        info = {"Q_dot": Q_dot, "ny": ny, "q": q / 1000, "tht": tht, "alpha": tht}

        return state_dot, info

    def AlphaMa2ClCd(self, alpha, mach):

        p00 = -0.2892
        p10 = 0.07719
        p01 = 0.03159
        p20 = -0.0006198
        p11 = -0.007331
        p02 = 0.0003381
        p21 = 8.544e-05
        p12 = 0.0003569
        p03 = -0.0001313
        p22 = -2.406e-06
        p13 = -5.55e-06
        p04 = 3.396e-06

        CL = p00 + p10 * alpha + p01 * mach + p20 * alpha ** 2 + p11 * alpha * mach + p02 * mach ** 2 + \
             p21 * alpha ** 2 * mach + p12 * alpha * mach ** 2 + p03 * mach ** 3 + p22 * alpha ** 2 * mach ** 2 \
             + p13 * alpha * mach ** 3 + p04 * mach ** 4

        p00 = 0.3604
        p10 = -0.02188
        p01 = -0.07648
        p20 = 0.001518
        p11 = 0.00450
        p02 = 0.0058
        p21 = -0.0001462
        p12 = -0.0001754
        p03 = -0.0002027
        p22 = 4.783e-06
        p13 = 1.025e-06
        p04 = 2.99e-06

        CD = p00 + p10 * alpha + p01 * mach + p20 * alpha ** 2 + p11 * alpha * mach + p02 * mach ** 2 + \
             p21 * alpha ** 2 * mach + p12 * alpha * mach ** 2 + p03 * mach ** 3 + p22 * alpha ** 2 * mach ** 2 \
             + p13 * alpha * mach ** 3 + p04 * mach ** 4

        return CL, CD

    def State_Feature_extraction(self):

        r = self.state[0]
        h = r - self.R0 * 1000
        range = self.state[1]
        v = self.state[2]
        theta = self.state[3]

        h_feature = (h / 1000 - (self.h0 + self.hf) / 2) / (self.h0 - self.hf)
        range_feature = (self.range_need - range * self.R0) / self.range_need
        v_feature = (v - (self.v0 + self.vf) / 2) / (self.v0 - self.vf)
        theta_feature = theta / (0.3)

        state_feature = np.hstack((h_feature, range_feature, v_feature, theta_feature))

        return state_feature
