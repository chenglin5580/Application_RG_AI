
import numpy as np

class SmallOptimalControl(object):

    def __init__(self):

        self.state = self.reset_stable()
        self.state_dim = len(self.state)
        self.action_dim = 1
        self.a_bound = np.array([0, 1])
        self.delta_t = 0.01


    def reset_stable(self):
        self.delta_t = 0.01
        self.x = np.array([0])
        self.t = np.array([0])
        self.state = np.hstack((self.x, self.t))
        return self.state.copy()

    def render(self):
        pass


    def step(self, u_nor):

        # if self.t > 0.9:
        #     self.delta_t = 0.01

        u = u_nor * abs(self.a_bound[1]-self.a_bound[0])/2 + (self.a_bound[1] + self.a_bound[0])/2

        x_dot = self.x + u
        self.x = self.x + self.delta_t*x_dot
        self.t = self.t + self.delta_t
        self.state = np.hstack((self.x, self.t))


        if self.t >= 1:
            done = True
            if abs(self.x-1)<0.01:
                reward = -u * u * self.delta_t*10
            else:
                reward = - u * u * self.delta_t*10 - 1000 * (self.x - 1) * (self.x - 1)
        else:
            done = False
            reward = - u * u * self.delta_t*10

        info = {}
        info['action'] = u
        info['time'] = self.t
        info['reward'] = reward


        return self.state.copy(), reward, done, info





