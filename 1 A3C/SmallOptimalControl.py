
import numpy as np

class SmallOptimalControl(object):

    def __init__(self):

        self.state = self.reset_stable()
        self.state_dim = len(self.state)
        self.action_dim = 1
        self.a_bound = np.array([0, 1])
        self.delta_t = 0.01


    def reset_stable(self):
        self.x = np.array([0])
        self.t = np.array([0])
        self.state = np.hstack((self.x, self.t))
        return self.state.copy()

    def render(self):
        pass


    def step(self, u):

        u = u / 2 + 0.5

        x_dot = self.x + u
        self.x = self.x + self.delta_t*x_dot
        self.t = self.t + self.delta_t
        self.state = np.hstack((self.x, self.t))


        if self.t >= 1:
            done = True
            # if abs(self.x-1)<0.01:
            #     reward = -u * u /100
            # else:
            reward = - u * u / 10 - 1000 * (self.x - 1) * (self.x - 1)
        else:
            done = False
            reward = - u * u / 10

        info = {}
        info['action'] = u
        info['time'] = self.t
        info['reward'] = reward


        return self.state.copy(), reward, done, info





