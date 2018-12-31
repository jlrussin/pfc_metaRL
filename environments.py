# Things to do:
#   -Vectorize everything so that multiple trials can be run in parallel

import numpy as np

class Two_step_env():
    def __init__(self,p_common=0.8,r_common=0.9,p_reversal=0.025):
        self.state_dim = 5 # States: start, S1, S2, reward, no_reward
        self.action_dim = 2 # Actions: A1, A2
        self.p_common = p_common # Common transition probability
        self.r_common = r_common # Common reward probability
        self.p_reversal = p_reversal # Probability of reward reversal
        self.rewarded_state = np.random.randint(1,3) # Commonly rewarded state

    def init_new_trial(self):
        self.state = np.array([1,0,0,0,0]) # State vector (one-hot)
        self.done = False

    def step(self,action):
        s = np.nonzero(self.state)[0]
        if s == 0:
            s_rand = np.random.rand()
            if action == 0:
                if s_rand < self.p_common:
                    self.state = np.array([0,1,0,0,0])
                else:
                    self.state = np.array([0,0,1,0,0])
            else:
                if s_rand < self.p_common:
                    self.state = np.array([0,0,1,0,0])
                else:
                    self.state = np.array([0,1,0,0,0])
            reward = 0
        elif s in [1,2]:
            r_rand = np.random.rand()
            if s == self.rewarded_state:
                if r_rand < self.r_common:
                    self.state = np.array([0,0,0,1,0])
                    reward = 1
                else:
                    self.state = np.array([0,0,0,0,1])
                    reward = 0
            else:
                if r_rand > self.r_common:
                    self.state = np.array([0,0,0,1,0])
                    reward = 1
                else:
                    self.state = np.array([0,0,0,0,1])
                    reward = 0
        else:
            self.state = np.array([1,0,0,0,0])
            reward = 0
            self.done = True
            if np.random.rand() < self.p_reversal:
                if self.rewarded_state == 1:
                    self.rewarded_state = 2
                else:
                    self.rewarded_state = 1
        return self.state, reward, self.done

class Two_step_task():
    def __init__(self,p_common_dist=[0.8,0.8],
                      r_common_dist=[0.9,0.9],
                      p_reversal_dist=[0.025,0.025]):
        self.state_dim = 3 # States: [start, S1, S2]
        self.action_dim = 2 # Actions: [A1, A2]
        self.p_common_dist = p_common_dist
        self.r_common_dist = r_common_dist
        self.p_reversal_dist = p_reversal_dist

    def sample(self):
        p_common = np.random.uniform(self.p_common_dist[0],self.p_common_dist[1])
        r_common = np.random.uniform(self.r_common_dist[0],self.r_common_dist[1])
        p_reversal = np.random.uniform(self.p_reversal_dist[0],self.p_reversal_dist[1])
        env = Two_step_env(p_common,r_common,p_reversal)
        return env
