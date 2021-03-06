# Things to do:
#   -Vectorize everything so that multiple trials can be run in parallel

import numpy as np
from utils import *
import matplotlib.pyplot as plt

# Daw two-step task
class Two_step_env():
    def __init__(self,p_common=0.8,r_common=0.9,p_reversal=0.025):
        self.state_dim = 5 # States: start, S1, S2, reward, no_reward
        self.action_dim = 2 # Actions: A1, A2
        self.p_common = p_common # Common transition probability
        self.r_common = r_common # Common reward probability
        self.p_reversal = p_reversal # Probability of reward reversal
        self.rewarded_state = np.random.randint(1,3) # Commonly rewarded state
        self.done = True

    def init_new_trial(self):
        self.state = np.array([1,0,0,0,0]) # Start state vector
        if not self.done:
            print("Warning: initializing new trial even though previous trial" +
                  "was not done")
        self.done = False
        # Decide if reward reversal occurs
        if np.random.rand() < self.p_reversal:
            if self.rewarded_state == 1:
                self.rewarded_state = 2
            else:
                self.rewarded_state = 1

    def step(self,action):
        if self.done:
            print("Need to reinitialize trial before taking another step")
            return
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
            reward = 0
            r_rand = np.random.rand()
            if s == self.rewarded_state:
                if r_rand < self.r_common:
                    self.state = np.array([0,0,0,1,0])
                else:
                    self.state = np.array([0,0,0,0,1])
            else:
                if r_rand > self.r_common:
                    self.state = np.array([0,0,0,1,0])
                else:
                    self.state = np.array([0,0,0,0,1])
        else:
            if s == 3:
                reward = 1
            else:
                reward = 0
            self.done = True
        return self.state, reward, self.done

class Two_step_task():
    def __init__(self,p_common_dist=[0.8,0.8],
                      r_common_dist=[0.9,0.9],
                      p_reversal_dist=[0.025,0.025]):
        self.state_dim = 5 # States: [start, S1, S2,reward,no_reward]
        self.action_dim = 2 # Actions: [A1, A2]
        self.p_common_dist = p_common_dist
        self.r_common_dist = r_common_dist
        self.p_reversal_dist = p_reversal_dist

    def sample(self):
        p_common = np.random.uniform(self.p_common_dist[0],
                                     self.p_common_dist[1])
        r_common = np.random.uniform(self.r_common_dist[0],
                                     self.r_common_dist[1])
        p_reversal = np.random.uniform(self.p_reversal_dist[0],
                                       self.p_reversal_dist[1])
        env = Two_step_env(p_common,r_common,p_reversal)
        return env

# Rocket task
class Rocket_env():
    def __init__(self,p_reversal=0.025,p_reward_reversal=0.5):
        self.state_dim = 6 # States: start1, start2, S1, S2, reward, no_reward
        self.action_dim = 2 # Actions: A1, A2
        self.p_reversal = p_reversal # Probability of reward reversal
        self.p_reward_reversal = p_reward_reversal
        self.rewarded_state = np.random.randint(2,4) # First rewarded state
        self.transition_regime = np.random.randint(0,2) # First transition regime
        self.done = True

    def init_new_trial(self):
        # Update done
        if not self.done:
            print("Warning: initializing new trial even though previous trial" +
                  "was not done")
        self.done = False
        # Start state of each trial is random
        s = np.random.randint(0,2)
        self.state = np.zeros(6)
        self.state[s] = 1
        # Stochastic reversals (reward or transition)
        if np.random.rand() < self.p_reversal:
            if np.random.rand() < self.p_reward_reversal:
                # Reward reversal occurs
                if self.rewarded_state == 2:
                    self.rewarded_state = 3
                else:
                    self.rewarded_state = 2
            else:
                # Transition reversal occurs
                if self.transition_regime == 0:
                    self.transition_regime = 1
                else:
                    self.transition_regime = 0

    def step(self,action):
        if self.done:
            print("Need to reinitialize trial before taking another step")
            return
        s = np.nonzero(self.state)[0]
        if s in [0,1]:
            if s == 0:
                if action == 0:
                    if self.transition_regime == 0:
                        self.state = np.array([0,0,1,0,0,0])
                    else:
                        self.state = np.array([0,0,0,1,0,0])
                else:
                    if self.transition_regime == 0:
                        self.state = np.array([0,0,0,1,0,0])
                    else:
                        self.state = np.array([0,0,1,0,0,0])
            else:
                if action == 0:
                    if self.transition_regime == 0:
                        self.state = np.array([0,0,0,1,0,0])
                    else:
                        self.state = np.array([0,0,1,0,0,0])
                else:
                    if self.transition_regime == 0:
                        self.state = np.array([0,0,1,0,0,0])
                    else:
                        self.state = np.array([0,0,0,1,0,0])

            reward = 0
        elif s in [2,3]:
            reward = 0
            if s == self.rewarded_state:
                self.state = np.array([0,0,0,0,1,0])
            else:
                self.state = np.array([0,0,0,0,0,1])
        else:
            if s == 4:
                reward = 1
            else:
                reward = 0
            self.done = True
        return self.state, reward, self.done

class Rocket_task():
    def __init__(self,p_reversal_dist=[0.025,0.025],
                      p_reward_reversal_dist=[0.5,0.5]):
        self.state_dim = 6
        self.action_dim = 2
        self.p_reversal_dist = p_reversal_dist
        self.p_reward_reversal_dist = p_reward_reversal_dist

    def sample(self):
        p_reversal = np.random.uniform(self.p_reversal_dist[0],
                                       self.p_reversal_dist[1])
        p_reward_reversal = np.random.uniform(self.p_reward_reversal_dist[0],
                                              self.p_reward_reversal_dist[1])
        env = Rocket_env(p_reversal,p_reward_reversal)
        return env

class Rooms_grid_env():
    def __init__(self,grid,reward_location=(1,1)):
        self.grid = grid
        self.reward_location = reward_location
        # Generate dictionary for converting locations to states
        xs,ys = np.nonzero(self.grid)
        num_locs = xs.shape[0]
        self.locs_to_states = {(xs[i],ys[i]):np.eye(num_locs)[i,:] for i in range(num_locs)}
    def init_new_trial(self):
        random_state_loc = sample_grid(self.grid)
        self.state_loc = random_state_loc
        self.state = self.locs_to_states[random_state_loc]
        self.done = False
    def step(self,action):
        if self.done:
            print("Need to reinitialize trial before taking another step")
            return
        # Get new location
        if action == 0: # Move down
            new_loc = (self.state_loc[0]+1, self.state_loc[1])
        elif action == 1: # Move up
            new_loc = (self.state_loc[0]-1, self.state_loc[1])
        elif action == 2: # Move left
            new_loc = (self.state_loc[0], self.state_loc[1]-1)
        elif action == 3: # Move right
            new_loc = (self.state_loc[0], self.state_loc[1]+1)
        # Can't go through walls
        if self.grid[new_loc[0],new_loc[1]] == 1:
            self.state_loc = new_loc
        else:
            self.state_loc = self.state_loc # stay in the same location
        # Get new state
        self.state = self.locs_to_states[self.state_loc]
        # Reward
        if self.state_loc == self.reward_location:
            reward = 1
            self.done = True
        else:
            reward = -0.01
        return self.state, reward, self.done
    def visualize(self):
        grid_with_reward = self.grid.copy()
        grid_with_reward[self.reward_location[0],self.reward_location[1]] = 2
        plt.imshow(grid_with_reward)
        plt.show()

class Rooms_grid_task():
    def __init__(self,room_size=3):
        self.state_dim = 4*room_size**2 + 4
        self.action_dim = 4
        assert room_size % 2 == 1, "Room size must be odd"
        self.room_size = room_size

        # Build grid
        grid = np.zeros([2*room_size+3,2*room_size+3])
        grid[1:room_size+1,1:room_size+1] = 1 # room1
        grid[room_size+2:2*room_size+2,1:room_size+1] = 1 # room2
        grid[1:room_size+1,room_size+2:2*room_size+2] = 1 # room3
        grid[room_size+2:2*room_size+2,room_size+2:2*room_size+2] = 1 # room4
        one_quarter = int((room_size+1)/2)
        grid_center = room_size+1
        three_quarters = grid_center + one_quarter
        grid[grid_center,one_quarter] = 1 # hall 1-2
        grid[one_quarter,grid_center] = 1 # hall 1-3
        grid[three_quarters,grid_center] = 1 # hall 2-4
        grid[grid_center,three_quarters] = 1 # hall 3-4
        self.grid = grid
    def sample(self):
        reward_location = sample_grid(self.grid)
        env = Rooms_grid_env(self.grid,reward_location)
        return env
