# Testing script for meta-reinforcement learning models

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from models import *
from environments import *
from utils import *

parser = argparse.ArgumentParser()
# Training
parser.add_argument('--episodes', type=int, default=300,
                    help='Number of episodes for training')
parser.add_argument('--trials', type=int, default=100,
                    help='Number of trials to run')
parser.add_argument('--gamma', type=float, default=0.9,
                    help='Temporal discounting parameter')

# Model
parser.add_argument('--model', default='LSTM',choices=['LSTM'],
                    help='Type of model to use')
parser.add_argument('--hidden_dim', type=int, default=48,
                    help='Number of hidden units in each layer')
parser.add_argument('--load_weights_from', default=None,
                    help='Path to saved weights of model')

# Environment
parser.add_argument('--task', default='two_step', choices=['two_step'],
                    help='Parameterizes a distribution of environments.')
parser.add_argument('--p_common_dist',  type=float, nargs=2, default=[0.8,0.8],
                    help='Parameters of uniform distribution for common' +
                    'transition probability in Two_step_task')
parser.add_argument('--r_common_dist',  type=float, nargs=2, default=[0.9,0.9],
                    help='Parameters of uniform distribution for common' +
                    'reward probability in Two_step_task')
parser.add_argument('--p_reversal_dist',  type=float, nargs=2, default=[0.025,0.025],
                    help='Parameters of uniform distribution for reward' +
                    'reversal probability in Two_step_task')

# Output options
parser.add_argument('--out_data_file', default='../data/test_data.npy',
                    help='Path to output data file with testing results')

def main(args):
    # CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Environment
    task = Two_step_task(args.p_common_dist,
                              args.r_common_dist,
                              args.p_reversal_dist)
    state_dim = task.state_dim
    action_dim = task.action_dim

    # Model
    if args.model == 'LSTM':
        model = LSTM(state_dim=state_dim,
                     action_dim=action_dim,
                     hidden_dim=args.hidden_dim,
                     device=device)
    if args.load_weights_from is not None:
        model.load_state_dict(torch.load(args.load_weights_from))
    model.to(device)
    model.eval()

    # Construct empty dataframe to record testing results
    col_names = ['Episode','Trial','T','State','Action','Reward','Rewarded_state']
    n_rows = args.episodes*args.trials*3
    n_cols = len(col_names)
    df = np.full_like(np.zeros([n_rows,n_cols]),np.nan)
    row = 0 # keep track of what row we're in

    # Testing loop
    with torch.no_grad():
        for episode in range(args.episodes):
            if episode % 5 == 0:
                print("Starting episode: ", episode)
            env = task.sample()
            model.reinitialize()
            r,a = 0,None
            for trial in range(args.trials):
                # Reset environment for new trial
                env.init_new_trial()
                s = env.state
                done = False
                # Run a trial
                T = 0
                while not done:
                    T += 1

                    # Record some data
                    df[row,0] = episode #episode number
                    df[row,1] = trial #trial number
                    df[row,2] = T #timestep
                    df[row,3] = np.nonzero(np.array(s))[0][0] #state
                    df[row,6] = env.rewarded_state #note: recorded before step

                    # Convert state, previous action and previous reward to torch.tensors
                    s = torch.tensor(s).type(torch.FloatTensor).to(device)
                    a_prev = torch.zeros(action_dim,dtype=torch.float).to(device)
                    if a is not None:
                        a_prev[a] = 1
                    r_prev = torch.tensor(r).type(torch.FloatTensor).to(device)

                    # Generate action and value prediction
                    probs,v = model(s,a_prev,r_prev)
                    m = Categorical(probs)
                    a = m.sample()

                    # Take a step in the environment
                    s,r,done = env.step(a)

                    # Record the rest of the row's data
                    df[row,4] = a.item() #action
                    df[row,5] = r #reward

                    # Update row
                    row += 1

    # Write output file
    print("Writing results to: ", args.out_data_file)
    np.save(args.out_data_file,df)

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)