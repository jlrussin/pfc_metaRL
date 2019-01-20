# Script for extracting hidden states of LSTM in rooms_grid environment

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

# Model
parser.add_argument('--model', default='LSTM',choices=['LSTM'],
                    help='Type of model to use')
parser.add_argument('--hidden_dim', type=int, default=48,
                    help='Number of hidden units in each layer')
parser.add_argument('--load_weights_from', default=None,
                    help='Path to saved weights of model')

# Environment
parser.add_argument('--room_size',  type=int, default=3,
                    help='Room height and width for rooms_grid task')

# Output options
parser.add_argument('--out_data_file', default='../data/rooms_hidden_states.npy',
                    help='Path to output data file with testing results')

def main(args):
    #use_cuda = torch.cuda.is_available()
    use_cuda = False # Faster on cpu
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Environment: rooms_grid_task only
    task = Rooms_grid_task(args.room_size)
    env = task.sample()
    env.init_new_trial()
    locs = env.locs_to_states.keys()
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

    # Array for recording results
    hidden_state_data = np.zeros([state_dim, 2+args.hidden_dim])

    # Start model in each state and record hidden state
    with torch.no_grad():
        for i,(row,col) in enumerate(locs):
            # First two columns represent state location
            hidden_state_data[i,0] = row
            hidden_state_data[i,1] = col
            model.reinitialize()
            s = env.locs_to_states[(row,col)]

            # Convert state, previous action and previous reward to torch.tensors
            s = torch.tensor(s).type(torch.FloatTensor).to(device)
            a_prev = torch.zeros(action_dim,dtype=torch.float).to(device)
            r_prev = torch.tensor(0).type(torch.FloatTensor).to(device)

            # Generate action and value prediction
            probs,v = model(s,a_prev,r_prev)

            # Record hidden state
            hidden_state = model.h.numpy()
            hidden_state_data[i,2:] = hidden_state

    # Write output file
    print("Writing results to: ", args.out_data_file)
    np.save(args.out_data_file,hidden_state_data)

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
