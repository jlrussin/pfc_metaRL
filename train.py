# Training script for meta-reinforcement learning models

# Things to do
#   -Build script for testing on rooms grid environment
#       -Record hidden states of model under certain conditions
#   -Write code for visualizing results
#       -Visualize paths taken
#       -Comparison with optimal model
#       -MDS/t-SNE for visualizing hidden states

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from models import *
from environments import *
from utils import *

parser = argparse.ArgumentParser()
# Training
parser.add_argument('--episodes', type=int, default=10000,
                    help='Number of episodes for training')
parser.add_argument('--trials', type=int, default=100,
                    help='Number of trials to run')
parser.add_argument('--gamma', type=float, default=0.9,
                    help='Temporal discounting parameter')
parser.add_argument('--epsilon', type=float, default=0.1,
                    help='Probability of selecting a random action')

# Model
parser.add_argument('--model', default='LSTM',choices=['LSTM'],
                    help='Type of model to use')
parser.add_argument('--hidden_dim', type=int, default=48,
                    help='Number of hidden units in each layer')
parser.add_argument('--load_weights_from', default=None,
                    help='Path to saved weights of model')

# Environment
parser.add_argument('--task', default='two_step', choices=['two_step','rocket','rooms_grid'],
                    help='Task to use for training')
parser.add_argument('--p_common_dist',  type=float, nargs=2, default=[0.8,0.8],
                    help='Parameters of uniform distribution for common' +
                    'transition probability in Two_step_task')
parser.add_argument('--r_common_dist',  type=float, nargs=2, default=[0.9,0.9],
                    help='Parameters of uniform distribution for common' +
                    'reward probability in Two_step_task')
parser.add_argument('--p_reversal_dist',  type=float, nargs=2, default=[0.025,0.025],
                    help='Parameters of uniform distribution for reward' +
                    'reversal probability in two step and rocket tasks')
parser.add_argument('--p_reward_reversal_dist',  type=float, nargs=2,
                    default=[0.5,0.5], help='Parameters of uniform' +
                    'distribution for conditional probability of reward' +
                    'reversal given a reversal will occur in rocket task')
parser.add_argument('--room_size',  type=int, default=3,
                    help='Room height and width for rooms_grid task')

# Optimization
parser.add_argument('--beta_v', type=float, default=0.05,
                    help='Weight on value gradient in A3C loss.')
parser.add_argument('--beta_e', type=float, default=0.05,
                    help='Weight on entropy gradient in A3C loss.')
parser.add_argument('--learning_rate', type=float, default=0.0007,
                    help='Fixed learning rate for RMSProp optimizer')
parser.add_argument('--t_max', type=int, default=300,
                    help='Max number of backprop-through-time steps')

# Output options
parser.add_argument('--out_data_file', default='../data/out_data.json',
                    help='Path to output data file with training loss data')
parser.add_argument('--checkpoint_path',
                    default=None,
                    help='Path to output saved weights of model')
parser.add_argument('--checkpoint_every', type=int, default=2000,
                    help='Number of episodes before checkpointing.')
parser.add_argument('--print_every', type=int, default=1,
                    help='Number of episodes before printing average reward.')

def main(args):
    # CUDA
    #use_cuda = torch.cuda.is_available()
    use_cuda = False # Faster on cpu
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Environment
    if args.task == 'two_step':
        task = Two_step_task(args.p_common_dist,
                                  args.r_common_dist,
                                  args.p_reversal_dist)
    elif args.task == 'rocket':
        task = Rocket_task(args.p_reversal_dist,
                           args.p_reward_reversal_dist)
    elif args.task == 'rooms_grid':
        task = Rooms_grid_task(args.room_size)
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

    # Loss function
    def A3C_loss(delta_list,probs_list,a_list):
        loss = torch.tensor(0.0)
        if torch.cuda.is_available():
            loss = loss.to("cuda:0")
        for delta,probs,a in zip(delta_list,probs_list,a_list):
            delta_no_grad = delta.detach()
            L_pi = (torch.log(probs[:,a])*delta_no_grad) # Policy loss
            L_v = args.beta_v*(delta**2) # Value loss
            L_e = args.beta_e*torch.sum(-1*torch.log(probs)*probs) # Entropy loss
            L = (-L_pi + L_v - L_e).squeeze()
            loss += L
        return loss
    loss_fn = A3C_loss

    # Optimizer
    params = model.parameters()
    optimizer = torch.optim.RMSprop(params, lr=args.learning_rate, alpha=0.99)

    # Training loop
    total_rewards = [] # Total rewards earned on each trial
    episode_rewards = [] # Total rewards earned on each trial in this episode
    for episode in range(args.episodes):
        if episode % args.print_every == 0:
            if episode > 0:
                print("Average reward in last episode: ", np.mean(episode_rewards))
                print("Average reward overall: ", np.mean(total_rewards))
            print("Starting episode: ", episode)
        episode_rewards = []
        env = task.sample()
        model.reinitialize()
        t = 0 # Number of steps since .detach() was last called
        r = 0
        a = None
        for trial in range(args.trials):

            # Zero gradients
            optimizer.zero_grad()

            # Reset environment for new trial
            env.init_new_trial()
            s = env.state
            done = False

            # New lists for recording v, probs, a, r
            v_history = []
            probs_history = []
            a_history = []
            r_history = []

            # Run a trial
            total_reward = 0 # Total reward earned on this trial
            while not done:
                t += 1

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

                # Update record of trial history
                v_history.append(v)
                probs_history.append(probs)
                a_history.append(a)
                r_history.append(r)
                total_reward += r

                # Compute sequence of losses, backpropagate, update params
                if t == args.t_max or done:
                    delta_list,probs_list,a_list = [],[],[]
                    if done:
                        R = 0
                    else:
                        R = v
                    for i in range(len(v_history)-1,-1,-1):
                        R = r_history[i] + args.gamma*R
                        delta_list.append((R - v_history[i]))
                        probs_list.append(probs_history[i])
                        a_list.append(a_history[i])
                    loss = loss_fn(delta_list,probs_list,a_list)
                    loss.backward(retain_graph=True) # Need to retain graph for later updates
                    optimizer.step()
                    optimizer.zero_grad()
                    if t == args.t_max:
                        model.h = model.h.detach()
                        model.c = model.c.detach()
                        t = 0
                    v_history = []
                    probs_history = []
                    a_history = []
                    r_history = []
            total_rewards.append(total_reward)
            episode_rewards.append(total_reward)

        # Save weights
        if episode % args.checkpoint_every == 0:
            print("Saving model weights to: ", args.checkpoint_path)
            if args.checkpoint_path is not None:
                torch.save(model.state_dict(),args.checkpoint_path)

    # Write output file
    print("Writing results to: ", args.out_data_file)
    training_record = {'total_rewards':total_rewards}
    with open(args.out_data_file,'w') as f:
        f = json.dump(training_record,f)

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
