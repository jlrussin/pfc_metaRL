# Training script for meta-reinforcement learning models

# Things to do:
#   Record states, actions, env.rewarded_state, rewards, etc.
#   -Questions:
#       -Advantage actor-critic
#           -Need to accumulate gradients over multiple trials?
#               -Straightforward in PyTorch: just update every n trials
#           -Should parallelize everything with multiple threads interacting with
#            multiple copies of the environment?
#               -With different exploration policies
#               -Don't need any CUDA stuff?
#   -Write testing script

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
parser.add_argument('--task', default='two_step', choices=['two_step'],
                    help='Channel to use for training')
parser.add_argument('--p_common_dist',  type=float, nargs=2, default=[0.8,0.8],
                    help='Parameters of uniform distribution for common' +
                    'transition probability in Two_step_task')
parser.add_argument('--r_common_dist',  type=float, nargs=2, default=[0.9,0.9],
                    help='Parameters of uniform distribution for common' +
                    'reward probability in Two_step_task')
parser.add_argument('--p_reversal_dist',  type=float, nargs=2, default=[0.025,0.025],
                    help='Parameters of uniform distribution for reward' +
                    'reversal probability in Two_step_task')

# Optimization
parser.add_argument('--beta_v', type=float, default=0.05,
                    help='Weight on value gradient in A3C loss.')
parser.add_argument('--beta_e', type=float, default=0.05,
                    help='Weight on entropy gradient in A3C loss.')
parser.add_argument('--learning_rate', type=float, default=0.0007,
                    help='Fixed learning rate for RMSProp optimizer')

# Output options
parser.add_argument('--out_data_file', default='../data/out_data.json',
                    help='Path to output data file with training loss data')
parser.add_argument('--checkpoint_path',
                    default=None,
                    help='Path to output saved weights of model')

def main(args):
    # CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Environment
    if args.task == 'two_step':
        task = Two_step_task(args.p_common_dist,
                                  args.r_common_dist,
                                  args.p_reversal_dist)
    state_dim = task.state_dim
    action_dim = task.action_dim

    # Model
    if args.model == 'LSTM':
        model = LSTM(state_dim=state_dim,
                     action_dim=action_dim,
                     hidden_dim=args.hidden_dim)
    if args.load_weights_from is not None:
        model.load_state_dict(torch.load(args.load_weights_from))
    model.to(device)

    print("Model is cuda after initializing: ", next(model.parameters()).is_cuda)

    # Loss function
    def A3C_loss(delta_list,probs_list,a_list):
        loss = torch.tensor(0.0)
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
    total_rewards = [] # List of total rewards earned on each trial
    for episode in range(args.episodes):
        if episode % 5 == 0:
            print("Starting episode: ", episode)
            print("Average reward per trial: ", np.mean(total_rewards))
        env = task.sample()
        model.reinitialize()

        print("Model is cuda after reinitializing: ", next(model.parameters()).is_cuda)

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
            T = 0
            total_reward = 0 # Total reward earned on this trial
            while not done:
                T += 1

                # Convert state, previous action and previous reward to torch.tensors
                s = torch.tensor(s).type(torch.FloatTensor).to(device)
                a_prev = torch.zeros(action_dim,dtype=torch.float).to(device)
                if a is not None:
                    a_prev[a] = 1
                r_prev = torch.tensor(r).type(torch.FloatTensor).to(device)

                print("Episode :", episode, "Trial :", trial, "T: ", T)
                print("s is cuda: ", s.is_cuda)
                print("a_prev is cuda: ", a_prev.is_cuda)
                print("r_prev is cuda: ", r_prev.is_cuda)
                print("Model is cuda: ", next(model.parameters()).is_cuda)

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
            total_rewards.append(total_reward)

            # Compute sequence of losses, backpropagate, and update params
            delta_list,probs_list,a_list = [],[],[]
            R = 0
            for t in range(T-1,-1,-1):
                R = r_history[t] + args.gamma*R
                delta_list.append((R - v_history[t]))
                probs_list.append(probs_history[t])
                a_list.append(a_history[t])
            loss = loss_fn(delta_list,probs_list,a_list)
            loss.backward(retain_graph=True) # Need to retain graph for later updates
            optimizer.step()

    # Save weights
    if args.checkpoint_path is not None:
        torch.save(model.state_dict(),
                   args.encoder_checkpoint_path)

    # Write output file
    print("Writing results to: ", args.out_data_file)
    training_record = {'total_rewards':total_rewards}
    with open(args.out_data_file,'w') as f:
        f = json.dump(training_record,f)

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
