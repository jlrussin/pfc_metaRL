# Training script for meta-reinforcement learning models

# Things to do:
#   -Training loop
#       -Read two papers
#       -Need to build an outer loop for sampling from a distribution of envs
#       -Advantage actor-critic
#       -Should parallelize everything?
#       -Need experience replay?
#   -Write LSTM model, put args into this script
#   -Write two_step environment
#       -Class should parameterize a distribution of environments
#   -Write testing script

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import *
from environments import *
from utils import *

parser = argparse.ArgumentParser()
# Training
parser.add_argument('--trials', type=int, default=1000,
                    help='Number of trials to run')
parser.add_argument('--batch_size', type=int, default=200,
                    help='Number of samples per batch')
parser.add_argument('--gamma', type=float, default=0.999,
                    help='Temporal discounting parameter')
parser.add_argument('--epsilon', type=float, default=0.1,
                    help='Probability of selecting a random action')

# Model
parser.add_argument('--model', choices=['LSTM'],
                    help='Type of model to use')
parser.add_argument('--hidden_dim', type=int, default=400,
                    help='Number of hidden units in each layer')
parser.add_argument('--n_layers', type=int, default=2,
                    help='Number of layers')
parser.add_argument('--bidirectional', type=str2bool, default=True,
                    help='Boolean indicating whether rnn model is bidirectional')
parser.add_argument('--batch_normalization', type=str2bool, default=True,
                    help='Boolean indicating whether to use batch normalization')
parser.add_argument('--load_weights_from', default=None,
                    help='Path to saved weights of model')

# Environment
# TODO: add parameters for two_step environment
parser.add_argument('--environment', default='two_step', choices=['two_step'],
                    help='Channel to use for training')

# Optimization
# TODO: correct optimization (loss, etc.)
parser.add_argument('--loss', default='MSE', choices=['BCE', 'MSE'],
                    help='Loss function to use: Binary cross-entropy (CE) or' +
                    'Mean squared error (MSE)')
parser.add_argument('--learning_rate', type=float, default=1e-3,
                    help='Fixed learning rate for Adam optimizer')
parser.add_argument('--clip_norm', type=float, default=1.0,
                    help='Maximum 2-norm at which gradients will be clipped.')

# Output options
parser.add_argument('--out_data_file', default='../data/out_data.json',
                    help='Path to output data file with training loss data')
parser.add_argument('--checkpoint_path',
                    default=None,
                    help='Path to output saved weights of model')
parser.add_argument('--checkpoint_every', type=int, default=None,
                    help='Trials before evaluating model and saving weights')
parser.add_argument('--record_loss_every', type=int, default=1,
                    help='Trials before printing and recording loss')

def main(args):
    # CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Environment
    if args.environment == 'two_step'
        # TODO: add arguments for two_step_task
        env = Two_step_task()

    # Model
    if args.model == 'LSTM':
        model = LSTM()
        if args.load_weights_from is not None:
            model.load_state_dict(torch.load(args.load_weights_from))

    # Loss function
    if args.loss == 'MSE':
        loss_fn = nn.MSELoss()
    loss_fn = loss_fn.to(device)

    # Optimizer
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Training loop
    # TODO: write training loop (Advantage actor critic)



if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
