# Utilities for meta-RL training and testing
import numpy as np

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def sample_grid(grid):
    xs,ys = np.nonzero(grid)
    num_locations = xs.shape[0]
    rand = np.random.randint(0,num_locations)
    sample_location = (xs[rand],ys[rand])
    return sample_location
