#!/usr/bin/env bash

# To run: sbatch -p local -A ecortex --mem=8G --time=48:00:00 --gres=gpu:1 scripts/train_rocket_reward_test_all.sh

export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate pytorch0.4.1

echo "Training metaRL system on rocket task with reward reversals only"

python train.py \
--episodes 10000 \
--trials 100 \
--task rocket \
--p_reversal_dist 0.025 0.025 \
--p_reward_reversal_dist 1.0 1.0 \
--beta_v 0.05 \
--beta_e 0.05 \
--learning_rate 0.001 \
--out_data_file results/train_rocket_reward.json \
--checkpoint_path ../data/model_weights/LSTM_rocket_reward.pt

echo "Testing metaRL system trained on rocket task with reward reversals only"
echo "on rocket task with reward reversals only"

python test.py \
--episodes 300 \
--trials 100 \
--load_weights_from ../data/model_weights/LSTM_rocket_reward.pt \
--task rocket \
--p_reversal_dist 0.025 0.025 \
--p_reward_reversal_dist 1.0 1.0 \
--out_data_file results/test_rocket_reward_on_rocket_reward.npy

echo "Testing metaRL system trained on rocket task with reward reversals only"
echo "on rocket task with transition reversals only"

python test.py \
--episodes 300 \
--trials 100 \
--load_weights_from ../data/model_weights/LSTM_rocket_reward.pt \
--task rocket \
--p_reversal_dist 0.025 0.025 \
--p_reward_reversal_dist 0.0 0.0 \
--out_data_file results/test_rocket_reward_on_rocket_transition.npy

echo "Testing metaRL system trained on rocket task with reward reversals only"
echo "on rocket task with both reward and transition reversals"

python test.py \
--episodes 300 \
--trials 100 \
--load_weights_from ../data/model_weights/LSTM_rocket_reward.pt \
--task rocket \
--p_reversal_dist 0.025 0.025 \
--p_reward_reversal_dist 0.5 0.5 \
--out_data_file results/test_rocket_reward_on_rocket_both.npy
