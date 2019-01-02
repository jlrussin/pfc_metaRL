#!/usr/bin/env bash

# To run: sbatch -p local -A ecortex --mem=8G --time=48:00:00 --gres=gpu:1 scripts/train_two_step.sh

export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate pytorch0.4.1

echo "Training metaRL system on two-step_task"

python train.py \
--episodes 10000 \
--trials 100 \
--p_common_dist 0.8 0.8 \
--r_common_dist 0.9 0.9 \
--p_reversal_dist 0.025 0.025 \
--beta_v 0.05 \
--beta_e 0.05 \
--learning_rate 0.0007 \
--out_data_file results/train.json \
--checkpoint_path ../data/model_weights/LSTM_two_step.pt
