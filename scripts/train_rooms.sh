#!/usr/bin/env bash

# To run: sbatch -p local -A ecortex --mem=8G --time=48:00:00 --gres=gpu:1 scripts/train_rooms.sh

export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate pytorch0.4.1

echo "Training metaRL system on rooms_grid task"

python train.py \
--episodes 1000 \
--trials 20 \
--task rooms_grid \
--beta_v 0.05 \
--beta_e 0.05 \
--learning_rate 0.0007 \
--t_max 300 \
--out_data_file results/train_rooms.json \
--checkpoint_path ../data/model_weights/LSTM_rooms.pt \
--checkpoint_every 200
