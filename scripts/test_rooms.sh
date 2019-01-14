#!/usr/bin/env bash

# To run: sbatch -p local -A ecortex --mem=8G --time=48:00:00 --gres=gpu:1 scripts/test_rooms.sh

export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate pytorch0.4.1

echo "Testing metaRL system on rooms_grid task"

python test.py \
--episodes 300 \
--trials 200 \
--load_weights_from ../data/model_weights/LSTM_rooms.pt \
--task rooms_grid \
--out_data_file results/test_rooms.npy
