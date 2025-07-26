#!/bin/bash
# set name of job
#SBATCH --job-name=gat_train
# set the number of nodes and processes per node
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n06
conda activate bio_info_env

#SBATCH --time=02:00:00
#SBATCH --output=gat_hp_%j.log
#SBATCH --error=gat_%j.err


python3 gcn_train.py
python3 gSage_train.py
python3 gin_train.py
python3 gat_train.py