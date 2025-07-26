#!/bin/bash
# set the number of nodes and processes per node

#SBATCH --job-name=bio_gnn
# set the number of nodes and processes per node
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --gres=gpu:2


#SBATCH --partition=all
# set max wallclock time
#SBATCH --time=20:00:00
#SBATCH --output=gnn_%j.log
#SBATCH --error=gnn_%j.err


module load python/miniconda3.10 
eval "$(conda shell.bash hook)"
conda activate bio_inf
#conda install -c conda-forge rdkit

PYTHON_EXECUTABLE=$(which python)

${PYTHON_EXECUTABLE} training/gcn_train.py
${PYTHON_EXECUTABLE} training/gSage_train.py
${PYTHON_EXECUTABLE} training/gin_train.py
${PYTHON_EXECUTABLE} training/gat_train.py

echo "All files done."