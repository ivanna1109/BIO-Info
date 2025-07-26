#!/bin/bash
# set the number of nodes and processes per node

#SBATCH --job-name=gnn_optuna
# set the number of nodes and processes per node
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --gres=gpu:2


#SBATCH --partition=all
# set max wallclock time
#SBATCH --time=32:00:00
#SBATCH --output=optuna_%j.log
#SBATCH --error=optuna_%j.err


module load python/miniconda3.10 
eval "$(conda shell.bash hook)"
conda activate bio_inf
#conda install -c conda-forge rdkit

PYTHON_EXECUTABLE=$(which python)

${PYTHON_EXECUTABLE} training/hyperparam_gat.py
${PYTHON_EXECUTABLE} training/hyperparam_gin.py


echo "All files done."