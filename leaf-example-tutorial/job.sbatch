#!/bin/bash
#SBATCH --gpus=1
#SBATCH --constraint="a100|v100"
#SBATCH -c 16
#SBATCH --mem-per-cpu=750M
#SBATCH --time=4-0:00:00
#SBATCH -J tops

# SLURM <-- 
module purge
module load Miniforge3

# Replace with your juypter conda enviroment. To get the enviroment name, run `conda env list` once you have loaded the Miniforge3 module.
conda activate Jupyter  

# Replace with your juypter kernel name. To get the kernel name, run `jupyter kernelspec list` once you have activated the Jupyter environment.
kernel_name=ect_cnn.gpu
jupyter nbconvert final.ipynb --to html --embed-images --ExecutePreprocessor.kernel_name=$kernel_name --execute
