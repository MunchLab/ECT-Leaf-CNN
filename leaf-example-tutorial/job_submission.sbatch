#!/bin/bash
#SBATCH --gpus=1
#SBATCH -c 20
#SBATCH --mem=256G
#SBATCH --time=1:00:00

module purge
module load Miniforge3

# Replace with your juypter conda enviroment. To get the enviroment name, run `conda env list` once you have loaded the Miniforge3 module.
conda activate Jupyter  

# Replace with your juypter kernel name. To get the kernel name, run `jupyter kernelspec list` once you have activated the Jupyter environment.
kernel_name=ect_cnn.gpu
jupyter nbconvert test_ECT_Direction_and_Threshold_Learning.ipynb --to html --embed-images --ExecutePreprocessor.kernel_name=$kernel_name --execute
