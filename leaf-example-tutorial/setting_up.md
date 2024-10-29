Set up your conda environment with the following command:

Make sure you change pytorch_env to the name you want to give to your environment. Only one of the two commands should be run, depending on whether you have an nvidia gpu or not.

```bash
# On hpcc:
module purge
module load Miniforge3

env=pytorch_env
# If you have an nvidia gpu and drivers for it installed:
conda create -n $env numpy matplotlib ipykernel numba scikit-learn scipy tqdm pandas seaborn pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia

# For people without nvidia gpu:
conda create -n $env numpy matplotlib ipykernel numba scikit-learn scipy tqdm pandas seaborn pytorch torchvision torchaudio cpuonly -c pytorch

# After the environment is created, you can activate it with the following command:
conda activate $env

# To install the ect package:
pip install ect

# To install the jupyter kernel for the environment:
python -m ipykernel install --user --name=$env

# If this does not work, try without --user flag:
python -m ipykernel install --name=$env

# HPCC users can use the following command to install the jupyter kernel:
module purge
module load Miniforge3
conda create -n Jupyter jupyterlab

# Using ondemand portal of hpcc: ondemand.hpcc.msu.edu
# Go to the page where you can start jupyter lab. (Interactive Apps -> Jupyter)
# Check Jupyter Lab
# Under Jupyter location, select "Conda Environment using Miniforge3 module"
# Type "Jupyter" in the box for "Conda Environment Name or Path"

```
