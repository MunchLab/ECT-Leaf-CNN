Set up your conda environment with the following command:

Make sure you change pytorch_env to the name you want to give to your environment. Only one of the two commands should be run, depending on whether you have an nvidia gpu or not.

```bash

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
```
