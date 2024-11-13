# pprl

Reinforcement learning on point clouds with representation learning.

## Getting Started (installation from source)

pprl can be installed with Python 3.10. We recommend using conda/mamba to manage your Python environment.

```
conda create -n pprl -f env.yaml
conda activate pprl
```

Next, install parllel and pprl itself
```
pip install -e dependencies/parllel
pip install -e .
```

### Troubleshooting

If you get an error message like

```
ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
```

try fixing it by modifying the LD_LIBRARY_PATH like this:

```
conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

## Environments

Each environment suite can be installed independently.

### sofa_env

- Make sure to install sofa after installing all other dependencies, so that it is compiled with all the correct dependencies (e.g. numpy version)
- Install sofa according to [instructions in sofa_env](https://github.com/balazsgyenes/sofa_env/blob/main/docs/source/setting_up_sofa.rst). Tip: if you have multiple sofa builds on your system, build into a folder within this repo for better organization.
- Install sofa_env python package with:
```
pip install -e dependencies/sofa_env
```

### Maniskill2

Install with:
```
conda install vulkan-tools  # required for maniskill2 to work on cluster
pip install mani_skill2
```

Make sure you download the required assets for each environment with:
```
python -m mani_skill2.utils.download_asset PushChair-v1
python -m mani_skill2.utils.download_asset OpenCabinetDrawer-v1
python -m mani_skill2.utils.download_asset OpenCabinetDoor-v1
python -m mani_skill2.utils.download_asset TurnFaucet-v0
```

## Training

We use [hydra](https://hydra.cc/docs/intro/) for configs.
Launch a single training run of PPRL (PointPatchRL without reconstruction loss) on your local machine with:
```
python scripts/train_sac.py env=push_chair model=ppt
```

Launch a single training run of PPRL + Aux (PointPatchRL with reconstruction loss) on your local machine with:
```
python scripts/train_sac.py env=push_chair model=pointgpt_rl algo=aux_sac
```


If you just want to test to see if everything works (fewer parallel environments, smaller batch size, fewer training steps), run:
```
WANDB_MODE=disabled python scripts/train_sac.py env=push_chair model=ppt platform=debug
```

Take a look at other options available through the config with:
```
python scripts/train_sac.py env=push_chair model=ppt -h
```
