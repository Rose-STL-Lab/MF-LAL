# Multi-Fidelity Latent Space Active Learning (MF-LAL)

This repository contains code for generating molecules using MF-LAL. `main.py` contains code to run the active learning loop, and `data.py` contains code to run each oracle and manage the datasets.

## Usage

Call `python main.py 'target'` to run active learning, where `target` is either `cmet` or `brd4-2`. The results of the run, including oracle outputs, will be saved to TensorBoard in the `logs/mf-lal` folder.

## Installation

First, install [RDKit](https://www.rdkit.org/docs/Install.html) and [OpenBabel](https://github.com/openbabel/openbabel) (make sure to get version < 3, due to protonation issues with BAT.py. Confirmed to work on version 2.4.1). Then, to install the necessary packages for the simulators, follow the subsections below. 

Additionally, note that the following python packages are required:

```
pytorch==2.0.1
gpytorch==1.10
botorch==0.8.5
selfies
rdkit
```

### AutoDock4

Download and compile [AutoDock-GPU](https://github.com/ccsb-scripps/AutoDock-GPU). The location of the AutoDock-GPU executable that the code assumes is `~/AutoDock-GPU/bin/autodock_gpu_128wi`.

### ABFE

We use BAT.py, from [https://github.com/GHeinzelmann/BAT.py/](https://github.com/GHeinzelmann/BAT.py/), to run ABFE calculations. All the necessary files are already included in this repostiory under the `BAT.py` directory. The full list of requirements to run BAT.py are listed [here](https://github.com/GHeinzelmann/BAT.py/?tab=readme-ov-file#getting-started), and they must all be installed to run the simulator. We use the AMBER based version of BAT.py, which requires a purchased license, but it should be possible to run with OpenMM as well with some modifications. The code assumes AMBER is installed to `/home/ubuntu/amber22`. The code also assumes that 8 CUDA GPUs are available to run AMBER simulations. To change which GPUs are used for the simulations, change the `ABFE_DEVICES` variable on line 23 of `utils.py`.