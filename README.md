# Sanity Checking Causal Representation Learning on a Simple Real-World System
Official code repository for the paper **Sanity Checking Causal
Representation Learning on a Simple Real-World System** by
Juan L. Gamella*, Simon Bing* and Jakob Runge.

### TODOs:
- Download instructions (submodule init and update stuff, create envornment file)
- run instructions for all models
- General instruction on how to include new datasets

## Setup

Since we include the source code of several different methods as submodules,
these need to be properly initialized and updated. After cloning the main
repository, first run
```
git submodule update --init --recursive
```
and then
```
git submodule update --recursive --remote
```

We recommend installing the required packages in a conda environment. To
do so:
1. Create a new conda environment from the provided config file:
```
conda env create -f requirements.yml
```
TODO: check which torch version is installed, add instructions to install cpu only version.

2. Activate the conda environment:
```
conda activate crc
```



## Experiments
- mention that we implement all of this with causal chambers package and data can be found there
- data is automaticaly downloaded when running an experiment
- add subsections with examples for all three methods

### Contrastive CRL

### Multiview CRL

### CITRIS

### Adding methods

## Citation
If you find our paper interesting or use the code in this repository,
please consider citing:

```bibtex
@article{gamellabing2025sanity,
  title     = {Sanity Checking Causal Representation Learning on a Simple Real-World System},
  author    = {Gamella*, Juan L. and Bing*, Simon and Runge, Jakob},
  year      = {2025},
  journal   = {arXiv preprint arXiv:TODO},
  eprint    = {TODO},
  note      = {*equal contribution}
}
```