# Sanity Checking Causal Representation Learning on a Simple Real-World System
[![Paper (CRL Sanity Check)](https://img.shields.io/static/v1.svg?logo=arxiv&label=Paper&message=CRLSanityCheck&color=green)](TODO)


Official code repository for the [paper](https://arxiv.org/abs/2502.20099) **Sanity Checking Causal
Representation Learning on a Simple Real-World System** (2025) by
Juan L. Gamella*, Simon Bing* and Jakob Runge.

<p align="center">
<img src="figures/tunnel.png" width="100%" align="center"/>
</p>

If you use our code or datasets in your work, please consider citing:

```bibtex
@article{gamellabing2025sanity,
  title     = {Sanity Checking Causal Representation Learning on a Simple Real-World System},
  author    = {Gamella*, Juan L. and Bing*, Simon and Runge, Jakob},
  year      = {2025},
  journal   = {arXiv preprint arXiv:2502.20099},
  note      = {*equal contribution}
}
```

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
do so, first create a new conda environment from the provided config file:
```
conda env create -f environment.yml
```
and activate it
```
conda activate crc
```

## Experiments

#### Accessing the datasets

The data used for the benchmark is part of the `lt_crl_benchmark_v1` dataset, which is hosted in the [dataset repository](https://github.com/juangamella/causal-chamber) of the Causal Chamber project. You can download it directly from its [dataset page](https://github.com/juangamella/causal-chamber/tree/main/datasets/lt_crl_benchmark_v1).

For our experiments, we use the `causalchamber` [package](https://github.com/juangamella/causal-chamber-package) to directly download and access the datasets from the Python code. Each dataset is automatically downloaded when running the respective experiment.

### Contrastive CRL
For the real-data experiment using the [Contrastive CRL](https://arxiv.org/abs/2306.02235) method, run

```bash
python contrastive_crl_experiment --dataset lt_crl_benchmark_v1 \
                                  --task contrast_crl_real \
                                  --batch_size 512 \
                                  --epochs 100 \
                                  --lat_dim 5 \
                                  --data_root /path/to/data \
                                  --root_dir /path/to/save/output
```
To run the experiment on the synthetic ablation data, set `--dataset contrast_semi_synth_decoder`
and optionally set the `--task` flag to a name that reflects this change.

### Multiview CRL
The [Multiview CRL](https://arxiv.org/abs/2311.04056) experiment on real data uses the following command
```bash
python multiview_experiment.py --dataset lt_crl_benchmark_v1 \
                               --task contrast_crl_real \
                               --exp_name buchholz_1 \
                               --train_steps 28500 \
                               --batch_size 512 \
                               --lat_dim 5 \
                               --data_root /path/to/data \
                               --root_dir /path/to/save/output
```
For the experiment with synthetic data, set `--dataset chambers_semi_synth_decoder`
and `--exp_name buchholz_1_synth_det`. Optionally, you can also rename the `--task`
flag to reflect this change.

### CITRIS
The [CITRIS](https://arxiv.org/abs/2202.03169) experiment using real data is run with the command
```bash
python citris_experiment.py --dataset chambers \
                            --task contrast_crl_real \
                            --epochs 250 \
                            --batch_size 512 \
                            --lat_dim 16 \
                            --data_root /path/to/data \
                            --root_dir /path/to/save/output
```
To use the synthetic data, set `--dataset chambers_semi_synth_decoder` and optionally change
the `--task` flag to a name that reflects this change.

## Help and feedback

In our paper, we only evaluate a handful of CRL methods representative of different families. Furthermore, the datasets in our benchmark constitute only a small fraction of all experiments that are possible with the [Causal Chambers](https://causalchamber.ai).

Therefore: please reach out if you need help in evaluating a new method, you have an idea for further experiments with the chambers, or encounter a bug. You can open a [GitHub Issue](https://github.com/simonbing/CRLSanityCheck/issues) or send us an [email](mailto:juangamella@gmail.com,bing@campus.tu-berlin.de). We are happy to help :)

## License

The code in this repository is shared under the permissive [MIT license](https://opensource.org/license/mit/). A copy of can be found in [LICENSE](LICENSE).

