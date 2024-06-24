
# Permutation-Contrastive Learning  (PCL)

This code is the official implementation of

Aapo Hyvärinen and Hiroshi Morioka, Nonlinear ICA of Temporally Dependent Stationary Sources. Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS2017)

If you are using pieces of the posted code, please cite the above paper.

## Requirements

Python3

Pytorch


## Training

To train the model(s) in the paper, run this command:

```train
python pcl_training.py
```

## Evaluation

To evaluate the trained model, run:

```eval
python pcl_evaluation.py
```
