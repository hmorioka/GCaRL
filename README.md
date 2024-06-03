
# Grouped Causal Representation Learning (G-CaRL)

This code is the official implementation of

Hiroshi Morioka and Aapo Hyvärinen, Causal representation learning made identifiable by grouping of observational variables. Proceedings of The 41st International Conference on Machine Learning  (ICML2024).

If you are using pieces of the posted code, please cite the above paper.

## Requirements

Python3

Pytorch


## Training

To train the model(s) in the paper, run this command:

```train
python gcarl_training.py
```

Set parameters depending on which simulation you want to run:

'Simulation1': parameters for Simulation1

'Simulation2': parameters for Simulation2

'GRN': parameters for gene regulatory network recovery

'3dIdent': parameters for high-dimensional image observations (https://zenodo.org/records/4502485#.YgWm1fXMKbg)


## Evaluation

To evaluate the trained model, run:

```eval
python gcarl_evaluation.py
```
