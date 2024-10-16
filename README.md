# GenCFD

This Repository is based on the swirl-dynamics code by Google from 
[arXiv:2305.15618](https://arxiv.org/abs/2305.15618) and rewritten from 
Jax into a PyTorch version.

## Training
The training which is right now only valid for the DataIC_Vel dataset can 
be run with the following command:
```shell
python -m train.train_gencfd
```

## Evaluate
The inference loop which is right now only valid for the DataIC_Vel dataset can 
be run with the following command:
```shell
python -m eval.evaluate_cfd
```