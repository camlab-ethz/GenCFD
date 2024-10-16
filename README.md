# GenCFD

This Repository is based on the swirl-dynamics code by Google from 
[arXiv:2305.15618](https://arxiv.org/abs/2305.15618) and rewritten from 
Jax into a PyTorch version.

You can execute training and evaluation / inference tasks directly from the root 
directory using the following commands.

## Training
The training which is right now only valid for the DataIC_Vel dataset can 
be run with the following command:
```shell
python3 -m train.train_gencfd
```

## Inference
The inference loop which is right now only valid for the DataIC_Vel dataset can 
be run with the following command:
```shell
python3 -m eval.evaluate_gencfd
```