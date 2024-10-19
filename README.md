# GenCFD

This Repository is based on the swirl-dynamics code by Google from 
[arXiv:2305.15618](https://arxiv.org/abs/2305.15618) and rewritten from 
Jax into a PyTorch version.

You can execute training and evaluation / inference tasks directly from the root 
directory using the following commands.

## Training

The training with default settings can be run as follows, where a save directory for the model
and metrics needs to be passed
```shell
python3 -m train.train_gencfd --save_dir outputs
```

If you want to specify a specific dataset and not use the default DataIC_Vel this can be done as 
follows. Also to avoid saving checkpoints this can be also set to False and the number of training steps
can be added manually as well
```shell
python3 -m train.train_gencfd --dataset DataIC_Vel --save_dir outputs --checkpoints False --num_train_steps 10000
```

## Inference
The inference loop can be run with the following command, where a model directory needs to be set. The default command 
looks as follows.
```shell
python3 -m eval.evaluate_gencfd --dataset DataIC_Vel --model_dir outputs/checkpoints
```
Further the number of time steps for the Euler Maruyama method to solve the SDE can be set 
manually as follows. The number of sampling steps should be preferably above 30 to reach convergence.
```shell
python3 -m eval.evaluate_gencfd --dataset DataIC_Vel --model_dir outputs/checkpoints --sampling_steps 80
```