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

### Train a 2D model
To run training for a 2D dataset, this can be achieved as follows for the dataset DataIC_Vel. In this case the 
default model PreconditionedDenoiser is used for training.
```shell
python3 -m train.train_gencfd --dataset DataIC_Vel --save_dir outputs --num_train_steps 10000
```
### Train a 3D model
Furthermore, to run a 3D model and dataset it is best to set the number of heads to a maximum of 4. With 8 even on the 
lowest layer only applied the required GPU memory succeeds 32 Gigabytes. The model can be trained as follows:
```shell
python3 -m train.train_gencfd --dataset DataIC_3D_Time --model_type PreconditionedDenoiser3D --save_dir outputs --batch_size 2 --num_heads 4
```
## Inference and Evaluation
The inference loop can be run with the following command, where a model directory needs to be set. The default command 
looks as follows. To run inference there are two options. First option is to set the flag --compute_metrics to True. In this 
case a default of 100 Monte Carlo Simulations is run to generate metrics such as the relative mean and standard deviation for each channel.
Second option is to set the flag --visualize to True. This allows a single inference run and visualizes the first generated sample of the
provided dataset.

Instead of the dataset used for training we want to compute metrics with a perturbed dataset called ConditionalDataIC_i, where i should relate to the dataset used for training.
```shell
python3 -m eval.evaluate_gencfd --dataset ConditionalDataIC_Vel --model_dir outputs/checkpoints
```
Furthermore another flag that can be used is --sampling_steps, this sets the number of time steps for the 
Euler Maruyama method to solve the SDE. The number of sampling steps should be preferably above 30 to reach convergence.
```shell
python3 -m eval.evaluate_gencfd --dataset ConditionalDataIC_Vel --model_dir outputs/checkpoints --sampling_steps 80
```
At last to get the visualization results of a single sample this can be done for the 3D as well as the 2D model by using the 
flag --visualize. The following command provides an example for a 2D model using again the same dataset and model. Moreover, to store the metric results or visualization results use the flag --save_dir to provide a relative path to the directory where you want to save the results.
```shell
python3 -m eval.evaluate_gencfd --dataset ConditionalDataIC_Vel --model_dir outputs/checkpoints --visualize True --save_dir outputs
```

### Compute evaluation metrics for the 2D model
For a trained 2D model to run the results, the directory where the relevant checkpoint folder with the models should be provided. In 
the following case the models can be found in outputs/checkpoints/. It is enough to only provide the relative path from the root directory.
```shell
python3 -m eval.evaluate_gencfd --dataset ConditionalDataIC_Vel --model_dir outputs --compute_metrics True --monte_carlo_samples 1000 --save_dir outputs
```

### Compute evaluation metrics for the 3D model
As a result, for the 3D model trained with the arguments presented above. Inference can be achieved as follows:
```shell
python3 -m eval.evaluate_gencfd --dataset ConditionalDataIC_3D --model_type PreconditionedDenoiser3D --model_dir outputs --compute_metrics True --monte_carlo_samples 1000 --save_dir outputs
```