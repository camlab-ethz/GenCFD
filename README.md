# GenCFD: Generative AI for CFD

This is the PyTorch based source code for the paper 
[Generative AI for fast and accurate Statistical Computation of Fluids](https://arxiv.org/abs/2409.18359).

### Overview

GenCFD is a PyTorch-based implementation designed for training and evaluating conditional score-based diffusion models for Computational Fluid Dynamics (CFD) tasks. These generative AI models enable fast, accurate, and robust statistical computation for simulating both two-dimensional and three-dimensional turbulent fluid flows.

![GenCFD Representation](https://drive.google.com/uc?export=view&id=12eiY6YnmZSBi12ZXw7HqqXfb6c_p1xBF)

## 🛠️ Installation

To set up a virtual environment and install the necessary dependencies for this project, follow these steps.

1) Create a Virtual Environment
```shell
python3 -m venv venv
```

2) Activate the Virtual Environment
```shell
source venv/bin/activate
```

3) Install dependencies: Make sure your virtual environment is active, then run:
```shell
pip install -r requirements.txt
```

## 🏋️ Training

Train a model using:
```shell
python3 -m train.train_gencfd \
--dataset <DATASET_NAME> \
--model_type <MODEL_NAME> \
--save_dir <DIRECTORY_PATH> \
--num_train_steps <INT>
```

### Notes:
* `DATASET_NAME` should be a valid dataset from the section Dataset.
* `MODEL_NAME` should either be `PreconditionedDenoiser` for the two-dimensional case, or `PreconditionedDenoiser3D` in the three-dimensional case.
    * 3D models have approximately 70M parameters with (64, 64, 64) resolution.
    * Recommended: GPU with 32GB memory (batch size 5) or 24GB memory (batch size 4).

After training, a JSON file with model settings is saved in the output directory for use during inference.

For a fast training, use a compiled model in parallel:

```shell
torchrun --nproc_per_node=<INT> \
-m train.train_gencfd \
--world_size <INT> \
--dataset <DATASET_NAME> \
--model_type <MODEL_NAME> \
--save_dir <DIRECTORY_PATH> \
--num_train_steps <INT>
--compile
```

The flag `--world_size` determines the size of the group associated with a communicator and it has to correspond to the number of processes 
or trainers used for parallelization. The relevant flag to set this is `--nproc_per_node`. Another advice for fast training is to chooce a 
proper number of workers which can be specified through the flag `--worker`.

## 🧪 Inference and Evaluation

Run inference with:

```shell
python3 -m eval.evaluate_gencfd \
--dataset <DATASET_NAME> \
--model_type <MODEL_NAME> \
--model_dir <DIRECTORY_PATH> \
--compute_metrics \
--monte_carlo_samples <INT> \
--visualize \
--save_gen_samples\
--save_dir <DIRECTORY_PATH>
```

Run inference in parallel:

```shell
torchrun --nproc_per_node=<INT> \
-m eval.evaluate_gencfd \
--world_size <INT> \
--dataset <DATASET_NAME> \
--model_type <MODEL_NAME> \
--model_dir <DIRECTORY_PATH> \
--compute_metrics \
--monte_carlo_samples <INT> \
--visualize \
save_gen_samples \
--save_dir <DIRECTORY_PATH>
```

Also here the number of models spawned should be the same for both flags `--world_size` and `--nproc_per_node`.
It's also possible to compile the model when run in a parallel or a sequential setup.

### Options:

* `--compute_metrics`: Computes evaluation metrics (e.g., mean and standard deviation) using Monte Carlo simulations.
* `--visualize`: Generates a single inference sample for visualization.
* `--save_gen_samples`: Saves randomly selected samples drawn from a uniform distribution

The number of sampling steps (`--sampling_steps`) for the Euler-Maruyama method should be preferably >30 for convergence.

## ⚙️ Summary of Additional Arguments

The following table summarizes key arguments that can help optimize memory usage or fine-tune model performance.

* **Action Arguments**: Simply add the flag (e.g., `--track_memory`), no need to specify `True` or `False`.
* **Boolean Flags**: Requires explicit specification of either `True` or `False` (e.g. `--use_mixed_precision True`)

A compiled version of the model can be used through adding the flag `--compile`. The compiler works without any issues on 
the following GPUs

* NVIDIA GeForce RTX 3090 
* NVIDIA GeForce RTX 4090 
* NVIDIA Tesla V100-SXM2 32 GiB 
* NVIDIA Tesla V100-SXM2 32 GB
* Nvidia Tesla A100

If there are some compiler warnings, you can always surpress them.

| Argument                 | Type   | Default                  | Scope     | Description                                                                                                                                                    |
|--------------------------|--------|--------------------------|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--nproc_per_node`              | int | 1 | Both         | Training or evaluation done with Distributed Data Parallel (DDP) across multiple machines.           |
| `--world_size`              | int | 1  | Both         | To enable training for DDP and a parallelized evaluation use the same integer value as for `--nproc_per_node` |
| `--compile`              | action | False  | Both         | Model can be compiled for faster training            |
| `--dataset`              | string | `<DATASET_NAME>`  | Both          | Dataset to use for training or evaluation. A list of available datasets is in the Dataset section.                                              |
| `--save_dir`             | string | `<DIRECTORY_PATH>`                | Both      | Directory to save models and metrics. If it doesn’t exist, it will be created automatically. Path is relative to the root directory.                                               |
| `--model_type`           | string | `PreconditionedDenoiser` | Both      | Model type to use. For 2D, options include `PreconditionedDenoiser`. For 3D, `PreconditionedDenoiser3D` is recommended.                            |
| `--batch_size`           | int | 5 | Both      | The number of samples per batch for the dataloader.                          |
| `--num_train_steps`      | int    | 10_000                   | Train     | Number of training steps. Increase for more training epochs or higher accuracy.   |
| `--track_memory`      | action    | False                   | Train     | If `True`, monitors memory usage for each training step.                                                                                 |
| `--use_mixed_precision`  | bool   | `True`                   | Train     | Enables mixed precision computation for faster training, using less memory. Set to `False` for full precision (default `torch.float32`).                                   |
| `--metric_aggregation_steps`      | int    | 500                   | Train     | Computes metrics (e.g., loss and its standard deviation) every specified number of training steps.                                                                                |
| `--save_every_n_steps`      | int    | 5000                   | Train     | Saves a checkpoint of the model and optimizer after every `n` steps.                                                                                |
| `--checkpoints`      | bool    | True                   | Train     | If `False`, disables checkpoint storage during training.               |
| `--num_blocks`  | int    | 4                      | Train      | Number of convolution blocks used in the model for each layer.                                                    |
| `--compute_metrics`      | action   | `False`                  | Eval      | If set to `True`, computes evaluation metrics over multiple samples for statistical accuracy.                                                                  |
| `--visualize`            | action   | `False`                  | Eval      | If set to `True`, generates a single visualized inference sample from the dataset for quick inspection of model output. The sample is drawn from a uniform distribution.                                      |
| `--sampling_steps`       | int    | 100                       | Eval      | Number of steps for the Euler-Maruyama method to solve the SDE during inference. Higher values generally improve convergence.                         |
| `--monte_carlo_samples`  | int    | 100                      | Eval      | Number of Monte Carlo samples to run for metric computation. Increase for more precise statistical results.                                                    |
|`--save_gen_samples` | action | `False` |  Eval | If set to `True`, stores the generated and ground truth results for randomly selected samples drawn from a uniform distribution. |

## 📊 Datasets

The table below provides a description of each dataset along with the corresponding flag argument for selection during training or evaluation.

| Dataset                    | Type      | Description                                                                | Use Case                        |
|----------------------------|-----------|----------------------------------------------------------------------------|---------------------------------|
| `DataIC_Vel`                | Train  | 2D incompressible flow dataset                                | 2D models              |
| `DataIC_3D_Time`            | Train  | Cylindrical Shear Flow dataset                                         | 3D models              |
| `DataIC_3D_Time_TG`     | Train  | Taylor-Green Dataset | 3D models         |
| `ConditionalDataIC_Vel`     | Eval| Perturbed 2D incompressible flow dataset                           | 2D Model         |
| `ConditionalDataIC_3D`      | Eval| Perturbed Cylindrical Shear Flow dataset                                  | 3D Model     |
| `ConditionalDataIC_3D_TG` | Eval| Perturbed Taylor-Green dataset           | 3D Model         |
