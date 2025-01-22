# Copyright 2024 The swirl_dynamics Authors.
# Modifications made by the CAM Lab at ETH Zurich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train states for gradient descent mini-batch training.

Train state classes are data containers that hold the model variables, optimizer
states, plus everything else that collectively represent a complete snapshot of
the training. In other words, by saving/loading a train state, one
saves/restores the training progress.
"""

from typing import Any, Optional, Dict
import torch
import torch.nn as nn
from torch import optim
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

Tensor = torch.Tensor


class TrainState:
    """Base train state class.

    Attributes:
      step: A counter that holds the number of gradient steps applied.
    """

    def __init__(self, step: int = 0):
        if isinstance(step, Tensor):
            self.step = step.clone().detach()
        else:
            self.step = torch.tensor(step)

    @property
    def int_step(self) -> int:
        """Returns the step as an int.

        This method works on both regular and replicated objects. It detects whether
        the current object is replicated by looking at the dimensions, and
        unreplicates the `step` field if necessary before returning it.
        """
        if isinstance(self.step, int):
            return self.step
        else:
            return int(self.step.item())

    @classmethod
    def restore_from_checkpoint(
        cls, ckpt_path: str, ref_state: Optional["TrainState"] = None
    ) -> "TrainState":
        """Restores train state from an orbax checkpoint directory.

        Args:
          ckpt_dir: A directory which may contain checkpoints at different steps. A
            checkpoint manager will be instantiated in this folder to load a
            checkpoint at the desired step.
          ref_state: A reference state instance. If provided, the restored state
            will be the same type with its leaves replaced by values in the
            checkpoint. Otherwise, the restored object will be raw dictionaries,
            which should be fine for inference but will become problematic to resume
            training from.

        Returns:
          Restored train state.
        """
        checkpoint = torch.load(ckpt_path, weights_only=True)

        if ref_state is not None:
            "update the reference state with restored values"
            ref_state.step = checkpoint.get("step", ref_state.step)
            ref_state.update_from_checkpoint(checkpoint)
            return ref_state
        else:
            "create a new state from the checkpoint"
            return cls(**checkpoint)

    def save_checkpoint(self, ckpt_path: str) -> None:
        "Saves the current state to a checkpoint"
        checkpoint = self.state_dict()
        torch.save(checkpoint, ckpt_path)

    def state_dict(self) -> Dict[str, Any]:
        "Returns the state dictionary for saving."
        return {"step": self.step.item()}

    def update_from_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        "Update state attributes from checkpoint"
        self.step = torch.tensor(checkpoint.get("step", self.step))


class BasicTrainState(TrainState):
    """Train state that stores optimizer state, flax model params and mutables.

    Attributes:
      params: The parameters of the model.
      opt_state: The optimizer state of the parameters.
      flax_mutables: The flax mutable fields (e.g. batch stats for batch norm
        layers) of the model being trained.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        params=None,
        opt_state=None,
        step: int = 0,
    ):
        super().__init__(step)
        self.model = model
        self.optimizer = optimizer
        # self.params = params if params is not None else self.model.state_dict()
        # self.opt_state = opt_state if opt_state is not None else self.optimizer.state_dict()
        self.params = params
        self.opt_state = opt_state

    @classmethod
    def restore_from_checkpoint(
        cls,
        ckpt_path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        is_compiled: bool,
        is_parallelized: bool,
        use_ema: bool = False,
        device: torch.device = None,
    ) -> TrainState:

        # if model was trained on gpu but is evaluated on the cpu 'map_location' is necessary
        checkpoint = torch.load(ckpt_path, weights_only=True, map_location=device)

        opt_state = (
            checkpoint["opt_state"]
            if "opt_state" in checkpoint.keys()
            else checkpoint["optimizer_state_dict"]
        )

        if use_ema:
            params = (
                checkpoint["ema_param"]
                if "ema_param" in checkpoint.keys()
                else checkpoint["model_state_dict"]
            )
        else:
            params = (
                checkpoint["params"]
                if "params" in checkpoint.keys()
                else checkpoint["model_state_dict"]
            )

        checkpoint_compiled = checkpoint[
            "is_compiled"
        ]  # check if stored model was compiled
        checkpoint_ddp = checkpoint[
            "is_parallelized"
        ]  # check if stored model was trained in parallel

        keyword_compiled = "_orig_mod."
        keyword_ddp = "module."

        if not is_compiled and checkpoint_compiled:
            # stored model was compiled, current model is not: delete _orig_mod. in every key
            params = {
                key.replace(keyword_compiled, ""): value
                for key, value in params.items()
            }

        if is_compiled and not checkpoint_compiled:
            # stored model not compiled, current model is compiled: add _orig_mod. at the beginning
            params = {keyword_compiled + key: value for key, value in params.items()}

        if not is_parallelized and checkpoint_ddp:
            # stored model trained in parallel but current model is not
            params = {
                key.replace(keyword_ddp, ""): value for key, value in params.items()
            }

        if is_parallelized and not checkpoint_ddp:
            # stored model was not trained in parallel but current model is
            params = {keyword_ddp + key: value for key, value in params.items()}

        model.load_state_dict(params)
        optimizer.load_state_dict(opt_state)

        return cls(
            model=model,
            optimizer=optimizer,
            params=params,
            opt_state=opt_state,
            step=checkpoint.get("step", 0),
        )

    def state_dict(self) -> Dict[str, Any]:
        "Extend base state_dict to include model and optimizer states"
        state = super().state_dict()
        state.update(
            {
                "params": self.model.state_dict(),
                "opt_state": self.optimizer.state_dict(),
            }
        )
        return state

    def update_from_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        "Update model and optimizer states from checkpoint."
        super().update_from_checkpoint(checkpoint)

        self.model.load_state_dict(checkpoint["params"])
        self.optimizer.load_state_dict(checkpoint["opt_state"])

    def replace(
        self, step: int, params: Dict[str, Any] = None, opt_state: Dict[str, Any] = None
    ):
        """Replaces state values with updated fields."""
        self.step = step
        self.params = params
        self.opt_state = opt_state


class DenoisingModelTrainState(BasicTrainState):
    """Train state with an additional field tracking the EMA parameters."""

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        params=None,
        opt_state=None,
        step: int = 0,
        ema_decay: float = 0.999,
        store_ema: bool = False,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            params=params,
            opt_state=opt_state,
            step=step,
        )

        self.ema_decay = ema_decay
        self.store_ema = store_ema
        self.ema_model = (
            AveragedModel(self.model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))
            if store_ema
            else None
        )
        self.ema = self.ema_parameters if store_ema else None

    @property
    def ema_parameters(self):
        """Return the EMA model's prarameters."""
        if self.ema_model:
            return self.ema_model.module.state_dict()
        else:
            raise ValueError("EMA model is None")

    def replace(
        self,
        step: int,
        params: Dict[str, Any] = None,
        opt_state: Dict[str, Any] = None,
        ema: Dict[str, Any] = None,
    ):
        """Replaces state values with updated fields."""

        self.step = step
        self.params = params
        self.opt_state = opt_state
        self.ema = ema
