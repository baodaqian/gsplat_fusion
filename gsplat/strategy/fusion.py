from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch
from typing_extensions import Literal

from .base import Strategy
from .ops import duplicate, remove, reset_opa, split
from .default import DefaultStrategy

# gsplat/strategy/fusion.py

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch
from .default import DefaultStrategy

@dataclass
class FusionStrategy(DefaultStrategy):
    """
    Updated 04072025
    A densification strategy that limits the maximum number of Gaussians
    used in the optimization. This strategy is based on the DefaultStrategy
    but adds an upper bound check before growing new Gaussians.
    """
    max_gaussians: int = 10_000  # Set your desired upper limit here.

    def _grow_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> Tuple[int, int]:
        # Check current number of Gaussians.
        current_num = len(params["means"])
        if current_num >= self.max_gaussians:
            if self.verbose:
                print(
                    f"Step {step}: Current Gaussians = {current_num} reached max threshold ({self.max_gaussians}). "
                    "Skipping growth (duplication/splitting)."
                )
            # Return zeros so that no additional Gaussians are grown.
            return 0, 0

        # Otherwise, perform the usual growth from DefaultStrategy.
        return super()._grow_gs(params, optimizers, state, step)

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
    ):
        # If we already have reached max_gaussians, we may want to bypass growth
        # but still allow pruning and resetting opacities.
        if len(params["means"]) >= self.max_gaussians:
            if self.verbose:
                print(f"Step {step}: Maximum Gaussians reached. Skipping growth, proceeding to pruning.")
            n_prune = self._prune_gs(params, optimizers, state, step)
            if self.verbose:
                print(f"Step {step}: {n_prune} GSs pruned. Now having {len(params['means'])} GSs.")
            state["grad2d"].zero_()
            state["count"].zero_()
            if self.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()
            # Skip calling the parent's _grow_gs. Exit the function.
            return

        # Otherwise, proceed as in the default strategy.
        super().step_post_backward(params, optimizers, state, step, info, packed)
