from __future__ import annotations

from typing import Any, Callable

from pyclad.models.cfa.cfa_cl import ContinualCFA
from pyclad.models.model import Model
from pyclad.models.padim.padim_cl import ContinualPaDiM
from pyclad.models.patchcore.patchcore_cl import ContinualPatchCore
from pyclad.strategies.replay.buffers.adaptive_balanced import AdaptiveBalancedReplayBuffer
from pyclad.strategies.replay.selection.random import RandomSelection
from pyclad.strategies.vision.cfa_cl import CFACLStrategy
from pyclad.strategies.vision.padim_cl import PaDiMCLStrategy
from pyclad.strategies.vision.patchcore_cl import PatchCoreCLStrategy


CL_MODEL_CLASSES = {
    "padim": ContinualPaDiM,
    "patchcore": ContinualPatchCore,
    "cfa": ContinualCFA,
}


def validate_strategy_choice(model_name: str, strategy_name: str) -> None:
    if strategy_name == "cl" and model_name not in {"padim", "patchcore", "cfa"}:
        raise ValueError(
            f"Strategy 'cl' is currently implemented only for visual memory-bank models: padim, patchcore, cfa. "
            f"Got model '{model_name}'."
        )


def build_model(base_runner: Any, args, seed: int) -> tuple[Model, Any]:
    config = base_runner.build_model_config(args, seed)
    if args.model in CL_MODEL_CLASSES:
        model_cls = CL_MODEL_CLASSES[args.model]
        return model_cls(config), config
    return base_runner.build_model(args, seed)


def resolve_continual_memory_bank_size(base_runner: Any, args, dataset) -> int:
    if getattr(args, "continual_memory_bank_size", None) is not None:
        return int(args.continual_memory_bank_size)
    return int(base_runner.resolve_replay_buffer_size(args, dataset))


def build_strategy(
    base_runner: Any,
    args,
    model: Model,
    model_creation_fn: Callable[[], Model],
    replay_buffer_size: int,
    run_dataset,
):
    validate_strategy_choice(args.model, args.strategy)

    if args.model == "padim" and args.strategy == "cl":
        if not isinstance(model, ContinualPaDiM):
            raise TypeError(f"PaDiM CL expects ContinualPaDiM, got {type(model).__name__}")
        return PaDiMCLStrategy(model)

    if args.model == "patchcore" and args.strategy == "cl":
        if not isinstance(model, ContinualPatchCore):
            raise TypeError(f"PatchCore CL expects ContinualPatchCore, got {type(model).__name__}")
        previous_ratio = float(getattr(args, "patchcore_previous_ratio", 0.25))
        return PatchCoreCLStrategy(model=model, previous_ratio=previous_ratio)

    if args.model == "cfa" and args.strategy in {"cl", "replay"}:
        if not isinstance(model, ContinualCFA):
            raise TypeError(f"CFA CL expects ContinualCFA, got {type(model).__name__}")
        buffer_size = replay_buffer_size
        if buffer_size <= 0:
            buffer_size = int(base_runner.resolve_replay_buffer_size(args, run_dataset))
        replay_buffer = AdaptiveBalancedReplayBuffer(
            selection_method=RandomSelection(),
            max_size=buffer_size,
        )
        return CFACLStrategy(model=model, buffer=replay_buffer)

    return base_runner.build_strategy(
        args.strategy,
        model,
        model_creation_fn,
        replay_buffer_size,
    )
