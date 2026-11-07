"""
Resumable wrapper around ``examples/clvad/run_continual_visual_ad.py``.

A 40-GPU-hour continual run that dies at concept 11 for a reason that has
nothing to do with the method — a wall-clock limit, a preempted node, a full
filesystem — currently has to start again at concept 0. This script is the same
runner with a checkpoint at every concept boundary, so a requeued job picks up
where the dead one stopped.

It is a *wrapper*, not a fork. The base runner is loaded as a module and every
piece of argument parsing, dataset loading, config building, metric wiring and
result serialisation is reused verbatim. Exactly five things are overridden:

1. the strategy and model classes (the checkpointable ``*V2`` variants from
   ``pyclad.resume.methods``);
2. the scenario class (``ResumableConcept{Aware,Incremental}Scenario``);
3. one extra callback (``StateCheckpointCallback``), registered last;
4. the resume path itself (``build_resume_plan`` + ``apply_resume``);
5. the JSON checkpoint callback, which must not rewrite the result file outside
   the commit window — see ``ResumeSafeJsonCheckpointCallback`` below.

Supported method/strategy pairs, matching the three published CAD baselines:

===================  =====================  ============================
``--model``          ``--strategy``         checkpointable classes
===================  =====================  ============================
``ucad``             ``ucad``               ``ResumableUCAD`` + ``ResumableUCADStrategy``
``patchcore``        ``replaycad``          ``CheckpointedPatchCore`` +
                                            ``ResumableReplayCADStrategy``
``one_for_more``     ``naive``              ``ResumableOneForMore`` +
                                            ``ResumableNaiveStrategy``
===================  =====================  ============================

Any other combination is refused rather than silently run without checkpoints.

Checkpointing is **on by default**; only *resuming* is opt-in. A job killed
without warning is therefore resumable even if nobody thought to ask for it.

Examples
--------
First submission (writes checkpoints, starts at concept 0)::

    python examples/clvad/run_continual_visual_ad_resumable.py \\
        --model ucad --strategy ucad \\
        --benchmark mvtec --inclad-bench \\
        --root $REPO/src/pyclad/data/datasets/visual_datasets/mvtec_ad \\
        --ordering-mode easy_to_hard --eval-level both \\
        --output-dir $HOME/ucad-results/mvtec/easy_to_hard/run \\
        --resume-dir $HOME/pyclad-checkpoints

Requeue of the same job (identical command plus ``--resume``)::

    ... --resume

The checkpoint directory is content-addressed: the requeued job finds its own
predecessor from ``--model``, ``--strategy``, ``--benchmark``, the seed, the
resolved concept order, ``--eval-level`` and the resolved resize, without being
told where the previous job's output lived.

Two operational notes
---------------------
``--output-dir`` must be stable across resubmits. The LSF job scripts derive it
as ``$OUTPUT_ROOT/job-$LSB_JOBID``, and a requeued job gets a new job id, so pin
``OUTPUT_DIR`` (or ``RUN_ID``) in the environment before submitting. If the
output directory does move, this runner still restores the metrics from the
path the manifest recorded and writes a complete result JSON at the new path —
it just says so, loudly.

``--replaycad-artifact-dir`` must point outside ``--output-dir`` for the same
reason, and ``--replaycad-cache-generated`` must be set: without the cache a
resumed ReplayCAD run regenerates every replay image from the diffusion
pipeline. Both are enforced here rather than discovered at resume time.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import pathlib
import sys
import time
import traceback
from typing import Any, Callable

from pyclad.callbacks.callback import Callback
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.output.output_writer import InfoProvider
from pyclad.resume import (
    DISK_FULL_EXIT_CODE,
    RESUME_PHASES,
    CheckpointDiskSpaceError,
    CheckpointStore,
    ResumableConceptAwareScenario,
    ResumableConceptIncrementalScenario,
    ResumeError,
    StateCheckpointCallback,
    apply_resume,
    build_resume_plan,
    compare_fingerprints,
    compute_run_id,
    ensure_resume_dir_outside_output,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- base module


def _load_base_runner():
    """Load the base runner by path; it is a script, not an importable module."""
    module_path = pathlib.Path(__file__).with_name("run_continual_visual_ad.py")
    spec = importlib.util.spec_from_file_location("pyclad_visual_runner_base", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load base visual runner from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_BASE = _load_base_runner()


#: ``(model, strategy)`` pairs this runner can checkpoint. The value is the key
#: used below to pick the model class, the strategy class and the scenario.
SUPPORTED_METHODS: dict[tuple[str, str], str] = {
    ("ucad", "ucad"): "ucad",
    ("patchcore", "replaycad"): "replaycad",
    ("one_for_more", "naive"): "one_for_more",
}

#: Fallback ``CheckpointStore(keep=...)`` when the method module names none.
DEFAULT_CHECKPOINT_KEEP = 2

#: Sentinel for ``--checkpoint-keep``: use the method's own recommendation.
_KEEP_FROM_METHOD = -1


def default_resume_root() -> pathlib.Path:
    """Where checkpoints live when ``--resume-dir`` is not given.

    Deliberately *not* derived from ``--output-dir``: the whole failure this
    layer exists to prevent is state stored under a per-job directory that a
    restarted job cannot find.
    """
    from_env = os.environ.get("PYCLAD_RESUME_DIR")
    if from_env:
        return pathlib.Path(from_env).expanduser()
    return pathlib.Path.home() / "pyclad-checkpoints"


# ------------------------------------------------------------------------ parsing


_RESUME_HELP = """\
Checkpoint / resume options (this wrapper only):
  --resume-dir PATH             Root for checkpoint directories. Must be outside
                                --output-dir so it survives a requeue.
                                (default: $PYCLAD_RESUME_DIR or ~/pyclad-checkpoints)
  --resume                      Resume from a checkpoint if one exists for this
                                exact configuration. Without it the run starts at
                                concept 0 but still writes checkpoints.
  --resume-require              Like --resume, but abort if no usable checkpoint
                                is found instead of starting over.
  --resume-phase {committed,pending}
                                'committed' resumes after the last fully evaluated
                                concept; 'pending' also replays the interrupted
                                concept's evaluation sweep. (default: committed)
  --allow-code-drift            Accept a checkpoint written by a different pyCLAD
                                commit/version. Recorded in the result JSON under
                                resume.version_drift.
  --[no-]checkpointing          Write checkpoints at every concept boundary.
                                (default: enabled)
  --checkpoint-keep N           Retained checkpoints per run; 0 means unlimited.
                                (default: the method's own recommendation)
  --resume-force-lock           Take over a checkpoint lock held by another
                                process. Only safe when that process is dead.
"""


def parse_args() -> argparse.Namespace:
    """Parse this wrapper's flags, then hand the rest to the base parser."""
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        help="Root directory for checkpoints (must be outside --output-dir)",
    )
    extra_parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from an existing checkpoint for this configuration if one exists",
    )
    extra_parser.add_argument(
        "--resume-require",
        action="store_true",
        default=False,
        help="Resume, and abort if no usable checkpoint is found",
    )
    extra_parser.add_argument(
        "--resume-phase",
        choices=list(RESUME_PHASES),
        default="committed",
        help="Resume after the last committed concept, or replay a pending one",
    )
    extra_parser.add_argument(
        "--allow-code-drift",
        action="store_true",
        default=False,
        help="Accept a checkpoint written by a different pyCLAD version or commit",
    )
    extra_parser.add_argument(
        "--checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a state checkpoint at every concept boundary",
    )
    extra_parser.add_argument(
        "--checkpoint-keep",
        type=int,
        default=_KEEP_FROM_METHOD,
        help="Retained checkpoints per run (0 = unlimited); default is method-specific",
    )
    extra_parser.add_argument(
        "--resume-force-lock",
        action="store_true",
        default=False,
        help="Take over a checkpoint lock held by another process",
    )

    extra_args, remaining = extra_parser.parse_known_args()

    # The base parser owns --help. Print this wrapper's section first so a single
    # --help shows both, then let the base parser print its own and exit.
    if any(argument in {"-h", "--help"} for argument in sys.argv[1:]):
        print(_RESUME_HELP)

    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], *remaining]
        args = _BASE.parse_args()
    finally:
        sys.argv = original_argv

    args.resume_dir = extra_args.resume_dir
    args.resume = bool(extra_args.resume or extra_args.resume_require)
    args.resume_require = bool(extra_args.resume_require)
    args.resume_phase = extra_args.resume_phase
    args.allow_code_drift = bool(extra_args.allow_code_drift)
    args.checkpointing = bool(extra_args.checkpointing)
    args.checkpoint_keep = int(extra_args.checkpoint_keep)
    args.resume_force_lock = bool(extra_args.resume_force_lock)
    return args


def resume_mode(args: argparse.Namespace) -> str:
    if args.resume_require:
        return "require"
    return "auto" if args.resume else "off"


# ---------------------------------------------------------------- result snapshot


class ResumeSafeJsonCheckpointCallback(_BASE.JsonCheckpointCallback):
    """``JsonCheckpointCallback`` that only touches the result JSON on commit.

    ``CheckpointStore.commit()`` records the sha256 of the result JSON at the
    moment of commit, and a later resume refuses to pair a checkpoint with a
    JSON that has changed since. The base callback rewrites the result file at
    ``before_scenario`` and again at every ``before_concept_processing``, so
    even an uninterrupted run invalidates the sha of the entry it just
    committed: crash during concept *k+1* and the resume aborts on a mismatch
    it caused itself.

    The fix is to move those two writes to a sibling *progress* file. The
    canonical result JSON is then written in exactly one place — after the
    evaluation sweep, immediately before the state callback commits — and its
    structure is unchanged, because the write itself is still the inherited
    ``_checkpoint`` -> ``write_result_snapshot`` path.

    Live progress is not lost: the progress file carries the same payload,
    including ``run_status``, and is also where the crash handler dumps partial
    results.
    """

    def __init__(
        self,
        output_path: pathlib.Path,
        run_status_provider: Any,
        write_snapshot: Callable[[], None],
        write_progress_snapshot: Callable[[], None],
        progress_path: pathlib.Path,
    ):
        super().__init__(
            output_path=output_path,
            run_status_provider=run_status_provider,
            write_snapshot=write_snapshot,
        )
        self._write_progress_snapshot = write_progress_snapshot
        self._progress_path = progress_path

    def before_scenario(self, *args, **kwargs):
        self._run_status_provider.mark_run_started()
        self._progress("run start")

    def before_concept_processing(self, concept, *args, **kwargs):
        self._concept_index += 1
        self._run_status_provider.mark_concept_started(concept.name, self._concept_index)
        self._progress(f"concept '{concept.name}' start")

    def _progress(self, phase: str) -> None:
        try:
            self._write_progress_snapshot()
        except Exception:
            logger.exception("Failed to write %s progress snapshot to %s", phase, self._progress_path)


# ------------------------------------------------------------- method construction


def method_key(args: argparse.Namespace) -> str:
    """Map ``(--model, --strategy)`` onto a checkpointable method, or explain."""
    key = SUPPORTED_METHODS.get((args.model, args.strategy))
    if key is not None:
        return key
    supported = "\n  ".join(f"--model {model} --strategy {strategy}" for model, strategy in sorted(SUPPORTED_METHODS))
    raise SystemExit(
        f"run_continual_visual_ad_resumable.py has no checkpointable implementation for "
        f"--model {args.model} --strategy {args.strategy}.\n"
        f"Supported combinations:\n  {supported}\n"
        "Refusing to run: a run that looks checkpointed but is not would be discovered "
        "only when a resume silently restarted it from concept 0. Use "
        "examples/clvad/run_continual_visual_ad.py for the other methods."
    )


def build_checkpointable_method(
    args: argparse.Namespace,
    seed: int,
    output_path: pathlib.Path,
) -> dict[str, Any]:
    """Build the v2 model + strategy + scenario class for the selected method.

    The heavy, method-specific dependencies (timm, diffusers, segment-anything)
    are imported here and nowhere else, so ``--help`` works in an environment
    that has none of them.
    """
    key = method_key(args)
    config = _BASE.build_model_config(args, seed)

    if key == "ucad":
        from pyclad.resume.methods import ucad

        model = ucad.ResumableUCAD(config)
        strategy = ucad.ResumableUCADStrategy(model)
        return {
            "key": key,
            "model": model,
            "config": config,
            "strategy": strategy,
            "scenario_cls": ResumableConceptAwareScenario,
            "keep": getattr(ucad, "RECOMMENDED_KEEP", DEFAULT_CHECKPOINT_KEEP),
            "run_payload": {},
        }

    if key == "replaycad":
        from pyclad.resume.methods import replaycad
        from pyclad.strategies.replaycad import ReplayCADMemory

        replaycad_config = _BASE.build_replaycad_config(args=args, seed=seed, output_path=output_path)
        if replaycad_config is None:  # pragma: no cover - guarded by method_key
            raise SystemExit("ReplayCAD configuration could not be built")
        # Hoisted out of the base build_strategy on purpose: the same memory
        # instance has to reach both the strategy and, through
        # attach_replaycad_memory, the model whose checkpoint records which
        # concepts are activated for replay.
        memory = ReplayCADMemory(replaycad_config)
        model = replaycad.CheckpointedPatchCore(config, replaycad_config=replaycad_config)
        strategy = replaycad.ResumableReplayCADStrategy(model, memory)
        return {
            "key": key,
            "model": model,
            "config": config,
            "strategy": strategy,
            "scenario_cls": ResumableConceptAwareScenario,
            "keep": getattr(replaycad, "RECOMMENDED_KEEP", DEFAULT_CHECKPOINT_KEEP),
            "run_payload": {"replaycad_config": replaycad_config.model_dump(mode="json")},
        }

    from pyclad.resume.methods import one_for_more

    model = one_for_more.ResumableOneForMore(config)
    strategy = one_for_more.ResumableNaiveStrategy(model)
    return {
        "key": key,
        "model": model,
        "config": config,
        "strategy": strategy,
        # One-for-More has no strategy of its own; the base runner routes
        # NaiveStrategy through the *incremental* scenario, which passes no
        # concept_id. Using the aware scenario here would silently diverge from
        # the published routing.
        "scenario_cls": ResumableConceptIncrementalScenario,
        "keep": getattr(one_for_more, "RECOMMENDED_KEEP", DEFAULT_CHECKPOINT_KEEP),
        "run_payload": {},
    }


def state_determinism(model: Any) -> dict[str, Any] | None:
    """Read the determinism report; the accessor name is not uniform."""
    for name in ("determinism", "determinism_report"):
        accessor = getattr(model, name, None)
        if callable(accessor):
            return accessor()
    return None


# ------------------------------------------------------------------- preconditions


def validate_replaycad_durability(args: argparse.Namespace, output_dir: pathlib.Path) -> None:
    """Refuse a ReplayCAD run whose artifacts cannot survive a restart.

    ``build_replaycad_config`` defaults the artifact directory *inside* the
    output directory. With a per-job output directory that means every restart
    re-runs ~40 minutes of condition optimisation per concept against artifacts
    it already owns — and ``ResumableReplayCADStrategy.activate_completed`` then aborts
    because the conditions the checkpoint names are gone.
    """
    if args.strategy != "replaycad" or not args.checkpointing:
        return

    if args.replaycad_artifact_dir is None:
        raise SystemExit(
            "--replaycad-artifact-dir is required when checkpointing is enabled.\n"
            "Without it the conditional artifacts default to a subdirectory of --output-dir, which is "
            "per-job: a requeued job would find no conditions for the concepts its checkpoint says are "
            "complete, and the resume would abort.\n"
            "Point it at a job-id-independent location, e.g. "
            "$HOME/replaycad-artifacts/<dataset>/<order>/<gpu>/<backbone>, or pass --no-checkpointing."
        )

    artifact_dir = pathlib.Path(args.replaycad_artifact_dir).expanduser().resolve()
    resolved_output = output_dir.expanduser().resolve()
    if artifact_dir == resolved_output or resolved_output in artifact_dir.parents:
        raise SystemExit(
            f"--replaycad-artifact-dir ({artifact_dir}) is inside --output-dir ({resolved_output}). "
            "ReplayCAD conditions must outlive a single job's output directory."
        )

    if not args.replaycad_cache_generated:
        raise SystemExit(
            "--replaycad-cache-generated is required when checkpointing is enabled.\n"
            "ResumableReplayCADStrategy.resume_restore refuses to resume without the generation cache: every "
            "step would otherwise regenerate its replay images from the diffusion pipeline, which costs "
            "hours per concept and does not reproduce byte-for-byte on the stable-diffusion family.\n"
            "Add --replaycad-cache-generated, or pass --no-checkpointing."
        )


def describe_near_miss_checkpoints(
    resume_root: pathlib.Path,
    store: CheckpointStore,
    *,
    concept_order: list[str],
    fingerprint: dict[str, Any],
    state_kind: str,
    seed: int,
) -> list[tuple[pathlib.Path, int, list[str]]]:
    """Explain why a checkpoint directory this run *should* have matched, didn't.

    A run id is a hash. When it changes, ``--resume`` finds nothing and — in
    ``auto`` mode — silently starts at concept 0, which is precisely the failure
    that burns a whole allocation before anyone looks. So when there is no
    manifest at our own run id, look at the neighbours and name the fields that
    differ.

    Returns ``(directory, completed_concepts, differences)`` per neighbour. An
    empty ``differences`` list is the dangerous case: every field the manifest
    records agrees, so what changed is one of the run-id inputs the manifest
    does *not* record.
    """
    reports: list[tuple[pathlib.Path, int, list[str]]] = []
    if not resume_root.is_dir():
        return reports

    for manifest_path in sorted(resume_root.glob("*/manifest.json")):
        if manifest_path.parent.name == store.run_id:
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not manifest.get("committed") and not manifest.get("history"):
            continue
        if str(manifest.get("state_kind") or "") != state_kind:
            continue

        differences: list[str] = []
        recorded_order = [str(name) for name in manifest.get("concept_order", [])]
        if recorded_order and recorded_order != concept_order:
            differences.append(
                f"concept_order (checkpoint has {len(recorded_order)} concepts starting "
                f"{recorded_order[:3]}, this run has {len(concept_order)} starting {concept_order[:3]})"
            )
        recorded_seed = manifest.get("seed")
        if recorded_seed is not None and int(recorded_seed) != int(seed):
            differences.append(f"seed (checkpoint={recorded_seed}, run={seed})")
        for diff in compare_fingerprints(manifest.get("fingerprint", {}) or {}, fingerprint):
            differences.append(str(diff))

        committed = manifest.get("committed") or {}
        completed = int(committed.get("index", -1)) + 1
        reports.append((manifest_path.parent, completed, differences))
    return reports


#: Inputs to ``compute_run_id`` that the manifest does not record, so a change in
#: one of them is invisible to :func:`describe_near_miss_checkpoints`.
_RUN_ID_ONLY_INPUTS = (
    "--benchmark",
    "--eval-level",
    "--resize",
    "--max-train",
    "--max-test",
    "--categories",
    "--ordering-mode",
    "--ordering-file",
    "--inclad-bench",
)


def report_missing_checkpoint(
    resume_root: pathlib.Path,
    store: CheckpointStore,
    *,
    concept_order: list[str],
    fingerprint: dict[str, Any],
    state_kind: str,
    seed: int,
) -> None:
    """``--resume`` found nothing at our run id: say why, and abort if warranted.

    Starting over is legitimate exactly once — the first submission of a chained
    job. Every other reason is a configuration change, and the expensive failure
    mode is discovering it forty GPU-hours later.
    """
    near_misses = describe_near_miss_checkpoints(
        resume_root,
        store,
        concept_order=concept_order,
        fingerprint=fingerprint,
        state_kind=state_kind,
        seed=seed,
    )

    identical = [item for item in near_misses if not item[2]]
    if identical:
        rendered = "\n  ".join(
            f"{directory} ({completed} concept(s) completed)" for directory, completed, _ in identical
        )
        raise SystemExit(
            f"--resume found no checkpoint at {store.directory}, but every field these checkpoints record "
            f"agrees with this run:\n  {rendered}\n"
            "Only the run id differs, so one of the run-identity inputs the manifest does not record "
            f"changed: {', '.join(_RUN_ID_ONLY_INPUTS)}.\n"
            "Restore the original value or pass --no-resume to deliberately start over. Refusing to "
            "silently restart at concept 0."
        )

    for directory, completed, differences in near_misses:
        logger.error(
            "  --resume found no checkpoint at %s, but %s (%d concept(s) completed) holds one for the same "
            "method that differs: %s",
            store.directory,
            directory,
            completed,
            "; ".join(differences),
        )
    logger.warning(
        "  --resume was given but no checkpoint exists yet at %s; starting from concept 0. "
        "This is expected for the first submission of a chained job.",
        store.directory,
    )


# ------------------------------------------------------------------------- the run


def run_single_resumable(
    args: argparse.Namespace,
    seed: int,
    run_index: int,
    total_runs: int,
    dataset: ConceptsDataset,
    output_dir: pathlib.Path,
    resume_root: pathlib.Path,
) -> pathlib.Path:
    """One seed, checkpointed at every concept boundary and resumable.

    Mirrors ``run_continual_visual_ad.run_single`` step for step — same output
    paths, same providers, same snapshot writer — and diverges only where the
    docstring at the top of this file says it does.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ordering_suffix = _BASE.ordering_output_suffix(args)
    replay_suffix = _BASE.replay_buffer_file_suffix(args)
    stem = f"{args.model}_{args.strategy}{ordering_suffix}{replay_suffix}_seed_{seed}"
    output_path = output_dir / f"{stem}.json"
    log_path = output_dir / f"{stem}.log"
    progress_path = output_dir / f"{stem}.progress.json"

    with _BASE.seed_log_handler(log_path):
        run_status = _BASE.RunStatusProvider(output_path=output_path, log_path=log_path)
        run_payload = _BASE.create_run_config_payload(
            args=args,
            seed=seed,
            output_path=output_path,
            log_path=log_path,
        )
        run_provider = _BASE.RunConfigProvider(run_payload)

        model: Any = None
        config: Any = None
        run_dataset: ConceptsDataset | None = None
        strategy: Any = None
        callbacks: list[Callback] = []
        store: CheckpointStore | None = None
        # Everything before the scenario starts is setup and validation. A
        # failure there is a configuration error, and marking the run "failed"
        # and writing a partial result for it would only add noise to the
        # message the operator actually needs to read.
        scenario_started = False

        def providers() -> list[InfoProvider]:
            return _BASE.collect_result_providers(
                model=model,
                run_dataset=run_dataset,
                strategy=strategy,
                callbacks=callbacks,
                extra_providers=[run_provider, run_status],
            )

        def write_snapshot() -> None:
            _BASE.write_result_snapshot(
                output_path=output_path,
                providers=providers(),
                completed_concepts=run_status.completed_concepts(),
            )

        def write_progress_snapshot() -> None:
            _BASE.write_result_snapshot(
                output_path=progress_path,
                providers=providers(),
                completed_concepts=run_status.completed_concepts(),
            )

        logger.info("═══ Run %d/%d  seed=%d ═══", run_index + 1, total_runs, seed)
        _BASE.set_global_seeds(seed)

        try:
            logger.info("  [1/6] Resolving concept order and run dataset")
            fixed_order, ordering_metadata = _BASE.resolve_fixed_concept_order(args, dataset)
            if fixed_order is not None:
                run_dataset = _BASE.reorder_dataset(dataset, fixed_order)
                logger.info(
                    "  Concept order source: %s (%s)",
                    ordering_metadata["source"],
                    ordering_metadata.get("ordering_file", "inline"),
                )
            elif args.shuffle_concepts:
                run_dataset = _BASE.shuffle_dataset(dataset, seed)
                ordering_metadata = {"source": "shuffle", "seed": seed}
                logger.info("  Concept order source: shuffle (seed=%d)", seed)
            else:
                run_dataset = dataset
                ordering_metadata = {"source": "dataset"}
                logger.info("  Concept order source: dataset")

            # Everything below — the run id, the resume plan, every restored
            # matrix row — is keyed on this list, so it must come from the
            # *reordered* dataset rather than from args.
            concept_order = [concept.name for concept in run_dataset.train_concepts()]
            run_status.set_total_concepts(len(concept_order))

            if ordering_metadata.get("skipped_concepts"):
                logger.warning("  Skipped concepts not present in dataset: %s", ordering_metadata["skipped_concepts"])
            logger.info("  Concept order: %s", concept_order)
            _BASE.log_dataset_overview(run_dataset, "  Run dataset order:")

            logger.info("  [2/6] Building checkpointable model and strategy")
            built = build_checkpointable_method(args, seed, output_path)
            model = built["model"]
            config = built["config"]
            strategy = built["strategy"]
            if args.eval_level in {"pixel", "both"} and not callable(getattr(model, "score_maps", None)):
                raise ValueError(
                    f"Model '{args.model}' does not expose score_maps(), so --eval-level {args.eval_level} "
                    "is unavailable"
                )
            run_payload.update(
                _BASE.create_run_config_payload(
                    args=args,
                    seed=seed,
                    output_path=output_path,
                    log_path=log_path,
                    config=config,
                    run_dataset=run_dataset,
                    ordering_metadata=ordering_metadata,
                )
            )
            run_payload.update(built["run_payload"])
            logger.info("  Model device: %s", getattr(model, "_device", "unknown"))
            logger.info("  Checkpointable classes: %s / %s", type(model).__name__, type(strategy).__name__)

            logger.info("  [3/6] Opening the checkpoint store")
            # The *resolved* resize, not args.resize: a job that omits --resize
            # and a requeue that spells out the model default must hash alike,
            # otherwise 'auto' quietly starts from concept 0.
            resolved_resize = (
                int(args.resize) if args.resize is not None else _BASE.MODEL_SPECS[args.model].default_resize
            )
            ordering_file = getattr(args, "ordering_file", None)
            run_id = compute_run_id(
                model=args.model,
                strategy=args.strategy,
                benchmark=args.benchmark,
                seed=seed,
                concept_order=concept_order,
                eval_level=args.eval_level,
                resize=resolved_resize,
                extra={
                    # Things that change the data but are invisible both to the
                    # config fingerprint and to the concept-name list.
                    "max_train": args.max_train,
                    "max_test": args.max_test,
                    "categories": args.categories,
                    "ordering_mode": args.ordering_mode,
                    "ordering_file": str(ordering_file) if ordering_file else None,
                    "inclad_bench": bool(getattr(args, "inclad_bench", False)),
                },
            )
            keep = args.checkpoint_keep if args.checkpoint_keep != _KEEP_FROM_METHOD else int(built["keep"])
            store = CheckpointStore(resume_root, run_id, keep=keep)
            store.acquire_lock(force=args.resume_force_lock)
            logger.info("  Checkpoint directory: %s (keep=%s)", store.directory, keep or "unlimited")

            fingerprint = model.state_fingerprint()
            state_kind = type(model).STATE_KIND
            committed = store.committed()

            if committed is not None and committed.index + 1 >= len(concept_order) and output_path.is_file():
                logger.info(
                    "  This run already completed all %d concepts (checkpoint committed concept %d, '%s'); "
                    "nothing to do. Result: %s",
                    len(concept_order),
                    committed.index,
                    committed.concept_id,
                    output_path,
                )
                return output_path

            mode = resume_mode(args)
            if mode == "off" and committed is not None:
                logger.warning(
                    "  A checkpoint for this exact configuration exists (%d concept(s) completed, last '%s') "
                    "but --resume was not given, so this run starts at concept 0 and will overwrite it. "
                    "Pass --resume if that is not what you meant.",
                    committed.index + 1,
                    committed.concept_id,
                )
            if mode != "off" and committed is None and store.pending() is None:
                report_missing_checkpoint(
                    resume_root,
                    store,
                    concept_order=concept_order,
                    fingerprint=fingerprint,
                    state_kind=state_kind,
                    seed=seed,
                )

            logger.info("  [4/6] Planning the resume")
            outcome = build_resume_plan(
                store,
                concept_order=concept_order,
                mode=mode,
                phase=args.resume_phase,
                expected_kind=state_kind,
                expected_state_version=type(model).STATE_VERSION,
                expected_fingerprint=fingerprint,
                # Deliberately None: the manifest records the JSON it was
                # committed against, and that pairing is sha-verified. Forcing
                # our own path here would break a run whose --output-dir moved
                # because the scheduler handed it a new job id.
                result_json_path=None,
                allow_version_drift=args.allow_code_drift,
            )
            if outcome.entry is not None and outcome.entry.result_json:
                recorded_json = pathlib.Path(outcome.entry.result_json)
                if recorded_json != output_path:
                    logger.warning(
                        "  The checkpoint was committed against %s but this run writes %s. The output "
                        "directory moved (a new scheduler job id?). Metrics are restored from the recorded "
                        "path and a complete result JSON is written at the new one; pin OUTPUT_DIR to keep "
                        "one file per run.",
                        recorded_json,
                        output_path,
                    )
            for warning in outcome.warnings:
                logger.warning("  Resume: %s", warning)

            logger.info("  [5/6] Building callbacks and restoring state")
            callbacks = _BASE.build_callbacks(run_dataset, strategy=strategy, args=args)
            if args.checkpointing or args.resume:
                json_callback: Any = ResumeSafeJsonCheckpointCallback(
                    output_path=output_path,
                    run_status_provider=run_status,
                    write_snapshot=write_snapshot,
                    write_progress_snapshot=write_progress_snapshot,
                    progress_path=progress_path,
                )
            else:
                json_callback = _BASE.JsonCheckpointCallback(
                    output_path=output_path,
                    run_status_provider=run_status,
                    write_snapshot=write_snapshot,
                )
            callbacks.append(json_callback)

            report = apply_resume(
                outcome,
                # The model, not the strategy: load_state lives on the model.
                state=model,
                strategy=strategy,
                callbacks=callbacks,
                train_concepts=run_dataset.train_concepts(),
                # Without this the snapshot writer filters every restored matrix
                # row out again, because it keys on run_status.completed_concepts().
                run_status_provider=run_status,
                json_checkpoint_callback=json_callback,
            )
            _prime_progress_callbacks(callbacks, len(outcome.completed_concepts))

            resume_info = {
                **outcome.info(),
                "run_id": store.run_id,
                "resume_dir": str(store.directory),
                "phase": args.resume_phase,
                "checkpointing": bool(args.checkpointing),
                "allow_code_drift": bool(args.allow_code_drift),
                "restore": report.info(),
            }
            strategy.set_resume_info(resume_info)
            run_payload["resume"] = resume_info
            if outcome.resumed:
                logger.info(
                    "  Resumed at concept %d/%d ('%s'); %d concept(s) restored",
                    outcome.plan.resume_index + 1,
                    len(concept_order),
                    (
                        concept_order[outcome.plan.resume_index]
                        if outcome.plan.resume_index < len(concept_order)
                        else "<none left>"
                    ),
                    len(outcome.completed_concepts),
                )
            else:
                logger.info("  Starting from concept 1/%d", len(concept_order))

            # Registered last, and constructed only now. The determinism report
            # is snapshotted at construction, and ResumableNaiveStrategy.resume_restore
            # (inside apply_resume, above) can downgrade ResumableOneForMore's
            # guarantee to 'statistical'. Building it earlier would publish a
            # bit-identical claim this run cannot honour.
            state_callback = StateCheckpointCallback(
                store,
                model,
                concept_order=concept_order,
                seed=seed,
                result_json_path=output_path,
                determinism=state_determinism(model),
                plan=outcome.plan,
                time_callback=_find_time_callback(callbacks),
                pending_train_seconds=outcome.pending_train_seconds,
                enabled=bool(args.checkpointing),
            )
            callbacks.append(state_callback)

            scenario = built["scenario_cls"](
                dataset=run_dataset,
                strategy=strategy,
                callbacks=callbacks,
                plan=outcome.plan,
            )

            logger.info("  [6/6] Running scenario")
            t0 = time.perf_counter()
            scenario_started = True
            scenario.run()
            elapsed = time.perf_counter() - t0
        except BaseException as exc:
            if not scenario_started:
                logger.error("  Run aborted before the scenario started: %s: %s", type(exc).__name__, exc)
                raise
            run_status.mark_failed(exc, traceback.format_exc())
            logger.exception("  Run did not finish cleanly; saving partial results to %s", progress_path)
            try:
                # The canonical result JSON is deliberately not touched here: the
                # manifest holds its sha256, and rewriting it now would make the
                # next resume abort on a mismatch this handler caused.
                write_progress_snapshot()
            except Exception:
                logger.exception("  Failed to save partial results to %s", progress_path)
            if store is not None and store.committed() is not None:
                logger.error(
                    "  Resumable: rerun the identical command with --resume to continue from %s",
                    store.directory,
                )
            raise
        finally:
            if store is not None:
                store.release_lock()

        logger.info("  Saving results")
        run_status.mark_completed(elapsed_seconds=elapsed)
        skipped_callbacks = [
            type(callback).__name__ for callback in callbacks if not isinstance(callback, InfoProvider)
        ]
        if skipped_callbacks:
            logger.info("    Skipping non-serializable callbacks in JSON output: %s", skipped_callbacks)
        write_snapshot()

        logger.info("    ✓ finished in %.1fs → %s", elapsed, output_path)

    return output_path


def _find_time_callback(callbacks: list[Callback]) -> Any | None:
    """The timing callback, needed to restore a replayed concept's wall clock."""
    for callback in callbacks:
        if hasattr(callback, "_time_by_concept") and hasattr(callback, "_train_time_total"):
            return callback
    return None


def _prime_progress_callbacks(callbacks: list[Callback], completed_count: int) -> None:
    """Continue the ``[07/15]`` progress counter instead of restarting it."""
    if completed_count <= 0:
        return
    for callback in callbacks:
        if hasattr(callback, "_current_train_index") and hasattr(callback, "_train_total"):
            callback._current_train_index = completed_count


# ----------------------------------------------------------------------- entrypoint


def _normalise_benchmark(name: str | None) -> str:
    """Fold the spellings people actually type into the manifest's own name.

    The HF config is literally "inclad-md", but the older result files call the
    same stream "multidataset_seq_v1" and shell users reach for "multidataset".
    Accepting all of them here is cheaper than a support round-trip.
    """
    if not name:
        return ""
    folded = str(name).strip().lower().replace("_", "-")
    if folded in {"inclad-md", "incladmd", "multidataset", "multidataset-seq-v1", "md"}:
        return "inclad-md"
    return folded


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.getLogger("pyclad.scenarios.concept_incremental").setLevel(logging.WARNING)

    args = _BASE.normalize_replaycad_args(args)
    method_key(args)  # fail before loading a dataset if the method is unsupported

    output_dir = _BASE.resolve_output_dir(args)
    resume_root = pathlib.Path(args.resume_dir).expanduser() if args.resume_dir else default_resume_root()
    ensure_resume_dir_outside_output(resume_root, output_dir)
    validate_replaycad_durability(args, output_dir)

    seeds = _BASE.generate_seeds(args.master_seed, args.n_runs)
    logger.info("Master seed: %d  →  per-run seeds: %s", args.master_seed, seeds)
    logger.info("Results directory: %s", output_dir)
    logger.info(
        "Checkpointing: %s | resume: %s (phase=%s) | checkpoint root: %s",
        "on" if args.checkpointing else "off",
        resume_mode(args),
        args.resume_phase,
        resume_root,
    )
    if args.output_dir is None:
        logger.warning(
            "No --output-dir was given, so results go to a timestamped directory (%s). A resumed run "
            "would write beside the original instead of continuing it; pass --output-dir explicitly.",
            output_dir,
        )

    logger.info("[1/2] Loading dataset")
    spec = _BASE.MODEL_SPECS[args.model]
    resize = int(args.resize) if args.resize is not None else spec.default_resize
    resize_to = (resize, resize)
    logger.info(
        "Dataset options: categories=%s, resize=%s, max_train=%s, max_test=%s",
        args.categories if args.categories is not None else "all",
        resize_to,
        args.max_train,
        args.max_test,
    )

    dataset_load_started_at = time.perf_counter()
    if args.inclad_bench:
        inclad_ordering = args.ordering_mode if args.ordering_mode else "easy_to_hard"

        # InCLAD-MD is a separate HF config ("inclad-md") holding one 27-concept
        # stream drawn from all five benchmarks, with concepts named
        # "<dataset>__<category>". It needs a different loader from the
        # per-dataset InCLAD-Bench configs, which is why the base runner --
        # having no such branch -- fails with "Could not resolve local root for
        # visual benchmark 'inclad-md'". The branch below mirrors the one in
        # examples/clvad/run_continual_visual_ad_levels.py.
        if _normalise_benchmark(args.benchmark) == "inclad-md":
            from pyclad.data.datasets.inclad_md_dataset import (
                INCLAD_MD_ORDERINGS,
                InCLADMDDataset,
            )

            if inclad_ordering not in INCLAD_MD_ORDERINGS:
                raise ValueError(
                    f"InCLAD-MD has no '{inclad_ordering}' ordering; the manifest ships "
                    f"only {INCLAD_MD_ORDERINGS}. Requesting a missing ordering would "
                    f"otherwise surface as an opaque HuggingFace split error."
                )

            # InCLADMDDataset resolves the five source roots itself and takes no
            # `root=`; it reads PYCLAD_VISUAL_DATASETS_ROOT. Set it from --root
            # when given so the flag is not silently ignored.
            if args.root:
                os.environ["PYCLAD_VISUAL_DATASETS_ROOT"] = str(args.root)

            # Loading the data is only half of it. The pixel-metric callbacks
            # index ground-truth masks through a separate path that ends in the
            # reader-preset lookup, which has no 'inclad-md' entry -- so with
            # --eval-level pixel|both the run dies in "[5/6] Building callbacks"
            # with "Unsupported visual benchmark 'inclad-md'" even though the
            # dataset above loaded all 27 concepts. This installs the same
            # manifest-based indexer the grouped-schedule MD runner uses.
            _md_pixel_dir = str(pathlib.Path(__file__).resolve().parent)
            if _md_pixel_dir not in sys.path:
                sys.path.insert(0, _md_pixel_dir)
            import _inclad_md_pixel

            # index_md_test_samples reads the ordering from the environment; keep
            # it consistent with the stream actually being run.
            os.environ["INCLAD_MD_ORDERING"] = str(inclad_ordering)
            _inclad_md_pixel.install()

            logger.info("Loading InCLAD-MD manifest: ordering=%s", inclad_ordering)
            dataset = InCLADMDDataset(
                ordering=inclad_ordering,
                registry_path=args.registry_path,
                categories=args.categories,
                data_mode="numpy",
                resize_to=resize_to,
                color_mode="rgb",
                max_train_samples_per_category=args.max_train,
                max_test_samples_per_category=args.max_test,
            )
        else:
            from pyclad.data.datasets.inclad_bench_dataset import InCLADBenchDataset

            logger.info(
                "Loading InCLAD-Bench manifest: benchmark=%s, ordering=%s",
                args.benchmark,
                inclad_ordering,
            )
            dataset = InCLADBenchDataset(
                benchmark=args.benchmark,
                ordering=inclad_ordering,
                root=args.root,
                registry_path=args.registry_path,
                categories=args.categories,
                data_mode="numpy",
                resize_to=resize_to,
                color_mode="rgb",
                max_train_samples_per_category=args.max_train,
                max_test_samples_per_category=args.max_test,
            )
    else:
        dataset_root = _BASE.resolve_visual_benchmark_root(
            benchmark=args.benchmark,
            root=args.root,
            registry_path=args.registry_path,
        )
        logger.info("Dataset root: %s", dataset_root)
        dataset = _BASE.read_registered_visual_benchmark_dataset(
            benchmark=args.benchmark,
            root=dataset_root,
            registry_path=args.registry_path,
            categories=args.categories,
            data_mode="numpy",
            resize_to=resize_to,
            color_mode="rgb",
            max_train_samples_per_category=args.max_train,
            max_test_samples_per_category=args.max_test,
        )
    logger.info(
        "Dataset loaded in %.1fs: %s (%d train concepts, %d test concepts)",
        time.perf_counter() - dataset_load_started_at,
        dataset.name(),
        len(dataset.train_concepts()),
        len(dataset.test_concepts()),
    )
    _BASE.log_dataset_overview(dataset, "Loaded dataset order:")

    logger.info("[2/2] Starting experiment runs")
    saved_paths: list[pathlib.Path] = []
    for run_index, seed in enumerate(seeds):
        saved_paths.append(
            run_single_resumable(
                args=args,
                seed=seed,
                run_index=run_index,
                total_runs=len(seeds),
                dataset=dataset,
                output_dir=output_dir,
                resume_root=resume_root,
            )
        )

    logger.info("All %d runs complete.", len(saved_paths))
    for path in saved_paths:
        logger.info("  %s", path)


if __name__ == "__main__":
    try:
        main()
    except CheckpointDiskSpaceError as error:
        # A distinct exit code so a scheduler can requeue onto a different node
        # instead of retrying the full one.
        logging.getLogger(__name__).error("Checkpoint write failed for lack of disk space: %s", error)
        raise SystemExit(DISK_FULL_EXIT_CODE)
    except ResumeError as error:
        # Fingerprint, order, provenance, corruption and pairing failures all
        # land here. They are configuration errors, not crashes: print the
        # message on its own rather than burying it in a traceback.
        logging.getLogger(__name__).error("%s: %s", type(error).__name__, error)
        raise SystemExit(2)
