"""Additive checkpoint/resume layer for pyCLAD.

Long continual-learning runs on a scheduler die for reasons that have nothing to
do with the method: a wall-clock limit, a preempted node, a full filesystem.
This package lets such a run pick up at the concept boundary it reached instead
of at concept zero, without editing a single existing file.

Layout:

* :mod:`pyclad.resume.protocol` - the three-method state protocol, the envelope
  schema, and the exception hierarchy;
* :mod:`pyclad.resume.fingerprint` - run identity and the checks that refuse a
  checkpoint written under a different configuration, ordering or commit;
* :mod:`pyclad.resume.store` - durability: atomic writes, the manifest commit
  protocol, sidecars, retention, locking;
* :mod:`pyclad.resume.scenario` - the two scenarios that can skip a completed
  prefix, and the plan describing what to skip;
* :mod:`pyclad.resume.callback` - the callback that writes state at each
  concept boundary;
* :mod:`pyclad.resume.metrics_restore` - rehydrating evaluation callbacks from
  the partial result JSON;
* :mod:`pyclad.resume.plan` - validation and the ordered restore driver.

``pyclad.resume.methods`` holds the per-method state classes and is deliberately
**not** imported here: those modules pull in optional heavy dependencies (timm,
diffusers), and ``from pyclad.resume import CheckpointStore`` must work in an
environment that has none of them.
"""

from pyclad.resume.callback import StateCheckpointCallback
from pyclad.resume.fingerprint import (
    FieldDiff,
    assert_concept_order_matches,
    assert_fingerprint_matches,
    assert_provenance_matches,
    compare_fingerprints,
    compute_run_id,
    current_provenance,
    fingerprint_from_config,
)
from pyclad.resume.metrics_restore import (
    RestoreReport,
    load_result_payload,
    restore_callbacks,
    restore_training_concepts,
    sha256_of_file,
)
from pyclad.resume.plan import RESUME_MODES, RESUME_PHASES, ResumeOutcome, apply_resume, build_resume_plan
from pyclad.resume.protocol import (
    DETERMINISM_BIT_IDENTICAL,
    DETERMINISM_STATISTICAL,
    DISK_FULL_EXIT_CODE,
    ENVELOPE_VERSION,
    PHASE_AFTER_CONCEPT_PROCESSING,
    PHASE_AFTER_TRAINING,
    SIDECAR_KEY,
    CheckpointCorruptionError,
    CheckpointDiskSpaceError,
    CheckpointEnvelopeError,
    CheckpointFingerprintMismatch,
    CheckpointKindError,
    CheckpointLockError,
    CheckpointMissingError,
    CheckpointOrderMismatch,
    CheckpointPayloadError,
    CheckpointResultMismatch,
    CheckpointVersionDrift,
    CheckpointVersionError,
    ResumableState,
    ResumableStrategy,
    ResumeError,
    build_envelope,
    merge_sidecars,
    require_payload_keys,
    split_sidecars,
    validate_envelope,
    validate_state_payload,
)
from pyclad.resume.scenario import (
    ResumableConceptAwareScenario,
    ResumableConceptIncrementalScenario,
    ResumePlan,
)
from pyclad.resume.store import CheckpointEntry, CheckpointStore, SidecarInfo, ensure_resume_dir_outside_output

__all__ = [
    # protocol
    "ENVELOPE_VERSION",
    "SIDECAR_KEY",
    "PHASE_AFTER_TRAINING",
    "PHASE_AFTER_CONCEPT_PROCESSING",
    "DETERMINISM_BIT_IDENTICAL",
    "DETERMINISM_STATISTICAL",
    "DISK_FULL_EXIT_CODE",
    "ResumableState",
    "ResumableStrategy",
    "build_envelope",
    "validate_envelope",
    "validate_state_payload",
    "require_payload_keys",
    "split_sidecars",
    "merge_sidecars",
    # errors
    "ResumeError",
    "CheckpointEnvelopeError",
    "CheckpointKindError",
    "CheckpointVersionError",
    "CheckpointFingerprintMismatch",
    "CheckpointOrderMismatch",
    "CheckpointPayloadError",
    "CheckpointCorruptionError",
    "CheckpointDiskSpaceError",
    "CheckpointLockError",
    "CheckpointMissingError",
    "CheckpointResultMismatch",
    "CheckpointVersionDrift",
    # fingerprint
    "FieldDiff",
    "compute_run_id",
    "fingerprint_from_config",
    "compare_fingerprints",
    "assert_fingerprint_matches",
    "assert_concept_order_matches",
    "assert_provenance_matches",
    "current_provenance",
    # store
    "CheckpointStore",
    "CheckpointEntry",
    "SidecarInfo",
    "ensure_resume_dir_outside_output",
    # scenario + callback
    "ResumePlan",
    "ResumableConceptAwareScenario",
    "ResumableConceptIncrementalScenario",
    "StateCheckpointCallback",
    # restore
    "RESUME_MODES",
    "RESUME_PHASES",
    "ResumeOutcome",
    "build_resume_plan",
    "apply_resume",
    "RestoreReport",
    "restore_callbacks",
    "restore_training_concepts",
    "load_result_payload",
    "sha256_of_file",
]
