import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pyclad.data.concept import Concept
from pyclad.output.output_writer import InfoProvider


def _load_visual_runner_module():
    module_path = Path(__file__).resolve().parents[2] / "examples" / "clvad" / "run_continual_visual_ad.py"
    spec = importlib.util.spec_from_file_location("test_visual_runner_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _StaticProvider(InfoProvider):
    def __init__(self, payload):
        self._payload = payload

    def info(self):
        return self._payload


def test_checkpoint_callback_writes_running_and_partial_status(tmp_path):
    runner = _load_visual_runner_module()
    output_path = tmp_path / "run.json"
    log_path = tmp_path / "run.log"
    status_provider = runner.RunStatusProvider(output_path=output_path, log_path=log_path, total_concepts=2)
    payload_provider = _StaticProvider({"payload": {"value": 1}})

    def write_snapshot():
        runner.write_result_snapshot(
            output_path,
            [payload_provider, status_provider],
            completed_concepts=status_provider.completed_concepts(),
        )

    callback = runner.JsonCheckpointCallback(
        output_path=output_path,
        run_status_provider=status_provider,
        write_snapshot=write_snapshot,
    )

    callback.before_scenario()
    started_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert started_payload["run_status"]["state"] == "running"
    assert started_payload["run_status"]["completed_concepts_count"] == 0

    concept = Concept(name="alpha", data=np.asarray([], dtype=np.float32))
    callback.before_concept_processing(concept)
    in_progress_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert in_progress_payload["run_status"]["active_concept"] == "alpha"
    assert in_progress_payload["run_status"]["active_concept_index"] == 1

    callback.after_concept_processing(concept)
    partial_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert partial_payload["run_status"]["completed_concepts"] == ["alpha"]
    assert partial_payload["run_status"]["completed_concepts_count"] == 1
    assert partial_payload["run_status"]["last_completed_concept"] == "alpha"
    assert partial_payload["run_status"]["active_concept"] is None


def test_ordering_output_suffix_uses_ordering_file_stem():
    runner = _load_visual_runner_module()
    args = SimpleNamespace(
        ordering_file="/tmp/btech_hard_to_easy_roc-auc.csv",
        ordering_mode=None,
        shuffle_concepts=False,
        concept_order=None,
    )

    assert runner.ordering_output_suffix(args) == "_order-btech-hard-to-easy-roc-auc"


def test_ordering_output_suffix_marks_shuffle_runs():
    runner = _load_visual_runner_module()
    args = SimpleNamespace(
        ordering_file=None,
        ordering_mode=None,
        shuffle_concepts=True,
        concept_order=None,
    )

    assert runner.ordering_output_suffix(args) == "_order-shuffle"
