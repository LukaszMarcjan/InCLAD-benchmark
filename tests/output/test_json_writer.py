import importlib.util
import json
import sys
from pathlib import Path

import pytest

from pyclad.output.output_writer import InfoProvider


class _StaticProvider(InfoProvider):
    def __init__(self, payload):
        self._payload = payload

    def info(self):
        return self._payload


def _load_visual_runner_module():
    module_path = Path(__file__).resolve().parents[2] / "examples" / "clvad" / "run_continual_visual_ad.py"
    spec = importlib.util.spec_from_file_location("test_visual_runner_module_for_writer", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runner_snapshot_replaces_target_atomically(tmp_path):
    runner = _load_visual_runner_module()
    output_path = tmp_path / "result.json"
    output_path.write_text('{"old": true}', encoding="utf-8")

    runner.write_result_snapshot(output_path, [_StaticProvider({"new": {"value": 1}})])

    assert json.loads(output_path.read_text(encoding="utf-8")) == {"new": {"value": 1}}
    assert sorted(path.name for path in tmp_path.iterdir()) == ["result.json"]


def test_runner_snapshot_preserves_existing_file_when_dump_fails(tmp_path, monkeypatch):
    runner = _load_visual_runner_module()
    output_path = tmp_path / "result.json"
    output_path.write_text('{"old": true}', encoding="utf-8")

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(runner.json, "dump", _boom)

    with pytest.raises(RuntimeError, match="boom"):
        runner.write_result_snapshot(output_path, [_StaticProvider({"new": {"value": 1}})])

    assert json.loads(output_path.read_text(encoding="utf-8")) == {"old": True}
    assert sorted(path.name for path in tmp_path.iterdir()) == ["result.json"]
