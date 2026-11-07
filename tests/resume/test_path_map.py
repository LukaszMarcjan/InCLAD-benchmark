from pyclad.resume.fingerprint import assert_fingerprint_matches
from pyclad.resume.path_map import ENV_VAR, map_path, map_values


def test_without_variable_paths_are_unchanged(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert map_path("/home/user-hpc/a.json") == "/home/user-hpc/a.json"
    assert map_values({"x": "/home/user-hpc/a", "y": 1}) == {"x": "/home/user-hpc/a", "y": 1}


def test_prefix_is_rewritten_only_at_a_path_boundary(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "/home/user-hpc=/home/user")
    assert map_path("/home/user-hpc/resumable-results/r.json") == "/home/user/resumable-results/r.json"
    assert map_path("/home/user-hpc") == "/home/user"
    assert map_path("/home/user-hpc2/r.json") == "/home/user-hpc2/r.json"
    assert map_path(42) == 42


def test_mapped_fingerprint_matches_relocated_artifact_dir(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "/home/user-hpc=/home/user")
    recorded = {"replaycad.artifact_dir": "/home/user-hpc/replaycad-artifacts/s", "replaycad.seed": 42}
    current = {"replaycad.artifact_dir": "/home/user/replaycad-artifacts/s", "replaycad.seed": 42}
    assert_fingerprint_matches(map_values(recorded), current)
