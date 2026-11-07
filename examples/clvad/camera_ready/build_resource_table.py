#!/usr/bin/env python3
"""Memory and compute accounting for the InCLAD methods, from the stored result files.

Uses exactly the runs behind the corrected Table 2 (article_model_aggregates_corrected.json)
and the baseline runs selected by build_baseline_table.py. Nothing is re-run.

Per method (continual variant):
  * replay buffer: final number of stored training samples (strategy.replay_buffer.current_size);
  * additional state retained after the last task, float32 estimate from stored shapes:
      PatchCore CL  sum(task bank sizes) x embedding dimension
      PaDiM CL      tasks x H x W x (d + d^2)   (per-location mean and precision)
      CFA CL        sum(task bank sizes) x descriptor dimension
      UCAD          key_prompt_knowledge_bytes (exact)
      ReplayCAD     conditional_storage_bytes + final PatchCore bank (bank size x embedding dimension)
    Replay methods keep no per-task state beyond the buffer; One-for-More stores GPM bases
    whose byte size is not recorded (the total projection rank is reported instead);
  Each column reports the median and interquartile range over runs.
  * training and evaluation time of the continual variant relative to NCL, median over
    pairs matched on dataset, backbone/setting and ordering (whole-run totals from
    time_evaluation_callback; runs on L40 and H100 GPUs are mixed, so ratios, not
    absolute times, are reported).

Inputs: --aggregates is article_model_aggregates_corrected.json written by
rebuild_main_table.py, --baselines is baselines.json written by
build_baseline_table.py (the paper's ReplayCAD row needs the baselines.json
built with the Table 3 inputs listed in that script). Run file paths in both
are mapped onto --results-root (see results_root.py); relative "source"
entries are taken relative to --results-root.

Outputs (in --output-dir): resources.json and resource_table.tex.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from collections import defaultdict

from rebuild_main_table import REPO_ROOT, resolve_source
from results_root import (
    OUTPUT_NAME,
    add_results_root_argument,
    require,
    require_input,
    results_root_from_argv,
)

MIB = 1024 ** 2
METHODS = ["patchcore", "padim", "cfa", "fastflow", "stfpm", "rd4ad", "paste", "efficientad", "ganomaly"]
LABELS = {
    "patchcore": "PatchCore CL", "padim": "PaDiM CL", "cfa": "CFA CL", "fastflow": "FastFlow replay",
    "stfpm": "STFPM replay", "rd4ad": "RD4AD replay", "paste": "PaSTe replay", "efficientad": "EfficientAD replay",
    "ganomaly": "GANomaly replay", "ucad": "UCAD", "one_for_more": "One-for-More", "replaycad": "ReplayCAD",
}
# The paper names One-for-More by its method name, CDAD (Li et al., CVPR 2025).
LABELS["one_for_more"] = "CDAD"
STORED = {
    "patchcore": "task patch banks", "padim": "task Gaussian statistics", "cfa": "task banks, replay",
    "fastflow": "replay buffer", "stfpm": "replay buffer", "rd4ad": "replay buffer", "paste": "replay buffer",
    "efficientad": "replay buffer", "ganomaly": "replay buffer", "ucad": "keys, prompts, features",
    "one_for_more": "GPM projection bases", "replaycad": "conditions, PatchCore bank",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_results_root_argument(parser)
    base = results_root_from_argv(argv) / OUTPUT_NAME
    parser.add_argument("--aggregates", type=pathlib.Path, default=base / "table2" / "article_model_aggregates_corrected.json")
    parser.add_argument("--baselines", type=pathlib.Path, default=base / "baselines" / "baselines.json")
    parser.add_argument("--output-dir", type=pathlib.Path, default=base / "resources")
    return parser.parse_args(argv)


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def state_bytes(method: str, data: dict) -> float | None:
    model = data.get("model") or {}
    strategy = data.get("strategy") or {}
    tasks = model.get("continual_num_tasks") or strategy.get("tasks_seen")
    if method == "patchcore":
        sizes = model.get("continual_task_bank_sizes")
        dim = model.get("target_embed_dimension")
        return sum(sizes) * dim * 4 if sizes and dim else None
    if method == "padim":
        feature_map = model.get("continual_feature_map_size")
        d = model.get("n_features")
        if not (feature_map and d and tasks):
            return None
        h, w = feature_map
        return tasks * h * w * (d + d * d) * 4
    if method == "cfa":
        sizes = model.get("continual_task_bank_sizes")
        shape = model.get("memory_bank_shape")
        if not sizes or not shape:
            return None
        dim = shape[0] if shape[1] == sizes[-1] else shape[1]
        return sum(sizes) * dim * 4
    if method == "ucad":
        return model.get("key_prompt_knowledge_bytes")
    if method == "replaycad":
        memory = strategy.get("replaycad_memory") or {}
        bank = model.get("memory_bank_size", 0) * model.get("target_embed_dimension", 0) * 4
        return memory.get("conditional_storage_bytes", 0) + bank
    return 0.0


def buffer_samples(data: dict) -> int | None:
    buffer = (data.get("strategy") or {}).get("replay_buffer")
    return buffer.get("current_size") if isinstance(buffer, dict) else None


def totals(data: dict) -> tuple[float, float] | None:
    by_concept = (data.get("time_evaluation_callback") or {}).get("time_by_concept")
    if not isinstance(by_concept, dict) or not by_concept:
        return None
    return (sum(v.get("train_time", 0) for v in by_concept.values()),
            sum(v.get("eval_time", 0) for v in by_concept.values()))


def summary(values: list[float]) -> dict | None:
    values = [v for v in values if v is not None]
    if not values:
        return None
    if len(values) == 1:
        return {"n": 1, "median": values[0], "q1": values[0], "q3": values[0], "min": values[0], "max": values[0]}
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return {"n": len(values), "median": statistics.median(values), "q1": q1, "q3": q3, "min": min(values), "max": max(values)}


def main_methods(aggregates: dict, results_root: pathlib.Path | None = None,
                 search_dirs: list[pathlib.Path] | None = None) -> dict:
    runs = defaultdict(dict)  # (method, variant) -> {(dataset, setting, ordering): data}
    for record in aggregates["records"]:
        payload = record.get("image")
        if not record.get("article_usable") or not payload or not payload.get("available"):
            continue
        source = resolve_source(payload["source_files"][0], results_root, search_dirs)
        if source is None:
            source = require(payload["source_files"][0], REPO_ROOT if results_root is None else results_root,
                             search_dirs or [], what="result file listed in --aggregates")
        variant = "continual" if record["strategy"] in ("cl", "replay") else "naive"
        runs[(record["model"], variant)][(record["dataset"], record["setting"], record["ordering"])] = load(source)
    out = {}
    for method in METHODS:
        continual = runs[(method, "continual")]
        naive = runs[(method, "naive")]
        train_ratio, eval_ratio = [], []
        for key, data in continual.items():
            if key in naive:
                c, n = totals(data), totals(naive[key])
                if c and n and n[0] > 0 and n[1] > 0:
                    train_ratio.append(c[0] / n[0])
                    eval_ratio.append(c[1] / n[1])
        out[method] = {
            "runs": len(continual),
            "buffer_samples": summary([buffer_samples(d) for d in continual.values()]),
            "state_mib": summary([(state_bytes(method, d) or 0) / MIB if state_bytes(method, d) is not None else None for d in continual.values()]),
            "train_time_ratio": summary(train_ratio),
            "eval_time_ratio": summary(eval_ratio),
        }
    return out


def baseline_methods(baselines: list[dict], results_root: pathlib.Path | None = None,
                     search_dirs: list[pathlib.Path] | None = None) -> dict:
    root = REPO_ROOT if results_root is None else results_root
    out = {}
    for method in ("ucad", "one_for_more", "replaycad"):
        # Only completed runs: state grows with the number of tasks.
        # files = [REPO_ROOT / r["source"] for r in baselines if r["method"] == method and r["complete"]]
        files = [require(r["source"], root, search_dirs or [], what="baseline run listed in --baselines")
                 for r in baselines if r["method"] == method and r["complete"]]
        datas = [load(f) for f in files]
        entry = {"runs": len(datas), "buffer_samples": None, "train_time_ratio": None, "eval_time_ratio": None}
        if method == "one_for_more":
            entry["state_mib"] = None
            entry["projection_total_rank"] = summary([(d.get("model") or {}).get("projection_total_rank") for d in datas])
        else:
            entry["state_mib"] = summary([state_bytes(method, d) / MIB for d in datas])
        if method == "replaycad":
            entry["generated_samples"] = summary([((d.get("strategy") or {}).get("replaycad_memory") or {}).get("generated_samples_total") for d in datas])
        out[method] = entry
    return out


def rng(s: dict | None, fmt: str) -> str:
    if not s:
        return "--"
    if abs(s["q3"] - s["q1"]) < 1e-9:
        return fmt.format(s["median"])
    return f"{fmt.format(s['median'])} [{fmt.format(s['q1'])}--{fmt.format(s['q3'])}]"


def mib(s: dict | None) -> str:
    if not s:
        return "--"
    def one(v: float) -> str:
        return f"{v:,.1f}".replace(",", "{,}") if v < 100 else f"{v:,.0f}".replace(",", "{,}")
    if s["median"] == 0 and s["max"] == 0:
        return "0"
    if abs(s["q3"] - s["q1"]) < 1e-9:
        return one(s["median"])
    return f"{one(s['median'])} [{one(s['q1'])}--{one(s['q3'])}]"


def render(rows: dict) -> str:
    lines = ["% Auto-generated by examples/clvad/camera_ready/build_resource_table.py",
             "% Median [interquartile range] over the runs behind Table 2 (and the completed baseline runs).", ""]
    for method in METHODS + ["ucad", "one_for_more", "replaycad"]:
        r = rows[method]
        stored = STORED[method]
        buffer = rng(r["buffer_samples"], "{:.0f}") if r["buffer_samples"] else "0"
        lines.append(" & ".join([
            LABELS[method], stored, buffer, mib(r["state_mib"]),
            rng(r["train_time_ratio"], "{:.2f}"), rng(r["eval_time_ratio"], "{:.2f}"),
        ]) + " \\\\")
        if method == "ganomaly":
            lines.append("\\midrule")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    require_input(args.aggregates, "--aggregates")
    require_input(args.baselines, "--baselines")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rows = main_methods(load(args.aggregates), args.results_root, args.search_dir)
    rows.update(baseline_methods(load(args.baselines), args.results_root, args.search_dir))
    (out / "resources.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    (out / "resource_table.tex").write_text(render(rows), encoding="utf-8")
    for method, r in rows.items():
        print(method, json.dumps({k: v for k, v in r.items()}, default=str)[:300])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
