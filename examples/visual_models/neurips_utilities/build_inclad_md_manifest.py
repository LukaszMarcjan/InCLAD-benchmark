"""
Build InCLAD-MD manifests for upload to HuggingFace.

Reads the selected_concepts CSV and ordering CSVs from multidataset_ordering/,
pulls matching rows from the per-benchmark InCLAD-Bench manifests
(<benchmark>_easy_to_hard_samples.csv), and writes two ordering splits:

    output_dir/
        easy_to_hard.csv          (inclad-md_easy_to_hard.csv with --hf-filenames)
        hard_to_easy.csv          (inclad-md_hard_to_easy.csv with --hf-filenames)

The published files are config ``inclad-md`` of lifelonglab/inclad-benchmark,
splits ``easy_to_hard`` and ``hard_to_easy``.

The per-benchmark manifests are not shipped in this repository. Pass
``--manifests-dir`` pointing at a folder that holds them (for example the
HuggingFace snapshot of lifelonglab/inclad-benchmark), or ``--from-hf`` to
download them with huggingface_hub.

Seed and license columns:
    --seed-mode run        (default) ordering_master_seed=42 and
                           ordering_seed=191664963 for both orderings, the seed
                           every InCLAD-MD paper run used.
    --seed-mode published  reproduces the published CSVs: hard_to_easy carries
                           ordering_seed=1662057957 (second derived seed). The
                           column is informational; the runs used 191664963.
    --seed-mode empty      leaves both seed columns empty (original behaviour).
    BTech rows get source_license "CC BY-SA 4.0"; ``--keep-source-license``
    copies the value from the source manifest instead (the published CSVs say
    "CC BY-SA").

``--seed-mode published --keep-source-license --hf-filenames`` with the HF
snapshot as --manifests-dir rebuilds the published inclad-md CSVs byte for byte.

Usage (from the repository root)
--------------------------------
python examples/visual_models/neurips_utilities/build_inclad_md_manifest.py \\
    --from-hf --output-dir /tmp/inclad_md_hf --hf-filenames

python examples/visual_models/neurips_utilities/build_inclad_md_manifest.py \\
    --manifests-dir /path/to/inclad-benchmark/snapshot \\
    --ordering-dir  examples/visual_models/neurips_utilities/multidataset_ordering \\
    --output-dir    /tmp/inclad_md_hf
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
# The script lives in examples/visual_models/neurips_utilities/, so the
# repository root is parents[2]; parents[1] is examples/.
# PROJECT_ROOT = SCRIPT_DIR.parents[1]
PROJECT_ROOT = SCRIPT_DIR.parents[2]

DEFAULT_MANIFESTS_DIR = (
    PROJECT_ROOT / "src" / "pyclad" / "data" / "datasets" / "visual_datasets" / "manifests"
)
DEFAULT_ORDERING_DIR = SCRIPT_DIR / "multidataset_ordering"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "inclad_md_manifests"

MANIFEST_FIELDNAMES = (
    "sample_id",
    "source_dataset",
    "source_category",
    "category",
    "category_order",
    "split",
    "image_relpath",
    "mask_relpath",
    "image_label",
    "defect_type",
    "ordering_name",
    "ordering_master_seed",
    "ordering_seed",
    "source_homepage",
    "source_license",
)

BENCHMARK_MANIFEST_SUFFIX = "_easy_to_hard_samples.csv"

HF_REPO_ID = "lifelonglab/inclad-benchmark"
MD_SOURCE_BENCHMARKS = ("btech", "dagm", "mpdd", "mvtec", "visa")
DEFAULT_MASTER_SEED = 42
SEED_MODES = ("run", "published", "empty")
ORDERING_RUN_INDEX = {"easy_to_hard": 0, "hard_to_easy": 1}

# License strings as stated in the paper; the source manifests drop the
# version for BTech ("CC BY-SA").
SOURCE_LICENSE_OVERRIDES = {"btech": "CC BY-SA 4.0"}

LINE_TERMINATORS = {"lf": "\n", "crlf": "\r\n"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build InCLAD-MD manifests from inclad-bench per-benchmark CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifests-dir",
        type=str,
        default=str(DEFAULT_MANIFESTS_DIR),
        help="Directory with per-benchmark <benchmark>_easy_to_hard_samples.csv files "
        "(not shipped; e.g. the HuggingFace snapshot directory)",
    )
    parser.add_argument(
        "--from-hf",
        action="store_true",
        help=f"Download the per-benchmark manifests from {HF_REPO_ID} instead of reading --manifests-dir",
    )
    parser.add_argument("--hf-repo-id", type=str, default=HF_REPO_ID, help="HuggingFace dataset repo for --from-hf")
    parser.add_argument(
        "--hf-revision", type=str, default=None, help="Optional HuggingFace revision (commit/tag) for --from-hf"
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        default=DEFAULT_MASTER_SEED,
        help="Master seed written to ordering_master_seed and used to derive ordering_seed",
    )
    parser.add_argument(
        "--seed-mode",
        type=str,
        choices=SEED_MODES,
        default="run",
        help="run: ordering_seed = first derived seed for both orderings (the seed of the paper runs); "
        "published: second derived seed for hard_to_easy, as in the published CSVs; empty: blank columns",
    )
    parser.add_argument(
        "--keep-source-license",
        action="store_true",
        help="Copy source_license verbatim from the source manifests (published CSVs say 'CC BY-SA' for BTech)",
    )
    parser.add_argument(
        "--hf-filenames",
        action="store_true",
        help="Name the outputs inclad-md_easy_to_hard.csv / inclad-md_hard_to_easy.csv as in the HF repo",
    )
    parser.add_argument(
        "--line-terminator",
        type=str,
        choices=sorted(LINE_TERMINATORS),
        default="lf",
        help="Line ending of the written CSVs (the published files use LF)",
    )
    parser.add_argument(
        "--ordering-dir",
        type=str,
        default=str(DEFAULT_ORDERING_DIR),
        help="Directory with multidataset_ordering CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for HuggingFace-ready CSVs",
    )
    return parser.parse_args()


def load_ordering_csv(path: Path, concept_column: str = "concept") -> list[str]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    order = [row[concept_column].strip() for row in rows if row.get(concept_column, "").strip()]
    if not order:
        raise ValueError(f"No concepts found in {path} column '{concept_column}'")
    return order


def load_selected_concepts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_benchmark_manifest(
    manifests_dir: Path, benchmark: str
) -> list[dict[str, str]]:
    path = manifests_dir / f"{benchmark}{BENCHMARK_MANIFEST_SUFFIX}"
    if not path.exists():
        raise FileNotFoundError(
            f"Per-benchmark manifest not found: {path}\n"
            f"Generate it first with: python examples/visual_models/neurips_utilities/export_visual_benchmark_manifests.py"
        )
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def derive_seed(master_seed: int, run_index: int) -> int:
    """Same derivation as pyclad.data.readers.visual_benchmark_manifest.derive_per_run_seed."""
    import numpy as np

    rng = np.random.default_rng(int(master_seed))
    return int(rng.integers(0, 2**31 - 1, size=run_index + 1)[run_index])


def seed_columns(ordering_name: str, master_seed: int, seed_mode: str) -> tuple[str, str]:
    if seed_mode == "empty":
        return "", ""
    run_index = ORDERING_RUN_INDEX[ordering_name] if seed_mode == "published" else 0
    return str(master_seed), str(derive_seed(master_seed, run_index))


def download_hf_manifests(repo_id: str, revision: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    paths = [
        Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=f"{benchmark}{BENCHMARK_MANIFEST_SUFFIX}",
                repo_type="dataset",
                revision=revision,
            )
        )
        for benchmark in MD_SOURCE_BENCHMARKS
    ]
    return paths[0].parent


def build_rows(
    selected_concepts: list[dict[str, str]],
    category_order: list[str],
    manifests_dir: Path,
    ordering_name: str,
    master_seed: int = DEFAULT_MASTER_SEED,
    seed_mode: str = "empty",
    keep_source_license: bool = True,
) -> list[dict[str, str]]:
    # Build lookup: (source_dataset, source_category) → benchmark_category (prefixed)
    concept_map: dict[tuple[str, str], str] = {}
    for row in selected_concepts:
        dataset = row["dataset"].strip().lower()
        src_cat = row["source_concept"].strip()
        bench_cat = row["benchmark_concept"].strip()
        concept_map[(dataset, src_cat)] = bench_cat

    category_positions = {cat: i + 1 for i, cat in enumerate(category_order)}

    # Cache per-benchmark manifests
    manifest_cache: dict[str, list[dict[str, str]]] = {}

    rows: list[dict[str, str]] = []
    sample_index = 0
    master_seed_value, ordering_seed_value = seed_columns(ordering_name, master_seed, seed_mode)

    for category in category_order:
        # Find which (dataset, source_category) maps to this category
        match = next(
            ((ds, sc) for (ds, sc), bc in concept_map.items() if bc == category), None
        )
        if match is None:
            raise ValueError(
                f"Category '{category}' from ordering not found in selected_concepts.csv"
            )
        source_dataset, source_category = match

        if source_dataset not in manifest_cache:
            manifest_cache[source_dataset] = load_benchmark_manifest(manifests_dir, source_dataset)

        source_rows = [
            r for r in manifest_cache[source_dataset]
            if r["category"] == source_category
        ]
        if not source_rows:
            raise ValueError(
                f"No rows found for {source_dataset}/{source_category} in manifest"
            )

        for src_row in source_rows:
            rows.append(
                {
                    "sample_id": f"inclad_md:{sample_index:06d}",
                    "source_dataset": source_dataset,
                    "source_category": source_category,
                    "category": category,
                    "category_order": str(category_positions[category]),
                    "split": src_row["split"],
                    "image_relpath": src_row["image_relpath"],
                    "mask_relpath": src_row.get("mask_relpath", ""),
                    "image_label": src_row["image_label"],
                    "defect_type": src_row.get("defect_type", ""),
                    "ordering_name": ordering_name,
                    # "ordering_master_seed": "",
                    # "ordering_seed": "",
                    "ordering_master_seed": master_seed_value,
                    "ordering_seed": ordering_seed_value,
                    "source_homepage": src_row.get("source_homepage", ""),
                    # "source_license": src_row.get("source_license", ""),
                    "source_license": (
                        src_row.get("source_license", "")
                        if keep_source_license
                        else SOURCE_LICENSE_OVERRIDES.get(source_dataset, src_row.get("source_license", ""))
                    ),
                }
            )
            sample_index += 1

    return rows


def write_csv(path: Path, rows: list[dict[str, str]], line_terminator: str = "\r\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDNAMES, lineterminator=line_terminator)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {len(rows)} rows → {path}")


def main() -> None:
    args = parse_args()
    if args.from_hf:
        manifests_dir = download_hf_manifests(args.hf_repo_id, args.hf_revision)
        print(f"Per-benchmark manifests from {args.hf_repo_id}: {manifests_dir}")
    else:
        manifests_dir = Path(args.manifests_dir).expanduser().resolve()
        if not manifests_dir.is_dir():
            raise SystemExit(
                f"Manifests directory does not exist: {manifests_dir}\n"
                "Pass --manifests-dir <dir with <benchmark>_easy_to_hard_samples.csv> "
                f"(e.g. a snapshot of {HF_REPO_ID}) or use --from-hf."
            )
    ordering_dir = Path(args.ordering_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    line_terminator = LINE_TERMINATORS[args.line_terminator]

    def output_name(ordering_name: str) -> str:
        return f"inclad-md_{ordering_name}.csv" if args.hf_filenames else f"{ordering_name}.csv"

    selected_path = ordering_dir / "multidataset_seq_selected_concepts.csv"
    e2h_path = ordering_dir / "multidataset_seq_easy_to_hard_blocks_roc-auc.csv"
    h2e_path = ordering_dir / "multidataset_seq_hard_to_easy_blocks_roc-auc.csv"

    selected_concepts = load_selected_concepts(selected_path)
    e2h_order = load_ordering_csv(e2h_path)
    h2e_order = load_ordering_csv(h2e_path)

    print(f"Selected concepts: {len(selected_concepts)}")
    print(f"easy_to_hard order ({len(e2h_order)} categories): {e2h_order}")
    print(f"hard_to_easy order ({len(h2e_order)} categories): {h2e_order}")
    print()

    print("Building easy_to_hard...")
    e2h_rows = build_rows(
        selected_concepts=selected_concepts,
        category_order=e2h_order,
        manifests_dir=manifests_dir,
        ordering_name="easy_to_hard",
        master_seed=args.master_seed,
        seed_mode=args.seed_mode,
        keep_source_license=args.keep_source_license,
    )
    # write_csv(output_dir / "easy_to_hard.csv", e2h_rows)
    write_csv(output_dir / output_name("easy_to_hard"), e2h_rows, line_terminator)

    print("Building hard_to_easy...")
    h2e_rows = build_rows(
        selected_concepts=selected_concepts,
        category_order=h2e_order,
        manifests_dir=manifests_dir,
        ordering_name="hard_to_easy",
        master_seed=args.master_seed,
        seed_mode=args.seed_mode,
        keep_source_license=args.keep_source_license,
    )
    # write_csv(output_dir / "hard_to_easy.csv", h2e_rows)
    write_csv(output_dir / output_name("hard_to_easy"), h2e_rows, line_terminator)

    print()
    print(f"Done. Upload the CSVs from {output_dir} to HuggingFace.")
    print("See docs/inclad-md-dataset.md for upload instructions.")


if __name__ == "__main__":
    main()
