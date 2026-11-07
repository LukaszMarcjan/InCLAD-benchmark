# Multidataset Sequential Benchmark v1

This directory defines a block-wise multidataset curriculum derived from the STE-based curricula in `examples/visual_models/ordering/`.

## Benchmark Idea

- Selected concepts:
  - `2 easiest + 2 median + 2 hardest` per dataset when available
  - datasets with fewer than 6 concepts use all available concepts
- Block order for `easy_to_hard`:
  1. `btech`
  2. `mpdd`
  3. `dagm`
  4. `visa`
  5. `mvtec`
- Each block is internally ordered `easy_to_hard` using the existing STE ranking.
- `hard_to_easy` is the exact reverse at both levels:
  - reverse dataset-block order
  - reverse concept order inside each block

The dataset-block order above was chosen from the STE ranking, using mean normalized rank across the selected concepts so datasets with different numbers of categories can still be compared fairly.

## Files

- `multidataset_seq_easy_to_hard_blocks_roc-auc.csv`
- `multidataset_seq_hard_to_easy_blocks_roc-auc.csv`
- `multidataset_seq_selected_concepts.csv`

The loader only needs the `concept` column from the ordering CSVs. The remaining columns are metadata.

## Category Naming

Use category names exactly as listed in the ordering files:

- `btech__03`
- `mpdd__connector`
- `dagm__Class4`
- `visa__pcb4`
- `mvtec__bottle`

In other words, the synthetic benchmark should prefix each source concept with its dataset:

`<dataset>__<concept>`

This avoids collisions and makes the source of each concept explicit.

## Recommended Loader Setup

This benchmark is not a predefined built-in visual benchmark, so the easiest path is to build it as a CSV benchmark and load it with `CsvBenchmarkSpec`.

Minimal split-file requirements:

- `category`: must match the `concept` values from the ordering CSV
- `split`: `train` or `test`
- `image`: path relative to the synthetic benchmark root
- `label`: `normal` for non-anomalous samples, any other label for anomalous samples
- `mask`: optional relative mask path
- `defect_type`: optional string

Example:

```python
from pyclad.data.readers.visual_benchmark_reader import CsvBenchmarkSpec, read_visual_benchmark_dataset
from pyclad.data.readers.visual_dataset_ordering import load_concept_order_from_file

spec = CsvBenchmarkSpec(
    name="multidataset_seq",
    csv_path="multidataset_seq.csv",
    category_column="category",
    split_column="split",
    train_split_value="train",
    test_split_value="test",
    image_column="image",
    label_column="label",
    normal_label_value="normal",
    mask_column="mask",
    defect_type_column="defect_type",
)

concept_order = load_concept_order_from_file(
    "examples/visual_models/neurips_utilities/multidataset_ordering/multidataset_seq_easy_to_hard_blocks_roc-auc.csv",
    ordering_column="concept",
)

dataset = read_visual_benchmark_dataset(
    root="/path/to/multidataset_seq_root",
    benchmark=spec,
    dataset_name="MULTIDATASET-SEQ",
    categories=concept_order,
    data_mode="numpy",
    color_mode="rgb",
)
```

For the reverse curriculum, switch the ordering path to:

`multidataset_seq_hard_to_easy_blocks_roc-auc.csv`

## Relation to the paper runs and to InCLAD-MD

The paper's InCLAD-MD rows were run on this CSV benchmark under the name
`multidataset_seq_v1`, with the two ordering files above (the runs refer to them
as `multidataset_seq_v1_easy_to_hard_blocks_roc-auc.csv` and
`multidataset_seq_v1_hard_to_easy_blocks_roc-auc.csv`; the content is the same).
`../build_multidataset_seq.py --benchmark-name multidataset_seq_v1` rebuilds
the CSV. The released loader `InCLADMDDataset` (HuggingFace config `inclad-md`,
runner flag `--benchmark inclad-md`) gives the same 27-concept `easy_to_hard`
and `hard_to_easy` orders; see `docs/inclad-md-dataset.md`.

## Practical Note

If you later want this benchmark to work through the higher-level registered loader by benchmark name alone, you will need to add a dedicated reader or alias in `visual_benchmark_reader.py`. For now, `CsvBenchmarkSpec` is the cleanest option and requires no framework changes. The runners accept `multidataset_seq_v1` as an alias only together with `--inclad-bench`; the alias loads `InCLADMDDataset` from the Hugging Face manifest (same 27 tasks and orders), not this CSV.
