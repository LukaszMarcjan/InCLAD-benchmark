"""The CR block-wise reduction must be numerically indistinguishable from the original."""

import pytest
import torch

from pyclad.models.patchcore.cr_chunked_reduce import (
    DEFAULT_CHUNK_ROWS,
    disable,
    enable,
    is_enabled,
)
from pyclad.models.patchcore.patchcore import ApproximateGreedyCoresetSampler


@pytest.fixture(autouse=True)
def _restore_original():
    disable()
    yield
    disable()


def _sampler(**kwargs):
    return ApproximateGreedyCoresetSampler(
        percentage=0.1,
        device=torch.device("cpu"),
        number_of_starting_points=10,
        dimension_to_project_features_to=128,
        random_seed=0,
        **kwargs,
    )


@pytest.mark.parametrize("rows,chunk", [(1000, 128), (5000, 997), (4096, 4096), (4097, 4096)])
def test_chunked_matches_original_numerically(rows, chunk):
    """Equal to within float32 noise -- see test_bit_equality_is_not_guaranteed."""
    features = torch.randn(rows, 1024, generator=torch.Generator().manual_seed(7))

    expected = _sampler()._reduce_features(features)
    enable(chunk_rows=chunk)
    actual = _sampler()._reduce_features(features)

    assert actual.shape == expected.shape
    diff = (actual - expected).abs().max().item()
    assert diff < 1e-5, f"max abs diff {diff:.3e} at rows={rows}, chunk={chunk}"


def test_bit_equality_is_not_guaranteed():
    """Pins the known limit so nobody later claims this patch is exact.

    BLAS picks its kernel partly from the row count, so a chunked call is not
    obliged to reproduce the single-call result bit for bit. Measured on CPU
    float32: equal for (1000, 128), (5000, 997) and the unchunked path, but
    ~1.4e-06 apart for 4097 rows in blocks of 4096 -- a 1-row tail block.

    1e-06 is far below any anomaly-score difference that matters, but PatchCore's
    greedy coreset selects by argmax over distances, so a perturbation that small
    can still flip a near-tie and change which patch enters the bank. Verify a
    real run end to end (BTech is ~6 h) before trusting this on the headline
    numbers, and describe it as "equivalent to within float32 noise" rather than
    "identical" in the paper.
    """
    features = torch.randn(4097, 1024, generator=torch.Generator().manual_seed(7))
    expected = _sampler()._reduce_features(features)
    enable(chunk_rows=4096)
    actual = _sampler()._reduce_features(features)

    assert not torch.equal(actual, expected)
    assert (actual - expected).abs().max().item() < 1e-5


def test_projection_shortcut_is_untouched():
    """features already at the target width take the original early-out."""
    features = torch.randn(300, 128)
    expected = _sampler()._reduce_features(features)
    enable(chunk_rows=16)
    actual = _sampler()._reduce_features(features)
    assert torch.equal(actual, expected)


def test_enable_disable_round_trip():
    original = ApproximateGreedyCoresetSampler._reduce_features
    assert not is_enabled()

    enable()
    assert is_enabled()
    assert ApproximateGreedyCoresetSampler._reduce_features is not original
    assert ApproximateGreedyCoresetSampler._cr_chunk_rows == DEFAULT_CHUNK_ROWS

    enable(chunk_rows=1024)  # idempotent, only the block size changes
    assert ApproximateGreedyCoresetSampler._cr_chunk_rows == 1024

    disable()
    assert not is_enabled()
    assert ApproximateGreedyCoresetSampler._reduce_features is original
    assert not hasattr(ApproximateGreedyCoresetSampler, "_cr_chunk_rows")


def test_rejects_non_positive_chunk():
    with pytest.raises(ValueError):
        enable(chunk_rows=0)
