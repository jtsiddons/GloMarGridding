import pytest
import numpy as np

from glomar_gridding.covariance_tools import (
    csum_up_to_val,
    eigenvalue_clip,
    perturb_cov_to_positive_definite,
    simple_clipping,
)


def test_eigenvalue_clip() -> None:
    # TEST: That trace is maintained, top eigenvalues (explaining 95% variance
    #       are unchanged)
    np.random.seed(90210)
    n = 10
    n_top = 6
    n_bot = 4
    explained = 0.95
    top_evals = 13 - 11 * np.random.rand(n_top)
    sum_top = np.sum(top_evals)
    total_sum = sum_top / explained
    remaining = total_sum - sum_top

    bot_evals = np.random.rand(n_bot)
    sum_bot = np.sum(bot_evals)
    adjustment = remaining / sum_bot
    bot_evals *= adjustment

    evals = np.sort(np.concatenate([top_evals, bot_evals]))

    # Create an initial covariance matrix
    A = np.random.rand(n, n)
    S = np.dot(A, A.T)

    # Use calculated eigenvalues with the eigenvectors of S to compute a new
    # Covariance
    _, evecs = np.linalg.eigh(S)

    S_new = evecs @ np.diag(evals) @ evecs.T

    assert np.allclose(np.linalg.eigvalsh(S_new), evals)

    # "Fix" the new Covariance
    S_fixed = eigenvalue_clip(S_new)
    evals_fixed = np.linalg.eigvalsh(S_fixed)

    # TEST: trace is not changed
    assert np.allclose(np.trace(S_fixed), np.trace(S_new))
    # TEST: top eigenvalues are not modified
    assert np.allclose(evals_fixed[-n_top:], evals[-n_top:])
    return None


def test_eigenvalue_clip_negative_evals() -> None:
    # TEST: That negative eigenvalues are now positive
    np.random.seed(90210)
    n = 10
    n_neg = 3

    # Create an initial covariance matrix
    A = np.random.rand(n, n)
    S = np.dot(A, A.T)

    evals, evecs = np.linalg.eigh(S)
    # Adjust the eigenvalues to ensure negative values
    evals[:n_neg] -= np.sum(evals[:n_neg])

    assert np.sum(evals < 0) == 3

    S_new = evecs @ np.diag(evals) @ evecs.T

    # "Fix" the new Covariance
    S_fixed = eigenvalue_clip(S_new)
    evals_fixed = np.linalg.eigvalsh(S_fixed)

    # TEST: trace is not changed
    assert np.allclose(np.trace(S_fixed), np.trace(S_new))
    # TEST: all eigenvalues are positive
    assert np.all(evals_fixed > 0)
    return None


def test_simple_clip():
    # np.random.seed(90210)
    n = 10
    n_neg = 3

    # Create an initial covariance matrix
    A = np.random.rand(n, n)
    S = np.dot(A, A.T)

    evals, evecs = np.linalg.eigh(S)
    # Adjust the eigenvalues to ensure negative values
    evals[:n_neg] -= np.sum(evals[:n_neg])

    assert np.sum(evals < 0) == n_neg

    S_new = evecs @ np.diag(evals) @ evecs.T

    # "Fix" the new Covariance
    S_fixed, _ = simple_clipping(S_new)
    evals_fixed = np.linalg.eigvalsh(S_fixed)

    # TEST: all eigenvalues are positive
    assert np.all(evals_fixed > 0)
    return None


def test_perturb_pos_def():
    # np.random.seed(90210)
    n = 10
    n_neg = 3

    # Create an initial covariance matrix
    A = np.random.rand(n, n)
    S = np.dot(A, A.T)

    evals, evecs = np.linalg.eigh(S)
    # Adjust the eigenvalues to ensure negative values
    evals[:n_neg] -= np.sum(evals[:n_neg])

    assert np.sum(evals < 0) == n_neg

    S_new = evecs @ np.diag(evals) @ evecs.T

    # "Fix" the new Covariance
    S_fixed = perturb_cov_to_positive_definite(S_new)
    evals_fixed = np.linalg.eigvalsh(S_fixed)

    # TEST: all eigenvalues are positive
    assert np.all(evals_fixed > 0)
    return None


@pytest.mark.parametrize(
    "name, n, expected_i",
    [
        ("n = 10, i = 7", 10, 7),
        ("n = 25, i = 24", 25, 24),
        ("n = 100, i = 100", 100, 100),
        ("n = 15, i = 1", 15, 1),
    ],
)
def test_cumsum(name, n, expected_i):
    vals = np.arange(n + 1)
    target_sum = expected_i * (expected_i + 1) / 2

    # Cumulatively sum to just under expected value
    csum, i = csum_up_to_val(vals, target_sum - 1, reverse=False)

    assert expected_i == i - 1
    assert target_sum == csum
