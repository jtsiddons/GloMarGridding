import numpy as np
import pytest

from glomar_gridding.covariance_tools import (
    csum_up_to_val,
    eigenvalue_clip,
    explained_variance_clip,
    laloux_clip,
    perturb_cov_to_positive_definite,
    simple_clipping,
    validate_covariance,
)
from glomar_gridding.stochastic import draw_from_cov, _fallback_draw_from_cov


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

    assert np.allclose(S_fixed, S_fixed.T)

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

    assert np.allclose(S_fixed, S_fixed.T)

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
    new_evals = np.linalg.eigvalsh(S_new)
    assert np.min(new_evals) < 0
    assert np.sum(new_evals < 0) == n_neg

    # "Fix" the new Covariance
    S_fixed = perturb_cov_to_positive_definite(S_new)
    assert np.allclose(S_fixed, S_fixed.T)
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


def test_cumsum_exceeds_target():
    vals = np.random.rand(50)
    vals = np.sort(vals)
    target = 0.9
    total = vals.sum()
    target_csum = target * total

    csum, _ = csum_up_to_val(vals, target_csum, reverse=True)
    assert csum >= target_csum


def test_explained_clip():
    np.random.seed(314159)
    A = np.random.rand(5, 5)
    S = np.dot(A, A.T)
    _, evecs = np.linalg.eigh(S)
    new_evals = np.array([-3.0, 2.0, 4.2, 5.5, 5.8])

    target_explained_variance_frac = 0.9
    total_variance = np.sum(new_evals)
    tar = target_explained_variance_frac * total_variance
    csum, i = csum_up_to_val(new_evals, tar)
    assert csum > total_variance

    valid_target = np.sum(new_evals[i + 1 :]) / total_variance

    S_new = evecs @ np.diag(new_evals) @ evecs.T

    # TEST: Fails with target as explained variance exceeds total
    with pytest.raises(ValueError, match=f"{valid_target:.2f}"):
        explained_variance_clip(S_new, target_explained_variance_frac)

    # TEST: Now works with valid target
    out = explained_variance_clip(S_new, valid_target * 0.99)

    assert out.shape == S.shape
    assert np.allclose(out, out.T)
    assert (np.linalg.eigvalsh(out) > 0).all()

    return None


def test_laloux_clip():
    np.random.seed(2718)
    n = 100
    A = np.random.rand(n, n)
    S = np.dot(A, A.T)
    evals, evecs = np.linalg.eigh(S)

    evals -= np.sum(evals[:5])

    S_new = evecs @ np.diag(evals) @ evecs.T
    new_evals = np.linalg.eigvalsh(S_new)
    assert (new_evals < 0).any()
    assert (new_evals > 0).any()

    out = laloux_clip(S_new, num_time_pts=20)

    assert out.shape == S.shape
    assert np.allclose(out, out.T)
    assert (np.linalg.eigvalsh(out) > 0).all()

    assert True


def test_cov_validaton():
    n = 15
    A = np.random.rand(n, n)
    # Symmetric
    S = np.dot(A, A.T)

    assert np.all(S == validate_covariance(S))

    S[1, 3] += 1e-6

    res = validate_covariance(S)
    assert not np.all(S == res)
    assert np.array_equal(res, res.T)

    S[1, 3] += 1e-3

    try:
        validate_covariance(S)
    except ValueError as e:
        assert e.args[0] == "cov is not symmetric."
    except Exception as e:
        raise e

    try:
        validate_covariance(np.array([1, 2, 3]))
    except ValueError as e:
        assert e.args[0] == "cov should be 2D."
    except Exception as e:
        raise e

    try:
        validate_covariance(np.array([[1, 2, 3], [4, 5, 6]]))
    except ValueError as e:
        assert e.args[0] == "cov is not a square matrix"
    except Exception as e:
        raise e


def test_draw_from_cov():
    n = 15
    A = np.random.rand(n, n)
    S = np.dot(A, A.T)
    loc = np.random.rand(n)

    ndraws = 6
    draws = draw_from_cov(loc, S, ndraws=ndraws)
    assert draws.shape == (ndraws, n)

    draw = draw_from_cov(loc, S, ndraws=1)
    assert draw.shape == (n,)


def test_draw_from_cov_fallback():
    n = 15
    A = np.random.rand(n, n)
    S = np.dot(A, A.T)
    loc = np.random.rand(n)

    ndraws = 6
    draws = _fallback_draw_from_cov(loc, S, ndraws=ndraws)
    assert draws.shape == (ndraws, n)

    draw = _fallback_draw_from_cov(loc, S, ndraws=1)
    assert draw.shape == (n,)
