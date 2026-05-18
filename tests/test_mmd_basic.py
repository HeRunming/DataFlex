"""
Tests for MMD Selector core algorithms.

Tests kernel correctness, median heuristic, and exact marginal greedy selection
using the actual method names from the MMDSelector implementation.
"""

import numpy as np
import pytest

from dataflex.train.selector.mmd_selector import MMDSelector


def _make_selector():
    """Create an MMDSelector instance without needing real accelerator/dataset.

    Uses __new__ to bypass __init__ so we can test pure math methods.
    """
    sel = MMDSelector.__new__(MMDSelector)
    return sel


class TestKernels:
    """Test kernel matrix computations."""

    def test_rbf_self_similarity_is_one(self):
        """k(x, x) = 1 for RBF kernel."""
        X = np.random.randn(10, 5)
        sigma = 1.0
        K = MMDSelector._rbf_kernel_matrix(X, X, sigma)
        # Diagonal should be 1.0 (self-similarity)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-10)

    def test_rbf_symmetry(self):
        """K(X, Y) = K(Y, X).T"""
        rng = np.random.RandomState(42)
        X = rng.randn(7, 4)
        Y = rng.randn(5, 4)
        sigma = 2.0
        K_xy = MMDSelector._rbf_kernel_matrix(X, Y, sigma)
        K_yx = MMDSelector._rbf_kernel_matrix(Y, X, sigma)
        np.testing.assert_allclose(K_xy, K_yx.T, atol=1e-12)

    def test_rbf_values_in_zero_one(self):
        """RBF kernel values are in (0, 1]."""
        rng = np.random.RandomState(123)
        X = rng.randn(20, 8)
        Y = rng.randn(15, 8)
        sigma = 1.5
        K = MMDSelector._rbf_kernel_matrix(X, Y, sigma)
        assert np.all(K > 0), "RBF kernel values should be strictly positive"
        assert np.all(K <= 1.0 + 1e-12), "RBF kernel values should be at most 1"

    def test_rbf_decreases_with_distance(self):
        """RBF kernel value decreases as distance increases."""
        x = np.array([[0.0, 0.0]])
        y_close = np.array([[0.1, 0.0]])
        y_far = np.array([[10.0, 0.0]])
        sigma = 1.0
        k_close = MMDSelector._rbf_kernel_matrix(x, y_close, sigma)[0, 0]
        k_far = MMDSelector._rbf_kernel_matrix(x, y_far, sigma)[0, 0]
        assert k_close > k_far

    def test_grad_cov_is_squared_inner_product(self):
        """k(x, y) = <x, y>^2 for grad_cov kernel."""
        rng = np.random.RandomState(7)
        X = rng.randn(6, 3)
        Y = rng.randn(4, 3)
        K = MMDSelector._grad_cov_kernel_matrix(X, Y)
        expected = (X @ Y.T) ** 2
        np.testing.assert_allclose(K, expected, atol=1e-12)

    def test_grad_cov_sign_invariant(self):
        """k(g, -g) should equal k(g, g) since it's squared."""
        rng = np.random.RandomState(99)
        g = rng.randn(1, 5)
        neg_g = -g
        K_same = MMDSelector._grad_cov_kernel_matrix(g, g)
        K_neg = MMDSelector._grad_cov_kernel_matrix(g, neg_g)
        np.testing.assert_allclose(K_same, K_neg, atol=1e-12)

    def test_grad_cov_non_negative(self):
        """Squared inner product is always non-negative."""
        rng = np.random.RandomState(0)
        X = rng.randn(10, 6)
        Y = rng.randn(8, 6)
        K = MMDSelector._grad_cov_kernel_matrix(X, Y)
        assert np.all(K >= 0)


class TestMedianHeuristic:
    """Test the median heuristic bandwidth estimation."""

    def test_positive(self):
        """Sigma should always be positive."""
        rng = np.random.RandomState(1)
        X = rng.randn(50, 10)
        sigma = MMDSelector._median_heuristic(X)
        assert sigma > 0

    def test_scales_with_data(self):
        """Scaling data by c should scale sigma by c."""
        rng = np.random.RandomState(2)
        X = rng.randn(100, 5)
        c = 3.0
        sigma_original = MMDSelector._median_heuristic(X)
        sigma_scaled = MMDSelector._median_heuristic(X * c)
        # sigma is based on median of pairwise distances, which scale linearly
        np.testing.assert_allclose(sigma_scaled, sigma_original * c, rtol=1e-6)

    def test_identical_points_returns_minimum(self):
        """If all points are identical, sigma should be the minimum guard value."""
        X = np.ones((20, 4))
        sigma = MMDSelector._median_heuristic(X)
        # Should return the guard value (1e-6) since all distances are 0
        assert sigma == pytest.approx(1e-6)

    def test_subsample_used_for_large_data(self):
        """Median heuristic should still work (not OOM) with large N."""
        rng = np.random.RandomState(3)
        X = rng.randn(5000, 10)
        # Should complete without error; uses subsampling internally
        sigma = MMDSelector._median_heuristic(X, subsample=500)
        assert sigma > 0


class TestComputeKernelColumn:
    """Test _compute_kernel_column which computes k(x_i, x_new) for all i."""

    def test_rbf_column_matches_matrix(self):
        """Column computation should match the corresponding column of the full matrix."""
        sel = _make_selector()
        rng = np.random.RandomState(10)
        X = rng.randn(20, 6)
        idx = 5
        sigma = 1.0
        col = sel._compute_kernel_column(X, X[idx], sigma, "rbf")
        full_K = MMDSelector._rbf_kernel_matrix(X, X[idx:idx+1], sigma)
        np.testing.assert_allclose(col, full_K.squeeze(), atol=1e-10)

    def test_cov_column_matches_matrix(self):
        """Column computation for cov kernel matches full matrix column."""
        sel = _make_selector()
        rng = np.random.RandomState(11)
        X = rng.randn(15, 4)
        idx = 3
        col = sel._compute_kernel_column(X, X[idx], None, "cov")
        full_K = MMDSelector._grad_cov_kernel_matrix(X, X[idx:idx+1])
        np.testing.assert_allclose(col, full_K.squeeze(), atol=1e-10)


class TestComputeSelfKernel:
    """Test _compute_self_kernel."""

    def test_rbf_self_kernel_is_one(self):
        """For RBF, k(x,x) = exp(0) = 1 for all x."""
        rng = np.random.RandomState(20)
        X = rng.randn(25, 8)
        self_k = MMDSelector._compute_self_kernel(X, sigma=2.0, kernel_type="rbf")
        np.testing.assert_allclose(self_k, np.ones(25), atol=1e-12)

    def test_cov_self_kernel_is_norm_to_fourth(self):
        """For cov, k(x,x) = <x,x>^2 = ||x||^4."""
        rng = np.random.RandomState(21)
        X = rng.randn(10, 5)
        self_k = MMDSelector._compute_self_kernel(X, sigma=None, kernel_type="cov")
        expected = np.sum(X ** 2, axis=1) ** 2
        np.testing.assert_allclose(self_k, expected, atol=1e-10)


class TestExactMarginalGreedy:
    """Test the core _greedy_mmd_exact selection algorithm."""

    def test_correct_count(self):
        """Returns exactly num_samples indices."""
        sel = _make_selector()
        rng = np.random.RandomState(30)
        candidates = rng.randn(50, 4)
        targets = rng.randn(20, 4)
        sigma = MMDSelector._median_heuristic(candidates)
        selected = sel._greedy_mmd_exact(candidates, targets, num_samples=10, sigma=sigma, kernel_type="rbf")
        assert len(selected) == 10

    def test_no_duplicates(self):
        """All selected indices are unique."""
        sel = _make_selector()
        rng = np.random.RandomState(31)
        candidates = rng.randn(40, 6)
        targets = rng.randn(15, 6)
        sigma = MMDSelector._median_heuristic(candidates)
        selected = sel._greedy_mmd_exact(candidates, targets, num_samples=20, sigma=sigma, kernel_type="rbf")
        assert len(set(selected)) == len(selected)

    def test_indices_in_valid_range(self):
        """All indices should be in [0, N_candidates)."""
        sel = _make_selector()
        rng = np.random.RandomState(32)
        N = 60
        candidates = rng.randn(N, 5)
        targets = rng.randn(10, 5)
        sigma = MMDSelector._median_heuristic(candidates)
        selected = sel._greedy_mmd_exact(candidates, targets, num_samples=15, sigma=sigma, kernel_type="rbf")
        assert all(0 <= idx < N for idx in selected)

    def test_first_pick_is_most_relevant_for_rbf(self):
        """First pick should be highest r_T(x) since k(x,x)=1 for all x with RBF.

        When the selected set is empty, the greedy criterion reduces to
        maximizing target relevance r_T(x) = mean_t k(x,t).
        """
        sel = _make_selector()
        rng = np.random.RandomState(33)
        # Create candidates where one is very close to all targets
        targets = rng.randn(20, 3)
        target_mean = targets.mean(axis=0)
        candidates = rng.randn(30, 3) * 5  # spread out
        # Make candidate 0 the centroid of targets
        candidates[0] = target_mean
        sigma = MMDSelector._median_heuristic(
            np.vstack([candidates, targets])
        )
        selected = sel._greedy_mmd_exact(candidates, targets, num_samples=5, sigma=sigma, kernel_type="rbf")
        # The point closest to target centroid should be picked first
        # (highest mean kernel to targets)
        assert selected[0] == 0

    def test_redundancy_promotes_diversity(self):
        """Given clustered data, greedy should cover multiple clusters.

        With 3 tight clusters in the candidates and a uniform target,
        the selection should pick from each cluster, not just the closest one.
        """
        sel = _make_selector()
        rng = np.random.RandomState(34)
        # 3 clusters of 20 points each
        cluster1 = rng.randn(20, 2) * 0.1 + np.array([5.0, 0.0])
        cluster2 = rng.randn(20, 2) * 0.1 + np.array([-5.0, 0.0])
        cluster3 = rng.randn(20, 2) * 0.1 + np.array([0.0, 5.0])
        candidates = np.vstack([cluster1, cluster2, cluster3])

        # Target is spread across all three cluster centers
        targets = np.array([[5.0, 0.0], [-5.0, 0.0], [0.0, 5.0]])

        sigma = MMDSelector._median_heuristic(candidates)
        selected = sel._greedy_mmd_exact(candidates, targets, num_samples=9, sigma=sigma, kernel_type="rbf")

        # Check that all 3 clusters are represented
        cluster_ids = [idx // 20 for idx in selected]
        assert 0 in cluster_ids, "Cluster 1 should be represented"
        assert 1 in cluster_ids, "Cluster 2 should be represented"
        assert 2 in cluster_ids, "Cluster 3 should be represented"

    def test_multimodal_target_coverage(self):
        """With bimodal target, selection should cover both modes."""
        sel = _make_selector()
        rng = np.random.RandomState(35)
        # Two well-separated modes in the target
        target_mode1 = rng.randn(10, 2) * 0.1 + np.array([10.0, 0.0])
        target_mode2 = rng.randn(10, 2) * 0.1 + np.array([-10.0, 0.0])
        targets = np.vstack([target_mode1, target_mode2])

        # Candidates uniformly around both modes
        cands_near_mode1 = rng.randn(25, 2) * 0.2 + np.array([10.0, 0.0])
        cands_near_mode2 = rng.randn(25, 2) * 0.2 + np.array([-10.0, 0.0])
        candidates = np.vstack([cands_near_mode1, cands_near_mode2])

        sigma = MMDSelector._median_heuristic(np.vstack([candidates, targets]))
        selected = sel._greedy_mmd_exact(candidates, targets, num_samples=10, sigma=sigma, kernel_type="rbf")

        # Check both modes are covered (indices 0-24 near mode1, 25-49 near mode2)
        near_mode1 = [idx for idx in selected if idx < 25]
        near_mode2 = [idx for idx in selected if idx >= 25]
        assert len(near_mode1) > 0, "Mode 1 should be covered"
        assert len(near_mode2) > 0, "Mode 2 should be covered"

    def test_mmd_decreases_with_more_samples(self):
        """MMD^2(S, T) should generally decrease as |S| increases.

        We verify by computing the actual MMD^2 at different subset sizes.
        """
        sel = _make_selector()
        rng = np.random.RandomState(36)
        candidates = rng.randn(100, 4)
        targets = rng.randn(30, 4)
        sigma = MMDSelector._median_heuristic(np.vstack([candidates, targets]))

        # Select a large set
        selected = sel._greedy_mmd_exact(candidates, targets, num_samples=50, sigma=sigma, kernel_type="rbf")

        # Compute MMD^2 at increasing subset sizes
        def mmd_squared(S_indices):
            S = candidates[S_indices]
            T = targets
            K_SS = MMDSelector._rbf_kernel_matrix(S, S, sigma)
            K_ST = MMDSelector._rbf_kernel_matrix(S, T, sigma)
            K_TT = MMDSelector._rbf_kernel_matrix(T, T, sigma)
            m = len(S)
            n = len(T)
            np.fill_diagonal(K_SS, 0.0)
            np.fill_diagonal(K_TT, 0.0)
            return (
                K_SS.sum() / (m * (m - 1))
                - 2.0 * K_ST.sum() / (m * n)
                + K_TT.sum() / (n * (n - 1))
            )

        mmd_5 = mmd_squared(selected[:5])
        mmd_20 = mmd_squared(selected[:20])
        mmd_50 = mmd_squared(selected[:50])

        # MMD should generally decrease (allow some tolerance for non-monotonicity
        # in early steps, but overall trend should hold)
        assert mmd_50 < mmd_5, (
            f"MMD should decrease with more samples: mmd(50)={mmd_50:.6f} vs mmd(5)={mmd_5:.6f}"
        )

    def test_works_with_cov_kernel(self):
        """Greedy selection also works with the covariance kernel."""
        sel = _make_selector()
        rng = np.random.RandomState(37)
        candidates = rng.randn(30, 4)
        targets = rng.randn(10, 4)
        selected = sel._greedy_mmd_exact(candidates, targets, num_samples=8, sigma=None, kernel_type="cov")
        assert len(selected) == 8
        assert len(set(selected)) == 8

    def test_num_samples_capped_at_pool_size(self):
        """If num_samples > N, should return at most N indices."""
        sel = _make_selector()
        rng = np.random.RandomState(38)
        candidates = rng.randn(10, 3)
        targets = rng.randn(5, 3)
        sigma = MMDSelector._median_heuristic(candidates)
        selected = sel._greedy_mmd_exact(candidates, targets, num_samples=100, sigma=sigma, kernel_type="rbf")
        assert len(selected) == 10  # capped at pool size


class TestTargetRelevance:
    """Test target relevance computation."""

    def test_rbf_relevance_shape(self):
        """Target relevance should have shape (N_candidates,)."""
        rng = np.random.RandomState(40)
        candidates = rng.randn(30, 5)
        targets = rng.randn(10, 5)
        sigma = 1.0
        relevance = MMDSelector._compute_target_relevance_rbf(candidates, targets, sigma)
        assert relevance.shape == (30,)

    def test_rbf_relevance_positive(self):
        """RBF relevance should always be positive."""
        rng = np.random.RandomState(41)
        candidates = rng.randn(20, 4)
        targets = rng.randn(8, 4)
        sigma = 2.0
        relevance = MMDSelector._compute_target_relevance_rbf(candidates, targets, sigma)
        assert np.all(relevance > 0)

    def test_rbf_relevance_closest_is_highest(self):
        """Point closest to targets should have highest relevance."""
        targets = np.array([[0.0, 0.0]])
        candidates = np.array([
            [0.0, 0.0],   # identical to target
            [1.0, 0.0],   # close
            [10.0, 0.0],  # far
        ])
        sigma = 1.0
        relevance = MMDSelector._compute_target_relevance_rbf(candidates, targets, sigma)
        assert relevance[0] > relevance[1] > relevance[2]

    def test_generic_relevance_matches_rbf(self):
        """_compute_target_relevance_generic with 'rbf' should match _compute_target_relevance_rbf."""
        sel = _make_selector()
        rng = np.random.RandomState(42)
        candidates = rng.randn(15, 3)
        targets = rng.randn(8, 3)
        sigma = 1.5
        r_rbf = MMDSelector._compute_target_relevance_rbf(candidates, targets, sigma)
        r_generic = sel._compute_target_relevance_generic(candidates, targets, sigma, "rbf")
        np.testing.assert_allclose(r_rbf, r_generic, atol=1e-10)


class TestGradKernelGuard:
    """Test guard clauses and validation logic."""

    def test_no_target_raises_for_grad_rbf(self):
        """Must raise ValueError if kernel_type=grad_rbf but no eval_dataset."""
        # We cannot use __new__ bypass here since we need __init__ to run the check.
        # Instead, we provide minimal mock objects.
        class MockAccelerator:
            device = "cpu"
            is_main_process = True
            is_local_main_process = True
            process_index = 0
            class state:
                deepspeed_plugin = None
            def wait_for_everyone(self):
                pass

        class MockCollator:
            def __call__(self, batch):
                return batch

        dataset = [{"text": "a"}]
        with pytest.raises(ValueError, match="requires a target dataset"):
            MMDSelector(
                dataset=dataset,
                accelerator=MockAccelerator(),
                data_collator=MockCollator(),
                cache_dir="/tmp/test_guard",
                eval_dataset=None,
                kernel_type="grad_rbf",
            )

    def test_no_target_raises_for_grad_cov(self):
        """Must raise ValueError if kernel_type=grad_cov but no eval_dataset."""
        class MockAccelerator:
            device = "cpu"
            is_main_process = True
            is_local_main_process = True
            process_index = 0
            class state:
                deepspeed_plugin = None
            def wait_for_everyone(self):
                pass

        class MockCollator:
            def __call__(self, batch):
                return batch

        dataset = [{"text": "a"}]
        with pytest.raises(ValueError, match="requires a target dataset"):
            MMDSelector(
                dataset=dataset,
                accelerator=MockAccelerator(),
                data_collator=MockCollator(),
                cache_dir="/tmp/test_guard",
                eval_dataset=None,
                kernel_type="grad_cov",
            )

    def test_unknown_kernel_type_in_column(self):
        """_compute_kernel_column should raise for unknown kernel_type."""
        sel = _make_selector()
        X = np.random.randn(5, 3)
        with pytest.raises(ValueError, match="Unknown kernel_type"):
            sel._compute_kernel_column(X, X[0], sigma=1.0, kernel_type="invalid")

    def test_unknown_kernel_type_in_self_kernel(self):
        """_compute_self_kernel should raise for unknown kernel_type."""
        X = np.random.randn(5, 3)
        with pytest.raises(ValueError, match="Unknown kernel_type"):
            MMDSelector._compute_self_kernel(X, sigma=1.0, kernel_type="invalid")
