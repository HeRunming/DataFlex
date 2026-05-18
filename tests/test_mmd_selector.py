"""
Unit tests for MMD selector kernel functions.

Tests kernel computation correctness using simple examples.
"""

import torch
import pytest
from dataflex.train.selector.mmd_selector import MMDSelector


class MockAccelerator:
    """Mock accelerator for testing."""
    def __init__(self):
        self.device = torch.device("cpu")
        self.process_index = 0
        self.is_main_process = True
        self.is_local_main_process = True
        self.state = MockState()
    
    def wait_for_everyone(self):
        pass
    
    class MockState:
        deepspeed_plugin = None


class MockDataCollator:
    """Mock data collator for testing."""
    def __call__(self, batch):
        return batch


@pytest.fixture
def selector():
    """Create a test MMD selector."""
    accelerator = MockAccelerator()
    dataset = [{"text": f"sample_{i}"} for i in range(10)]
    eval_dataset = [{"text": f"eval_{i}"} for i in range(5)]
    
    selector = MMDSelector(
        dataset=dataset,
        eval_dataset=eval_dataset,
        accelerator=accelerator,
        data_collator=MockDataCollator(),
        cache_dir="/tmp/mmd_test",
        kernel_type="rbf",
        sigma=1.0,
        degree=3,
        coef0=0.0
    )
    return selector


def test_rbf_kernel_basic(selector):
    """Test RBF kernel computation on simple examples."""
    # Simple 2D examples for easy verification
    X = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    Y = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    
    K = selector.compute_rbf_kernel(X, Y, sigma=1.0)
    
    # K should be [2, 2]
    assert K.shape == (2, 2)
    
    # K[i, j] should be between 0 and 1
    assert (K >= 0).all() and (K <= 1).all()
    
    # Diagonal elements (identical points) should be close to 1
    assert K[0, 0].item() > 0.99
    
    # Different points should have lower values
    assert K[0, 1].item() < K[0, 0].item()


def test_rbf_kernel_symmetry(selector):
    """Test that RBF kernel matrix respects distance symmetry."""
    X = torch.randn(5, 10)
    Y = torch.randn(8, 10)
    
    K = selector.compute_rbf_kernel(X, Y, sigma=2.0)
    
    # Kernel matrix shape should be [N, M]
    assert K.shape == (5, 8)
    
    # All values should be non-negative
    assert (K >= 0).all()
    
    # All values should be <= 1 for RBF
    assert (K <= 1).all()


def test_polynomial_kernel_basic(selector):
    """Test polynomial kernel computation."""
    X = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    Y = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    
    K = selector.compute_polynomial_kernel(X, Y, degree=2, coef0=1.0)
    
    # K should be [2, 2]
    assert K.shape == (2, 2)
    
    # K[0, 0] = (1.0 + 1.0)^2 = 4.0
    assert torch.isclose(K[0, 0], torch.tensor(4.0), atol=1e-6)


def test_polynomial_kernel_shape(selector):
    """Test polynomial kernel output shape."""
    X = torch.randn(10, 20)
    Y = torch.randn(15, 20)
    
    K = selector.compute_polynomial_kernel(X, Y, degree=3, coef0=0.5)
    
    assert K.shape == (10, 15)
    assert K.dtype == torch.float32


def test_linear_kernel_basic(selector):
    """Test linear kernel (dot product)."""
    X = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    Y = torch.tensor([[1.0, 0.0]])
    
    K = selector.compute_linear_kernel(X, Y)
    
    # K[0, 0] = 1.0 * 1.0 + 0.0 * 0.0 = 1.0
    assert torch.isclose(K[0, 0], torch.tensor(1.0))
    
    # K[1, 0] = 0.0 * 1.0 + 1.0 * 0.0 = 0.0
    assert torch.isclose(K[1, 0], torch.tensor(0.0))


def test_linear_kernel_dot_product(selector):
    """Test that linear kernel is equivalent to matrix multiplication."""
    X = torch.randn(5, 10)
    Y = torch.randn(8, 10)
    
    K_linear = selector.compute_linear_kernel(X, Y)
    K_expected = torch.mm(X, Y.t())
    
    assert torch.allclose(K_linear, K_expected)


def test_kernel_matrix_dispatch_rbf(selector):
    """Test that compute_kernel_matrix dispatches to RBF correctly."""
    selector.kernel_type = "rbf"
    X = torch.randn(3, 5)
    Y = torch.randn(4, 5)
    
    K = selector.compute_kernel_matrix(X, Y)
    
    assert K.shape == (3, 4)
    assert (K >= 0).all() and (K <= 1).all()


def test_kernel_matrix_dispatch_polynomial(selector):
    """Test that compute_kernel_matrix dispatches to polynomial correctly."""
    selector.kernel_type = "polynomial"
    X = torch.randn(3, 5)
    Y = torch.randn(4, 5)
    
    K = selector.compute_kernel_matrix(X, Y)
    
    assert K.shape == (3, 4)


def test_kernel_matrix_dispatch_linear(selector):
    """Test that compute_kernel_matrix dispatches to linear correctly."""
    selector.kernel_type = "linear"
    X = torch.randn(3, 5)
    Y = torch.randn(4, 5)
    
    K = selector.compute_kernel_matrix(X, Y)
    
    assert K.shape == (3, 4)
    # Should match matrix multiplication
    assert torch.allclose(K, torch.mm(X, Y.t()))


def test_kernel_matrix_unknown_type(selector):
    """Test that unknown kernel type raises error."""
    selector.kernel_type = "unknown_kernel"
    X = torch.randn(3, 5)
    Y = torch.randn(4, 5)
    
    with pytest.raises(ValueError, match="Unknown kernel type"):
        selector.compute_kernel_matrix(X, Y)


def test_rbf_bandwidth_effect(selector):
    """Test that RBF bandwidth (sigma) affects kernel values."""
    X = torch.tensor([[0.0, 0.0]])
    Y = torch.tensor([[1.0, 0.0]])
    
    # Small sigma -> smaller kernel values for distant points
    K_small = selector.compute_rbf_kernel(X, Y, sigma=0.5)
    
    # Large sigma -> larger kernel values for distant points
    K_large = selector.compute_rbf_kernel(X, Y, sigma=2.0)
    
    assert K_small[0, 0] < K_large[0, 0]


def test_kernel_scoring_example(selector):
    """Test that kernel scoring produces reasonable results."""
    # Create simple example: 3 train samples, 2 eval samples
    train_grads = torch.tensor([
        [1.0, 0.0],  # Similar to eval[0]
        [0.0, 1.0],  # Similar to eval[1]
        [0.5, 0.5],  # Medium similarity to both
    ])
    eval_grads = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    
    selector.kernel_type = "linear"
    K = selector.compute_kernel_matrix(train_grads, eval_grads)
    
    # K should be [3, 2]
    assert K.shape == (3, 2)
    
    # Scores: mean across eval samples
    scores = K.mean(dim=1)
    
    # First sample should be highest (matches eval[0])
    assert scores[0] > scores[2]
    # Second sample should also be high (matches eval[1])
    assert scores[1] > scores[2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
