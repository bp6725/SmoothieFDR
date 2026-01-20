"""
Unified Point-wise Optimization for Spatial FDR (Stage 1).

This module provides a configurable optimizer for the point-wise
alpha estimation problem in the Spatial FDR framework.

The optimization problem:
    min_c  -Σ log[α(x_i) f₀(p_i) + (1-α(x_i)) f₁(p_i)]
           + λ_reg ||α||²_H
           + λ_bound Σ[max(0, α_i-1)² + max(0, -α_i)²]

where α(x) = K @ c  (representer theorem)

Reference: "Spatially Smooth Bayesian FDR via Reproducing Kernels"
"""

import numpy as np
from typing import Optional, Dict, Tuple, Literal, Callable, Union
from dataclasses import dataclass, field


@dataclass
class OptimizerConfig:
    """Configuration for point-wise optimization."""
    # Regularization
    lambda_reg: float = 10.0
    lambda_bound: float = 500.0

    # Optimization
    learning_rate: float = 0.0005
    max_iter: int = 5000
    tol: float = 1e-6

    # Gradient control
    gradient_clip: float = 5.0

    # Initialization
    c_init: str = "auto"  # "auto", "zeros", "random"

    # Logging
    log_interval: int = 50

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'lambda_reg': self.lambda_reg,
            'lambda_bound': self.lambda_bound,
            'learning_rate': self.learning_rate,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'gradient_clip': self.gradient_clip,
            'c_init': self.c_init,
            'log_interval': self.log_interval
        }


# Preset configurations for different use cases
OPTIMIZER_PRESETS: Dict[str, OptimizerConfig] = {
    'default': OptimizerConfig(),

    'euclidean_cv': OptimizerConfig(
        lambda_reg=10.0,
        lambda_bound=500.0,
        learning_rate=0.0005,
        max_iter=5000,
        c_init="auto"
    ),

    'graph_gsea': OptimizerConfig(
        lambda_reg=10.0,
        lambda_bound=500.0,
        learning_rate=0.05,
        max_iter=15000,
        c_init="auto"
    ),

    'fast_tuning': OptimizerConfig(
        lambda_reg=10.0,
        lambda_bound=500.0,
        learning_rate=0.001,
        max_iter=2000,
        c_init="auto"
    ),
}


class PointwiseOptimizer:
    """
    Point-wise optimizer for Stage 1 of Spatial FDR.

    Implements natural gradient descent for the mixture likelihood
    with RKHS regularization and boundary constraints.

    Parameters
    ----------
    config : OptimizerConfig or str
        Configuration object or preset name ('default', 'euclidean_cv',
        'graph_gsea', 'fast_tuning')
    verbose : bool, default=False
        Print optimization progress

    Attributes
    ----------
    history_ : dict
        Optimization history containing:
        - 'losses': Total loss at each logged iteration
        - 'grad_norms': Gradient norms
        - 'alpha_history': Alpha snapshots
        - 'violations': Number of constraint violations

    Examples
    --------
    >>> from spatial_fdr_evaluation.methods.optimization import PointwiseOptimizer
    >>>
    >>> # Using preset
    >>> optimizer = PointwiseOptimizer('euclidean_cv', verbose=True)
    >>> c, history = optimizer.optimize(K, p_values, f0_vals, f1_vals)
    >>>
    >>> # Custom config
    >>> config = OptimizerConfig(lambda_reg=5.0, max_iter=10000)
    >>> optimizer = PointwiseOptimizer(config)
    >>> c, history = optimizer.optimize(K, p_values, f0_vals, f1_vals)
    """

    def __init__(
        self,
        config: Union[OptimizerConfig, str] = 'default',
        verbose: bool = False
    ):
        if isinstance(config, str):
            if config not in OPTIMIZER_PRESETS:
                raise ValueError(
                    f"Unknown preset '{config}'. "
                    f"Available: {list(OPTIMIZER_PRESETS.keys())}"
                )
            self.config = OPTIMIZER_PRESETS[config]
        else:
            self.config = config

        self.verbose = verbose
        self.history_ = None

    def _initialize_coefficients(
        self,
        K: np.ndarray,
        n: int
    ) -> np.ndarray:
        """Initialize optimization coefficients."""
        if self.config.c_init == "auto":
            # Initialize so alpha starts near 1.0 (conservative)
            row_sums = K.sum(axis=1).mean() + 1e-10
            return np.ones(n) * (1.0 / row_sums)
        elif self.config.c_init == "zeros":
            return np.zeros(n)
        elif self.config.c_init == "random":
            return (np.random.rand(n) - 0.5) * 2
        else:
            raise ValueError(f"Unknown c_init: {self.config.c_init}")

    def _compute_loss(
        self,
        c: np.ndarray,
        K: np.ndarray,
        f0_vals: np.ndarray,
        f1_vals: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """
        Compute total loss and its components.

        Returns
        -------
        total_loss : float
        nll_loss : float
            Negative log-likelihood
        reg_loss : float
            RKHS regularization
        bound_loss : float
            Boundary penalty
        """
        alpha = K @ c

        # Mixture density
        mixture = alpha * f0_vals + (1 - alpha) * f1_vals
        mixture = np.clip(mixture, 1e-12, None)

        # Components
        nll_loss = -np.sum(np.log(mixture))

        reg_loss = self.config.lambda_reg * (c @ K @ c)

        bound_loss = self.config.lambda_bound * np.sum(
            np.maximum(0, alpha - 1)**2 +
            np.maximum(0, -alpha)**2
        )

        total_loss = nll_loss + reg_loss + bound_loss

        return total_loss, nll_loss, reg_loss, bound_loss

    def _compute_gradient(
        self,
        c: np.ndarray,
        K: np.ndarray,
        f0_vals: np.ndarray,
        f1_vals: np.ndarray
    ) -> np.ndarray:
        """
        Compute natural gradient.

        The natural gradient for this problem is:
        ∇̃L = -(f₀ - f₁)/mixture + 2λ_reg·c + 2λ_bound·(violations)
        """
        alpha = K @ c

        # Mixture density
        mixture = alpha * f0_vals + (1 - alpha) * f1_vals
        mixture = np.clip(mixture, 1e-12, None)

        # Gradient components
        grad_nll = -(f0_vals - f1_vals) / mixture
        grad_reg = 2 * self.config.lambda_reg * c
        grad_bound = 2 * self.config.lambda_bound * (
            np.maximum(0, alpha - 1) - np.maximum(0, -alpha)
        )

        return grad_nll + grad_reg + grad_bound

    def optimize(
        self,
        K: np.ndarray,
        p_values: np.ndarray,
        f0_vals: np.ndarray,
        f1_vals: np.ndarray,
        c_init: Optional[np.ndarray] = None,
        lambda_reg: Optional[float] = None,
        lambda_bound: Optional[float] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Run point-wise optimization.

        Parameters
        ----------
        K : np.ndarray, shape (n, n)
            Kernel matrix
        p_values : np.ndarray, shape (n,)
            P-values (used for logging only)
        f0_vals : np.ndarray, shape (n,)
            Null density evaluated at p_values
        f1_vals : np.ndarray, shape (n,)
            Alternative density evaluated at p_values
        c_init : np.ndarray, optional
            Custom initialization for coefficients
        lambda_reg : float, optional
            Override config lambda_reg (useful for CV)
        lambda_bound : float, optional
            Override config lambda_bound

        Returns
        -------
        c : np.ndarray
            Optimized coefficients
        history : dict
            Optimization history
        """
        n = K.shape[0]

        # Allow runtime override of regularization parameters
        orig_lambda_reg = self.config.lambda_reg
        orig_lambda_bound = self.config.lambda_bound

        if lambda_reg is not None:
            self.config.lambda_reg = lambda_reg
        if lambda_bound is not None:
            self.config.lambda_bound = lambda_bound

        try:
            # Initialize
            if c_init is not None:
                c = c_init.copy()
            else:
                c = self._initialize_coefficients(K, n)

            # History tracking
            self.history_ = {
                'losses': [],
                'grad_norms': [],
                'alpha_history': [],
                'violations': []
            }

            # Optimization loop
            for t in range(self.config.max_iter):
                # Compute gradient
                grad = self._compute_gradient(c, K, f0_vals, f1_vals)
                grad_norm = np.linalg.norm(grad)

                # Gradient clipping
                if grad_norm > self.config.gradient_clip:
                    grad = grad * (self.config.gradient_clip / grad_norm)

                # Update
                c = c - self.config.learning_rate * grad

                # Log at intervals
                if t % self.config.log_interval == 0:
                    alpha = K @ c
                    total_loss, _, _, _ = self._compute_loss(c, K, f0_vals, f1_vals)
                    violations = np.sum((alpha < 0) | (alpha > 1))

                    self.history_['losses'].append(total_loss)
                    self.history_['grad_norms'].append(grad_norm)
                    self.history_['alpha_history'].append(alpha.copy())
                    self.history_['violations'].append(violations)

                    if self.verbose and t % (self.config.log_interval * 10) == 0:
                        print(f"  Iter {t}: loss={total_loss:.1f}, |∇|={grad_norm:.2f}, "
                              f"α∈[{alpha.min():.3f}, {alpha.max():.3f}], viol={violations}")

                # Check convergence
                if grad_norm < self.config.tol and t > 100:
                    if self.verbose:
                        print(f"  Converged at iteration {t}")
                    break

            return c, self.history_

        finally:
            # Restore original config
            self.config.lambda_reg = orig_lambda_reg
            self.config.lambda_bound = orig_lambda_bound


def optimize_pointwise(
    K: np.ndarray,
    p_values: np.ndarray,
    f0_vals: np.ndarray,
    f1_vals: np.ndarray,
    lambda_reg: float = 10.0,
    lambda_bound: float = 500.0,
    learning_rate: float = 0.0005,
    max_iter: int = 5000,
    gradient_clip: float = 5.0,
    tol: float = 1e-6,
    c_init: Optional[np.ndarray] = None,
    verbose: bool = False
) -> Tuple[np.ndarray, Optional[dict]]:
    """
    Convenience function for point-wise optimization.

    This is a drop-in replacement for the duplicated optimize_pointwise
    functions across main scripts.

    Parameters
    ----------
    K : np.ndarray, shape (n, n)
        Kernel matrix
    p_values : np.ndarray, shape (n,)
        P-values
    f0_vals : np.ndarray, shape (n,)
        Null density evaluated at p_values
    f1_vals : np.ndarray, shape (n,)
        Alternative density evaluated at p_values
    lambda_reg : float, default=10.0
        RKHS regularization parameter
    lambda_bound : float, default=500.0
        Boundary penalty parameter
    learning_rate : float, default=0.0005
        Learning rate for gradient descent
    max_iter : int, default=5000
        Maximum number of iterations
    gradient_clip : float, default=5.0
        Maximum gradient norm
    tol : float, default=1e-6
        Convergence tolerance
    c_init : np.ndarray, optional
        Initial coefficients
    verbose : bool, default=False
        Print progress

    Returns
    -------
    c : np.ndarray
        Optimized coefficients
    history : dict or None
        Optimization history (None if verbose=False for backward compat)
    """
    config = OptimizerConfig(
        lambda_reg=lambda_reg,
        lambda_bound=lambda_bound,
        learning_rate=learning_rate,
        max_iter=max_iter,
        gradient_clip=gradient_clip,
        tol=tol,
        c_init="auto" if c_init is None else "custom"
    )

    optimizer = PointwiseOptimizer(config, verbose=verbose)

    c, history = optimizer.optimize(
        K, p_values, f0_vals, f1_vals,
        c_init=c_init
    )

    return c, history if verbose else None


def create_optimizer(
    preset: str = 'default',
    **overrides
) -> PointwiseOptimizer:
    """
    Factory function to create optimizer with preset and overrides.

    Parameters
    ----------
    preset : str
        Preset name: 'default', 'euclidean_cv', 'graph_gsea', 'fast_tuning'
    **overrides
        Override specific config values

    Returns
    -------
    PointwiseOptimizer
        Configured optimizer
    """
    if preset not in OPTIMIZER_PRESETS:
        raise ValueError(f"Unknown preset: {preset}")

    # Get base config
    base_config = OPTIMIZER_PRESETS[preset]
    config_dict = base_config.to_dict()

    # Apply overrides
    for key, value in overrides.items():
        if key in config_dict:
            config_dict[key] = value

    config = OptimizerConfig(**config_dict)
    return PointwiseOptimizer(config)
