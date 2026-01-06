"""
Global FDR Inference via Kernel Logistic Regression (Stage 2).

This module provides the GlobalFDRRegressor class that extends point-wise
alpha estimates to unseen locations using kernel logistic regression.

The two-stage procedure:
1. Stage 1 (Point-wise): Optimize alpha at observed locations (see spatial_fdr.py)
2. Stage 2 (Global): Extend predictions to entire domain using KLR (this module)

Reference: "Spatially Smooth Bayesian FDR via Reproducing Kernels"
"""

import numpy as np
from scipy.special import expit, logit
from typing import Optional, Dict, Literal
from dataclasses import dataclass


@dataclass
class GlobalFDRConfig:
    """Configuration for GlobalFDRRegressor."""
    lambda_global: float = 1.0
    learning_rate: float = 0.005
    max_iter: int = 2000
    tol: float = 1e-5
    clip_alpha: tuple = (0.01, 0.99)
    regularization_epsilon: float = 1e-4
    gradient_clip: float = 5.0


class GlobalFDRRegressor:
    """
    Stage 2: Global Inference using Natural Gradient Kernel Logistic Regression.

    This class takes the point-wise alpha estimates from Stage 1 and learns
    a smooth function that can predict alpha at any location in the domain.

    The optimization problem:
        min_c  -Σ[α*_i log(σ(g_i)) + (1-α*_i) log(1-σ(g_i))] + λ ||c||²_K

    where:
        - g = K @ c  (linear predictor)
        - σ(g) = expit(g) = 1/(1+exp(-g))  (ensures output in [0,1])
        - α*_i are the Stage 1 point-wise estimates
        - λ is the regularization parameter

    Parameters
    ----------
    lambda_global : float, default=1.0
        Regularization parameter for RKHS norm
    learning_rate : float, default=0.005
        Learning rate for natural gradient descent
    max_iter : int, default=2000
        Maximum number of optimization iterations
    tol : float, default=1e-5
        Convergence tolerance for gradient norm
    clip_alpha : tuple, default=(0.01, 0.99)
        Clipping bounds for alpha targets to avoid numerical issues
    gradient_clip : float, default=5.0
        Maximum gradient norm (for stability)
    verbose : bool, default=False
        Print optimization progress

    Attributes
    ----------
    c_ : np.ndarray
        Fitted coefficients (representer theorem)
    history_ : dict
        Optimization history (losses, gradient norms)
    is_fitted_ : bool
        Whether the model has been fitted

    Examples
    --------
    >>> from spatial_fdr_evaluation.methods.global_inference import GlobalFDRRegressor
    >>>
    >>> # After Stage 1 optimization
    >>> klr = GlobalFDRRegressor(lambda_global=0.5)
    >>> klr.fit(K_train, alpha_stage1)
    >>>
    >>> # Predict on new locations
    >>> alpha_pred = klr.predict(K_test_cross)
    """

    def __init__(
        self,
        lambda_global: float = 1.0,
        learning_rate: float = 0.005,
        max_iter: int = 2000,
        tol: float = 1e-5,
        clip_alpha: tuple = (0.01, 0.99),
        gradient_clip: float = 5.0,
        verbose: bool = False
    ):
        self.lambda_global = lambda_global
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.clip_alpha = clip_alpha
        self.gradient_clip = gradient_clip
        self.verbose = verbose

        # Will be set during fit
        self.c_ = None
        self.history_ = None
        self.is_fitted_ = False
        self._K_train = None  # Store for diagnostics

    @classmethod
    def from_config(cls, config: GlobalFDRConfig, verbose: bool = False) -> 'GlobalFDRRegressor':
        """Create a GlobalFDRRegressor from a configuration object."""
        return cls(
            lambda_global=config.lambda_global,
            learning_rate=config.learning_rate,
            max_iter=config.max_iter,
            tol=config.tol,
            clip_alpha=config.clip_alpha,
            gradient_clip=config.gradient_clip,
            verbose=verbose
        )

    def _compute_loss(
        self,
        c: np.ndarray,
        K: np.ndarray,
        alpha_target: np.ndarray
    ) -> float:
        """
        Compute the kernel logistic regression loss.

        L(c) = -Σ[α* log(σ(Kc)) + (1-α*) log(1-σ(Kc))] + λ c^T K c
        """
        g = K @ c
        sigma = expit(np.clip(g, -500, 500))
        sigma = np.clip(sigma, 1e-10, 1 - 1e-10)

        # Cross-entropy loss
        log_loss = -np.sum(
            alpha_target * np.log(sigma) +
            (1 - alpha_target) * np.log(1 - sigma)
        )

        # RKHS regularization
        rkhs_penalty = self.lambda_global * (c @ K @ c)

        return log_loss + rkhs_penalty

    def _compute_gradient(
        self,
        c: np.ndarray,
        K: np.ndarray,
        alpha_target: np.ndarray
    ) -> np.ndarray:
        """
        Compute the natural gradient.

        Natural gradient: ∇L = K @ (σ - α*) + 2λ K @ c
                             = K @ [(σ - α*) + 2λc]

        We use the natural gradient (without K premultiplication) for efficiency:
        ∇̃L = (σ - α*) + 2λc
        """
        g = K @ c
        sigma = expit(np.clip(g, -500, 500))

        # Natural gradient (more stable, avoids K^{-1})
        grad = (sigma - alpha_target) + 2 * self.lambda_global * c

        return grad

    def fit(
        self,
        K_train: np.ndarray,
        alpha_hat_stage1: np.ndarray,
        warm_start: bool = True
    ) -> 'GlobalFDRRegressor':
        """
        Fit the global inference model.

        Parameters
        ----------
        K_train : np.ndarray, shape (n, n)
            Kernel matrix for training locations
        alpha_hat_stage1 : np.ndarray, shape (n,)
            Point-wise alpha estimates from Stage 1
        warm_start : bool, default=True
            Initialize with ridge regression solution

        Returns
        -------
        self : GlobalFDRRegressor
            Fitted model
        """
        n = K_train.shape[0]

        # Clip alpha targets to avoid numerical issues with logit
        alpha_clipped = np.clip(
            alpha_hat_stage1,
            self.clip_alpha[0],
            self.clip_alpha[1]
        )

        # Store for diagnostics
        self._K_train = K_train

        # Initialize coefficients
        if warm_start:
            # Warm start with ridge regression solution in logit space
            target_logits = logit(alpha_clipped)
            try:
                self.c_ = np.linalg.solve(
                    K_train + self.clip_alpha[0] * np.eye(n),
                    target_logits
                )
            except np.linalg.LinAlgError:
                if self.verbose:
                    print("Warning: Ridge warm start failed, using zeros")
                self.c_ = np.zeros(n)
        else:
            self.c_ = np.zeros(n)

        # Optimization history
        self.history_ = {
            'losses': [],
            'grad_norms': [],
            'alpha_range': []
        }

        # Natural gradient descent
        for iteration in range(self.max_iter):
            # Compute gradient
            grad = self._compute_gradient(self.c_, K_train, alpha_clipped)
            grad_norm = np.linalg.norm(grad)

            # Gradient clipping
            if grad_norm > self.gradient_clip:
                grad = grad * (self.gradient_clip / grad_norm)

            # Update
            self.c_ = self.c_ - self.learning_rate * grad

            # Track history
            if iteration % 50 == 0:
                loss = self._compute_loss(self.c_, K_train, alpha_clipped)
                # Compute alpha directly (don't use predict() as is_fitted_ not yet set)
                alpha_current = expit(K_train @ self.c_)

                self.history_['losses'].append(loss)
                self.history_['grad_norms'].append(grad_norm)
                self.history_['alpha_range'].append((alpha_current.min(), alpha_current.max()))

                if self.verbose and iteration % 200 == 0:
                    print(f"  Iter {iteration}: loss={loss:.2f}, |∇|={grad_norm:.2e}, "
                          f"α∈[{alpha_current.min():.3f}, {alpha_current.max():.3f}]")

            # Check convergence
            if grad_norm < self.tol:
                if self.verbose:
                    print(f"  Converged at iteration {iteration}")
                break

        self.is_fitted_ = True

        if self.verbose:
            final_alpha = self.predict(K_train)
            print(f"  Final α range: [{final_alpha.min():.3f}, {final_alpha.max():.3f}]")

        return self

    def predict(self, K_test: np.ndarray) -> np.ndarray:
        """
        Predict alpha at new locations.

        Parameters
        ----------
        K_test : np.ndarray, shape (m, n) or (n, n)
            Kernel matrix between test and training locations.
            If K_test is (n, n), predicts at training locations.
            If K_test is (m, n), predicts at m new locations using
            cross-kernel with n training locations.

        Returns
        -------
        alpha_pred : np.ndarray, shape (m,) or (n,)
            Predicted alpha values (probability of null)
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")

        g = K_test @ self.c_
        return expit(g)

    def predict_with_uncertainty(
        self,
        K_test: np.ndarray,
        K_test_diag: Optional[np.ndarray] = None
    ) -> tuple:
        """
        Predict alpha with uncertainty estimates.

        Uses Laplace approximation for uncertainty quantification.

        Parameters
        ----------
        K_test : np.ndarray, shape (m, n)
            Cross-kernel matrix
        K_test_diag : np.ndarray, shape (m,), optional
            Diagonal of K(X_test, X_test) for variance computation

        Returns
        -------
        alpha_pred : np.ndarray
            Predicted alpha values
        alpha_std : np.ndarray
            Standard deviation estimates (if K_test_diag provided)
        """
        alpha_pred = self.predict(K_test)

        if K_test_diag is None:
            return alpha_pred, None

        # Approximate variance using Laplace approximation
        # This is a simplified version - full version would need Hessian
        g = K_test @ self.c_
        sigma = expit(g)

        # Variance in logit space (approximation)
        var_logit = K_test_diag - np.sum(
            K_test @ np.linalg.solve(
                self._K_train + self.lambda_global * np.eye(len(self.c_)),
                K_test.T
            ),
            axis=1
        )
        var_logit = np.maximum(var_logit, 1e-6)

        # Transform to probability space using delta method
        # Var(σ(g)) ≈ σ(g)²(1-σ(g))² * Var(g)
        alpha_std = sigma * (1 - sigma) * np.sqrt(var_logit)

        return alpha_pred, alpha_std

    def get_coefficients(self) -> np.ndarray:
        """Return fitted coefficients."""
        if not self.is_fitted_:
            raise ValueError("Model not fitted.")
        return self.c_.copy()

    def get_history(self) -> dict:
        """Return optimization history."""
        if self.history_ is None:
            return {}
        return self.history_.copy()


def create_global_regressor(
    kernel_type: Literal['euclidean', 'graph'] = 'euclidean',
    **kwargs
) -> GlobalFDRRegressor:
    """
    Factory function to create GlobalFDRRegressor with sensible defaults.

    Parameters
    ----------
    kernel_type : {'euclidean', 'graph'}
        Type of kernel being used (affects default hyperparameters)
    **kwargs
        Override default parameters

    Returns
    -------
    GlobalFDRRegressor
        Configured regressor instance
    """
    # Default configurations based on kernel type
    defaults = {
        'euclidean': {
            'lambda_global': 0.1,
            'learning_rate': 0.005,
            'max_iter': 2000,
        },
        'graph': {
            'lambda_global': 0.5,
            'learning_rate': 0.005,
            'max_iter': 2000,
        }
    }

    config = defaults.get(kernel_type, defaults['euclidean'])
    config.update(kwargs)

    return GlobalFDRRegressor(**config)
