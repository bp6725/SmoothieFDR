"""
Spatial FDR control via RKHS regularization.

Implements the two-stage procedure from the paper:
1. Point-wise solution: Solve at observed locations (MAIN METHOD)
2. Entire-domain solution: Extend to unobserved locations (OPTIONAL)
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta as beta_dist
from typing import Tuple, Dict, Literal, Optional, Callable
from .kernels import compute_kernel_matrix, add_regularization
from .baseline import benjamini_hochberg
import numpy as np
from scipy import stats
from sklearn.neighbors import KernelDensity
from typing import Tuple, Callable

class SpatialFDR:
    """
    Spatial FDR control using RKHS regularization.

    Implements the two-stage procedure:
    1. Point-wise: Solve at observed locations using p-value likelihood
    2. Entire-domain: Optionally extend to unobserved locations via kernel logistic regression

    Parameters
    ----------
    kernel_type : {'matern', 'rbf'}, default='matern'
        Type of kernel to use
    lambda_reg : float, default=0.1
        RKHS regularization parameter
    lambda_bound : float, default=10.0
        Boundary penalty parameter (keeps α ∈ [0,1])
    kernel_params : dict, optional
        Parameters for the kernel (e.g., nu, length_scale)
    optimizer : {'natural_gradient', 'lbfgs'}, default='natural_gradient'
        Optimization method
    max_iter : int, default=1000
        Maximum optimization iterations
    tol : float, default=1e-6
        Convergence tolerance
    verbose : bool, default=False
        Print optimization progress
    """

    def __init__(
        self,
        kernel_type: Literal['matern', 'rbf'] = 'matern',
        lambda_reg: float = 0.05,
        lambda_bound: float = 500.0,
        kernel_params: Optional[Dict] = None,
        optimizer: Literal['natural_gradient', 'lbfgs'] = 'natural_gradient',
        max_iter: int = 1000,
        tol: float = 1e-6,
        max_grad_norm: float = 10.0,
        verbose: bool = False
    ):
        self.kernel_type = kernel_type
        self.lambda_reg = lambda_reg
        self.lambda_bound = lambda_bound
        self.kernel_params = kernel_params or {}
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.max_grad_norm = max_grad_norm
        self.tol = tol
        self.verbose = verbose

        # Will be set during fit
        self.locations_ = None
        self.K_ = None
        self.c_ = None  # Coefficients for point-wise
        self.c_full_ = None  # Coefficients for entire-domain (optional)
        self.alpha_pointwise_ = None  # α at observed locations
        self.pi0_ = None  # Estimated null proportion
        self.f0_ = None  # Null density function
        self.f1_ = None  # Alternative density function

    import numpy as np
    from scipy import stats
    from sklearn.neighbors import KernelDensity
    from typing import Tuple, Callable

    def _estimate_marginal_mixture_robust(self, p_values: np.ndarray) -> Tuple[float, Callable, Callable]:
        """
        Estimates marginal f0 and f1 distributions robustly.

        Strategy:
        1. Acknowledge user's proof: Marginal f(p) = pi0*f0(p) + (1-pi0)*f1(p).
        2. Acknowledge Efron (Sec 4): Real spatial data often has a 'dilated' null
           due to unobserved spatial covariates/correlations.
        3. Solution: 'Right-Tail Anchoring'. We estimate f0 properties strictly
           from the high p-values (which are almost certainly null), allowing for
           dilation (sigma > 1) if the data supports it.

        Returns:
            pi0_est: Estimated proportion of nulls.
            f0: Callable density for the Null (handles dilation).
            f1: Callable density for the Alternative (strictly low-p).
        """

        # 1. Clean and Transform to Z-space
        # Working in Z-space is numerically stable for detecting 'dilation'.
        p_clean = p_values[(~np.isnan(p_values)) & (p_values > 0) & (p_values < 1)]
        # Clip to avoid infs
        p_clean = np.clip(p_clean, 1e-15, 1 - 1e-15)
        z_scores = stats.norm.ppf(p_clean)

        if len(z_scores) < 100:
            # Fallback for tiny data: assume theoretical null
            return 0.9, lambda p: np.ones_like(p), lambda p: 2 * (1 - p)

        # 2. Fit Global Density f(z) using KDE
        # This captures the actual marginal shape of the data.
        n = len(z_scores)
        std_dev = np.std(z_scores, ddof=1)
        bw_scott = 1.06 * std_dev * (n ** (-0.2))

        kde = KernelDensity(bandwidth=bw_scott, kernel='gaussian')
        kde.fit(z_scores.reshape(-1, 1))

        def f_z_global(z_in):
            return np.exp(kde.score_samples(np.atleast_1d(z_in).reshape(-1, 1)))

        # 3. Estimate Empirical Null Parameters (Right-Tail Anchoring)
        # We assume the right half of the distribution (z > median) is Pure Null.
        # If the null is theoretical, this right half matches N(0,1).
        # If the null is dilated (spatial noise), it matches N(0, sigma^2).

        median_z = np.median(z_scores)
        q75_z = np.percentile(z_scores, 75)

        # Robust sigma estimate using IQR of the right tail only
        # For N(0,1), distance from median to Q3 is ~0.6745
        sigma_est = (q75_z - median_z) / 0.6745

        # Force sigma >= 1.0.
        # Logic: Spatial correlation usually adds variance (dilation).
        # It rarely reduces variance below theoretical random noise.
        sigma_est = max(1.0, sigma_est)

        # We assume the null is centered at the median (or 0 if you prefer strict centering)
        # Using median is safer if f1 is small (<20%).
        delta_est = median_z

        if self.verbose:
            print(f"Marginal Estimation:")
            print(f"  Empirical Null Sigma: {sigma_est:.3f} (Theoretical=1.0)")
            if sigma_est > 1.1:
                print("  -> Detected spatial correlation/overdispersion in Null.")

        # 4. Define P-space Densities (with Jacobian)

        def get_jacobian(z):
            # Jacobian |dz/dp| = 1 / phi(z)
            return 1.0 / stats.norm.pdf(z)

        def f0(p):
            """
            The Null density in p-space.
            If sigma_est > 1, this will be U-shaped (Efron's Empirical Null).
            If sigma_est = 1, this will be Flat (Theoretical Null).
            """
            p_arr = np.atleast_1d(p)
            p_safe = np.clip(p_arr, 1e-15, 1 - 1e-15)
            z = stats.norm.ppf(p_safe)

            # Null density in Z
            pdf_z = stats.norm.pdf(z, loc=delta_est, scale=sigma_est)

            # Transform to P
            return pdf_z * get_jacobian(z)

        def f1(p):
            """
            The Alternative density.
            Defined as Positive Residual: max(0, f_global - pi0*f0).
            Strictly zero for p > 0.5 (one-sided constraint).
            """
            p_arr = np.atleast_1d(p)
            p_safe = np.clip(p_arr, 1e-15, 1 - 1e-15)
            z = stats.norm.ppf(p_safe)

            # 1. Global density
            pdf_global = f_z_global(z)

            # 2. Null density (unweighted)
            pdf_null = stats.norm.pdf(z, loc=delta_est, scale=sigma_est)

            # 3. Calculate Residual
            # We need a temporary estimate of pi0 here to define the shape of f1.
            # Conservative check: how much mass is in the right tail?
            # In right tail, f_global should approx equal pi0 * pdf_null
            ratio = pdf_global / (pdf_null + 1e-10)
            # pi0 is roughly the minimum of this ratio in the null region
            pi0_temp = np.min(ratio[z > median_z]) if np.any(z > median_z) else 1.0
            pi0_temp = np.clip(pi0_temp, 0.5, 1.0)

            excess = pdf_global - (pi0_temp * pdf_null)

            # Constraint: Signal is strictly on the left (z < delta)
            excess[z > delta_est] = 0
            excess = np.maximum(0, excess)

            # Transform to P
            return (excess * get_jacobian(z)) / (1 - pi0_temp + 1e-6)

        # 5. Final Pi0 Estimate (Storey's method adapted for general f0)
        # Standard Storey: pi0 = #{p > lambda} / ((1-lambda)*N)
        # Generalized: pi0 = #{p > lambda} / (Integral_{lambda}^1 f0(p) dp * N)

        lambda_val = 0.5
        num_null_region = np.sum(p_clean > lambda_val)

        # How much probability mass does our fitted f0 have in [0.5, 1.0]?
        # In Z-space, this corresponds to mass in [0, inf) (assuming delta=0)
        # For N(0, sigma), mass > 0 is exactly 0.5.
        mass_f0_in_region = 0.5

        pi0_estimate = (num_null_region / len(p_clean)) / mass_f0_in_region
        pi0_estimate = np.clip(pi0_estimate, 0.1, 1.0)

        return pi0_estimate, f0, f1

    def _pointwise_loss(
        self,
        c: np.ndarray,
        K: np.ndarray,
        p_values: np.ndarray,
        f0_vals: np.ndarray,
        f1_vals: np.ndarray
    ) -> float:
        """
        Point-wise objective function (MAIN METHOD).

        L(c) = -Σ log[α(loc_i) f₀(p_i) + (1-α(loc_i)) f₁(p_i)]
               + λ_reg ||α||²_H
               + λ_bound Σ[max(0, α_i-1)² + max(0, -α_i)²]

        where α(loc_i) = (Kc)_i
        """
        # α(loc) = Kc (linear, not sigmoid!)
        alpha = K @ c

        # P-value likelihood: mixture density
        mixture = alpha * f0_vals + (1 - alpha) * f1_vals

        # Clip for numerical stability
        mixture = np.clip(mixture, 1e-10, None)

        # Negative log-likelihood
        neg_log_lik = -np.sum(np.log(mixture))

        # RKHS regularization
        rkhs_penalty = self.lambda_reg * (c @ K @ c)

        # Boundary penalty to enforce α ∈ [0,1]
        boundary_penalty = self.lambda_bound * np.sum(
            np.maximum(0, alpha - 1)**2 +
            np.maximum(0, -alpha)**2
        )

        return neg_log_lik + rkhs_penalty + boundary_penalty

    def _pointwise_gradient(
        self,
        c: np.ndarray,
        K: np.ndarray,
        p_values: np.ndarray,
        f0_vals: np.ndarray,
        f1_vals: np.ndarray
    ) -> np.ndarray:
        """
        Gradient of point-wise objective.

        Natural gradient:
        ∇L = w + 2λ_reg c + 2λ_bound ∇_boundary

        where w_i = -(f₀(p_i) - f₁(p_i)) / mixture_i
        """
        # α(loc) = Kc
        alpha = K @ c

        # Mixture density
        mixture = alpha * f0_vals + (1 - alpha) * f1_vals
        mixture = np.clip(mixture, 1e-10, None)

        # Data term gradient
        w = -(f0_vals - f1_vals) / mixture

        # RKHS regularization gradient
        rkhs_grad = 2 * self.lambda_reg * c

        # Boundary penalty gradient
        boundary_grad = 2 * self.lambda_bound * (
            np.maximum(0, alpha - 1) -
            np.maximum(0, -alpha)
        )

        # Natural gradient (sum of components)
        grad = w + rkhs_grad + boundary_grad

        return grad

    def _natural_gradient_descent_pointwise(
        self,
        K: np.ndarray,
        p_values: np.ndarray,
        f0_vals: np.ndarray,
        f1_vals: np.ndarray,
        c_init: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Natural gradient descent for point-wise optimization.
        """
        N = len(p_values)

        if c_init is None:
            c = (np.random.rand(N)-0.5)*2
        else:
            c = c_init.copy()

        # Adaptive learning rate
        learning_rate = 0.01

        for iteration in range(self.max_iter):
            # Compute gradient
            grad = self._pointwise_gradient(c, K, p_values, f0_vals, f1_vals)

            # Gradient clipping (CRITICAL for stability!)
            grad_norm = np.linalg.norm(grad)
            if grad_norm > self.max_grad_norm:
                grad = grad * (self.max_grad_norm / grad_norm)

            # Update
            c_new = c - learning_rate * grad

            # # Check convergence
            # grad_norm = np.linalg.norm(grad)
            #
            # if grad_norm < self.tol:
            #     if self.verbose:
            #         print(f"Converged in {iteration} iterations")
            #     break

            # Adaptive step size
            loss_old = self._pointwise_loss(c, K, p_values, f0_vals, f1_vals)
            loss_new = self._pointwise_loss(c_new, K, p_values, f0_vals, f1_vals)

            if loss_new < loss_old:
                c = c_new
                learning_rate = min(learning_rate * 1.1, 0.1)  # Increase but cap
            else:
                learning_rate *= 0.5  # Decrease
                learning_rate = max(learning_rate, 1e-6)  # Don't go too small

            if self.verbose and iteration % 100 == 0:
                alpha = K @ c
                violations = np.sum((alpha < 0) | (alpha > 1))
                print(f"Iter {iteration}: loss = {loss_old:.6f}, "
                      f"|grad| = {grad_norm:.2e}, "
                      f"violations = {violations}, "
                      f"α range = [{alpha.min():.3f}, {alpha.max():.3f}]")

        return c

    def fit_pointwise(
        self,
        locations: np.ndarray,
        p_values: np.ndarray
    ) -> 'SpatialFDR':
        """
        Fit the point-wise model (MAIN METHOD).

        This solves the optimization problem at observed locations:

        min -Σ log[α(loc_i) f₀(p_i) + (1-α(loc_i)) f₁(p_i)]
            + λ_reg ||α||²_H + λ_bound * boundary_penalty

        Parameters
        ----------
        locations : np.ndarray, shape (N, d)
            Spatial locations
        p_values : np.ndarray, shape (N,)
            P-values at those locations

        Returns
        -------
        self : SpatialFDR
            Fitted model
        """
        N = len(p_values)

        if len(locations) != N:
            raise ValueError(f"locations and p_values must have same length")

        # Store locations
        self.locations_ = locations

        # Estimate f₀ and f₁ from marginal
        self.pi0_, self.f0_, self.f1_ = self._estimate_marginal_mixture_robust(p_values)

        # Evaluate densities at observed p-values
        f0_vals = self.f0_(p_values)
        f1_vals = self.f1_(p_values)

        # Compute kernel matrix
        self.K_ = compute_kernel_matrix(
            locations,
            kernel_type=self.kernel_type,
            **self.kernel_params
        )

        # Add small regularization for numerical stability
        self.K_ = add_regularization(self.K_, epsilon=1e-6)

        # Optimize
        if self.optimizer == 'natural_gradient':
            self.c_ = self._natural_gradient_descent_pointwise(
                self.K_, p_values, f0_vals, f1_vals
            )
        elif self.optimizer == 'lbfgs':
            # L-BFGS optimization
            result = minimize(
                fun=lambda c: self._pointwise_loss(c, self.K_, p_values, f0_vals, f1_vals),
                x0=np.zeros(N),
                jac=lambda c: self.K_ @ self._pointwise_gradient(
                    c, self.K_, p_values, f0_vals, f1_vals
                ),
                method='L-BFGS-B',
                options={'maxiter': self.max_iter, 'ftol': self.tol}
            )
            self.c_ = result.x

            if self.verbose:
                print(f"L-BFGS-B: {result.message}")
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        # Store α at observed locations
        self.alpha_pointwise_ = self.K_ @ self.c_

        if self.verbose:
            print(f"\nFinal α range: [{self.alpha_pointwise_.min():.3f}, "
                  f"{self.alpha_pointwise_.max():.3f}]")
            violations = np.sum((self.alpha_pointwise_ < 0) | (self.alpha_pointwise_ > 1))
            print(f"Constraint violations: {violations}/{N}")

        return self

    def _kernel_logistic_loss(
        self,
        c: np.ndarray,
        K: np.ndarray,
        alpha_target: np.ndarray
    ) -> float:
        """
        Kernel logistic regression loss (for entire-domain extension).

        L(c) = -Σ[α* log(σ(Kc)) + (1-α*) log(1-σ(Kc))] + λ c^T K c

        where σ(z) = 1/(1 + exp(-z))
        """
        # Linear predictor
        g = K @ c

        # Sigmoid
        sigma = 1 / (1 + np.exp(-np.clip(g, -500, 500)))
        sigma = np.clip(sigma, 1e-10, 1 - 1e-10)

        # Cross-entropy loss
        log_loss = -np.sum(
            alpha_target * np.log(sigma) +
            (1 - alpha_target) * np.log(1 - sigma)
        )

        # RKHS regularization
        rkhs_penalty = self.lambda_reg * (c @ K @ c)

        return log_loss + rkhs_penalty

    def _kernel_logistic_gradient(
        self,
        c: np.ndarray,
        K: np.ndarray,
        alpha_target: np.ndarray
    ) -> np.ndarray:
        """
        Natural gradient for kernel logistic regression.

        ∇L = (σ - α*) + 2λc
        """
        # Linear predictor
        g = K @ c

        # Sigmoid
        sigma = 1 / (1 + np.exp(-np.clip(g, -500, 500)))

        # Natural gradient
        grad = (sigma - alpha_target) + 2 * self.lambda_reg * c

        return grad

    def fit_entire_domain(
        self,
        lambda_reg_full: Optional[float] = None
    ) -> 'SpatialFDR':
        """
        Fit entire-domain extension (OPTIONAL - for visualization only).

        Uses α from fit_pointwise() as targets for kernel logistic regression.
        This ensures α ∈ [0,1] everywhere via sigmoid squashing.

        Parameters
        ----------
        lambda_reg_full : float, optional
            Regularization for entire-domain fit. If None, uses self.lambda_reg.
            Typically set lower than point-wise to closely follow α*.

        Returns
        -------
        self : SpatialFDR
            Model with entire-domain fit
        """
        # Must run fit_pointwise first
        if self.alpha_pointwise_ is None:
            raise ValueError("Run fit_pointwise() first!")

        # Use α from Stage 1 as targets (clip for numerical stability)
        alpha_target = np.clip(self.alpha_pointwise_, 0.01, 0.99)

        # Use separate regularization if provided
        if lambda_reg_full is not None:
            lambda_orig = self.lambda_reg
            self.lambda_reg = lambda_reg_full

        # Initialize with inverse logit transform
        logit_alpha = np.log(alpha_target / (1 - alpha_target))
        try:
            c_init = np.linalg.solve(self.K_, logit_alpha)
        except:
            c_init = np.zeros(len(alpha_target))

        # Optimize
        N = len(alpha_target)
        learning_rate = 0.1
        c = c_init.copy()

        for iteration in range(self.max_iter):
            grad = self._kernel_logistic_gradient(c, self.K_, alpha_target)

            c_new = c - learning_rate * grad

            grad_norm = np.linalg.norm(grad)
            if grad_norm < self.tol:
                if self.verbose:
                    print(f"Entire-domain converged in {iteration} iterations")
                break

            # Adaptive step size
            loss_old = self._kernel_logistic_loss(c, self.K_, alpha_target)
            loss_new = self._kernel_logistic_loss(c_new, self.K_, alpha_target)

            if loss_new < loss_old:
                c = c_new
                learning_rate = min(learning_rate * 1.1, 0.5)
            else:
                learning_rate *= 0.5
                learning_rate = max(learning_rate, 1e-6)

            if self.verbose and iteration % 100 == 0:
                print(f"Iter {iteration}: loss = {loss_old:.6f}, |grad| = {grad_norm:.2e}")

        self.c_full_ = c

        # Restore original regularization
        if lambda_reg_full is not None:
            self.lambda_reg = lambda_orig

        if self.verbose:
            g = self.K_ @ self.c_full_
            alpha_full = 1 / (1 + np.exp(-g))
            print(f"Entire-domain α range: [{alpha_full.min():.3f}, {alpha_full.max():.3f}]")

        return self

    def fit(
        self,
        locations: np.ndarray,
        p_values: np.ndarray,
        fit_full: bool = False,
        lambda_reg_full: Optional[float] = None
    ) -> 'SpatialFDR':
        """
        Convenience method to fit the model.

        Parameters
        ----------
        locations : np.ndarray
            Spatial locations
        p_values : np.ndarray
            P-values
        fit_full : bool, default=False
            If True, also fit entire-domain extension
        lambda_reg_full : float, optional
            Regularization for entire-domain (if fit_full=True)

        Returns
        -------
        self : SpatialFDR
            Fitted model
        """
        # Always fit point-wise (main method)
        self.fit_pointwise(locations, p_values)

        # Optionally fit entire-domain
        if fit_full:
            self.fit_entire_domain(lambda_reg_full)

        return self

    def predict_alpha(
        self,
        locations_new: Optional[np.ndarray] = None,
        use_full: bool = False
    ) -> np.ndarray:
        """
        Predict α(loc) at locations.

        Parameters
        ----------
        locations_new : np.ndarray, optional
            New locations. If None, uses training locations.
        use_full : bool, default=False
            If True, use entire-domain model (must have called fit_entire_domain())

        Returns
        -------
        alpha : np.ndarray
            Predicted prior null probabilities
        """
        if self.c_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if use_full and self.c_full_ is None:
            raise ValueError("Entire-domain model not fitted. Call fit_entire_domain() first.")

        # Choose coefficients
        c = self.c_full_ if use_full else self.c_

        if locations_new is None:
            # Predict at training locations
            if use_full:
                g = self.K_ @ c
                alpha = 1 / (1 + np.exp(-g))  # Sigmoid for full model
            else:
                alpha = self.K_ @ c  # Linear for point-wise
        else:
            # Predict at new locations
            from .kernels import matern_kernel, rbf_kernel
            if self.kernel_type == 'matern':
                K_cross = matern_kernel(locations_new, self.locations_, **self.kernel_params)
            else:
                K_cross = rbf_kernel(locations_new, self.locations_, **self.kernel_params)

            if use_full:
                g = K_cross @ c
                alpha = 1 / (1 + np.exp(-g))
            else:
                alpha = K_cross @ c

        # Clip for safety (point-wise may have small violations)
        if not use_full:
            alpha = np.clip(alpha, 0, 1)

        return alpha

    def compute_local_fdr(
        self,
        p_values: np.ndarray,
        locations: Optional[np.ndarray] = None,
        use_full: bool = False
    ) -> np.ndarray:
        """
        Compute local FDR at given p-values and locations.

        lfdr(p, loc) = α(loc) f₀(p) / [α(loc) f₀(p) + (1-α(loc)) f₁(p)]

        Parameters
        ----------
        p_values : np.ndarray
            P-values
        locations : np.ndarray, optional
            Locations. If None, uses training locations.
        use_full : bool, default=False
            Use entire-domain model

        Returns
        -------
        lfdr : np.ndarray
            Local false discovery rates
        """
        # Get α at locations
        alpha = self.predict_alpha(locations, use_full=use_full)

        # Evaluate densities
        f0_vals = self.f0_(p_values)
        f1_vals = self.f1_(p_values)

        # Local FDR
        numerator = alpha * f0_vals
        denominator = alpha * f0_vals + (1 - alpha) * f1_vals

        lfdr = numerator / np.clip(denominator, 1e-10, None)

        return np.clip(lfdr, 0, 1)

    def reject(
        self,
        p_values: np.ndarray,
        fdr_level: float = 0.1,
        locations: Optional[np.ndarray] = None,
        use_full: bool = False
    ) -> np.ndarray:
        """
        Make rejection decisions at target FDR level.

        Rejects hypotheses where local FDR ≤ fdr_level.

        Parameters
        ----------
        p_values : np.ndarray
            P-values
        fdr_level : float, default=0.1
            Target FDR level
        locations : np.ndarray, optional
            Locations. If None, uses training locations.
        use_full : bool, default=False
            Use entire-domain model

        Returns
        -------
        discoveries : np.ndarray, dtype=bool
            Boolean array indicating rejections
        """
        # Compute local FDR
        lfdr = self.compute_local_fdr(p_values, locations, use_full)

        print(f"lfdr range: [{lfdr.min():.3f}, {lfdr.max():.3f}]")
        print(f"lfdr mean: {lfdr.mean():.3f}")
        print(f"lfdr <= 0.1: {np.sum(lfdr <= 0.1)}")

        # Reject where local FDR is below threshold
        discoveries = lfdr <= fdr_level

        return discoveries
