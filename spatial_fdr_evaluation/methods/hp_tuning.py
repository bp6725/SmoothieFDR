"""
Hyperparameter tuning for Spatial FDR via cross-validation.
"""

import numpy as np
from sklearn.model_selection import KFold
from itertools import product
from typing import Dict, List, Tuple, Optional, Callable
import pandas as pd


def cross_validate_hp(
        locations: np.ndarray,
        p_values: np.ndarray,
        f0_func: Callable,
        f1_func: Callable,
        K_func: Callable,
        lambda_reg: float,
        lambda_bound: float,
        learning_rate: float,
        max_grad_norm: float,
        n_folds: int = 5,
        max_iter: int = 1000,
        verbose: bool = False
) -> Tuple[float, float, float]:
    """
    Perform K-fold cross-validation for given hyperparameters.

    Parameters
    ----------
    locations : np.ndarray
        Spatial locations
    p_values : np.ndarray
        P-values
    f0_func : callable
        Null density function f₀(p)
    f1_func : callable
        Alternative density function f₁(p)
    K_func : callable
        Kernel function K(X1, X2)
    lambda_reg : float
        RKHS regularization parameter
    lambda_bound : float
        Boundary penalty parameter
    learning_rate : float
        Learning rate for optimization
    max_grad_norm : float
        Gradient clipping threshold
    n_folds : int, default=5
        Number of CV folds
    max_iter : int, default=1000
        Maximum optimization iterations
    verbose : bool, default=False
        Print progress

    Returns
    -------
    mean_loglik : float
        Mean test log-likelihood across folds
    std_loglik : float
        Standard deviation of test log-likelihood
    mean_violations : float
        Mean number of constraint violations
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_logliks = []
    fold_violations = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(locations)):
        # Split data
        loc_train = locations[train_idx]
        loc_test = locations[test_idx]
        p_train = p_values[train_idx]
        p_test = p_values[test_idx]

        # Compute kernel matrices
        K_train = K_func(loc_train, loc_train)
        K_test_train = K_func(loc_test, loc_train)

        # Evaluate densities
        f0_train = f0_func(p_train)
        f1_train = f1_func(p_train)
        f0_test = f0_func(p_test)
        f1_test = f1_func(p_test)

        # Train on fold
        c_train = _optimize_fold(
            K_train, f0_train, f1_train,
            lambda_reg, lambda_bound, learning_rate,
            max_grad_norm, max_iter
        )

        # Evaluate on held-out fold
        test_loglik, n_viol = _evaluate_fold(
            c_train, K_test_train, f0_test, f1_test
        )

        fold_logliks.append(test_loglik)
        fold_violations.append(n_viol)

        if verbose:
            print(f"  Fold {fold_idx + 1}/{n_folds}: "
                  f"test loglik={test_loglik:.2f}, viol={n_viol}")

    mean_loglik = np.mean(fold_logliks)
    std_loglik = np.std(fold_logliks)
    mean_violations = np.mean(fold_violations)

    return mean_loglik, std_loglik, mean_violations


def _optimize_fold(
        K: np.ndarray,
        f0_vals: np.ndarray,
        f1_vals: np.ndarray,
        lambda_reg: float,
        lambda_bound: float,
        learning_rate: float,
        max_grad_norm: float,
        max_iter: int
) -> np.ndarray:
    """
    Optimize on a single fold.

    Returns trained coefficients c.
    """
    n = len(f0_vals)
    c = np.zeros(n)

    for iteration in range(max_iter):
        # Forward pass
        alpha = K @ c
        mixture = alpha * f0_vals + (1 - alpha) * f1_vals
        mixture = np.clip(mixture, 1e-10, None)

        # Natural gradient
        grad_alpha = -(f0_vals - f1_vals) / mixture
        grad_bound = 2 * lambda_bound * (
                np.maximum(0, alpha - 1) - np.maximum(0, -alpha)
        )
        grad_nat = grad_alpha + grad_bound + 2 * lambda_reg * c

        # Gradient clipping (CRITICAL!)
        grad_norm = np.linalg.norm(grad_nat)
        if grad_norm > max_grad_norm:
            grad_nat = grad_nat * (max_grad_norm / grad_norm)

        # Update
        c = c - learning_rate * grad_nat

    return c


def _evaluate_fold(
        c_train: np.ndarray,
        K_test_train: np.ndarray,
        f0_test: np.ndarray,
        f1_test: np.ndarray
) -> Tuple[float, int]:
    """
    Evaluate trained model on held-out fold.

    Returns test log-likelihood and number of violations.
    """
    # Predict on test set
    alpha_test = K_test_train @ c_train

    # Compute mixture
    mixture_test = alpha_test * f0_test + (1 - alpha_test) * f1_test
    mixture_test = np.clip(mixture_test, 1e-10, None)

    # Test log-likelihood
    test_loglik = np.sum(np.log(mixture_test))

    # Violations
    n_violations = np.sum((alpha_test < 0) | (alpha_test > 1))

    return test_loglik, n_violations


def grid_search_hp(
        locations: np.ndarray,
        p_values: np.ndarray,
        f0_func: Callable,
        f1_func: Callable,
        K_func: Callable,
        hp_grid: Dict[str, List],
        n_folds: int = 5,
        max_iter: int = 1000,
        max_violations: float = 5.0,
        verbose: bool = True
) -> Tuple[Dict, pd.DataFrame]:
    """
    Grid search over hyperparameters with K-fold CV.

    Parameters
    ----------
    locations : np.ndarray
        Spatial locations
    p_values : np.ndarray
        P-values
    f0_func : callable
        Null density function
    f1_func : callable
        Alternative density function
    K_func : callable
        Kernel function
    hp_grid : dict
        Hyperparameter grid with keys:
        - 'lambda_reg': list of regularization values
        - 'lambda_bound': list of boundary penalty values
        - 'learning_rate': list of learning rates
        - 'max_grad_norm': list of gradient clipping thresholds
    n_folds : int, default=5
        Number of CV folds
    max_iter : int, default=1000
        Maximum iterations per fold
    max_violations : float, default=5.0
        Maximum acceptable mean violations
    verbose : bool, default=True
        Print progress

    Returns
    -------
    best_hp : dict
        Best hyperparameters
    results_df : pd.DataFrame
        All CV results
    """
    if verbose:
        print("=" * 70)
        print("GRID SEARCH WITH K-FOLD CROSS-VALIDATION")
        print("=" * 70)
        print(f"\nSearch space:")
        for key, values in hp_grid.items():
            print(f"  {key:15s}: {values}")

    # Generate all combinations
    combinations = list(product(
        hp_grid['lambda_reg'],
        hp_grid['lambda_bound'],
        hp_grid['learning_rate'],
        hp_grid['max_grad_norm']
    ))

    if verbose:
        print(f"\nTotal combinations: {len(combinations)}")
        print(f"K-fold: {n_folds}")
        print(f"Total training runs: {len(combinations) * n_folds}\n")

    results = []

    for i, (lr, lb, lrate, mgn) in enumerate(combinations):
        if verbose:
            print(f"[{i + 1:3d}/{len(combinations)}] "
                  f"λ_reg={lr:.3f}, λ_bound={lb:5.1f}, "
                  f"lr={lrate:.3f}, clip={mgn:4.1f}...", end=" ")

        try:
            mean_loglik, std_loglik, mean_viol = cross_validate_hp(
                locations, p_values, f0_func, f1_func, K_func,
                lambda_reg=lr,
                lambda_bound=lb,
                learning_rate=lrate,
                max_grad_norm=mgn,
                n_folds=n_folds,
                max_iter=max_iter,
                verbose=False
            )

            result = {
                'lambda_reg': lr,
                'lambda_bound': lb,
                'learning_rate': lrate,
                'max_grad_norm': mgn,
                'mean_cv_loglik': mean_loglik,
                'std_cv_loglik': std_loglik,
                'mean_violations': mean_viol
            }
            results.append(result)

            if verbose:
                print(f"CV loglik: {mean_loglik:7.2f} ± {std_loglik:5.2f}, "
                      f"viol: {mean_viol:4.1f}")

        except Exception as e:
            if verbose:
                print(f"FAILED: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Find best HP
    df_sorted = df.sort_values('mean_cv_loglik', ascending=False)

    # Best with violation constraint
    df_valid = df[df['mean_violations'] <= max_violations].copy()
    if len(df_valid) > 0:
        df_valid_sorted = df_valid.sort_values('mean_cv_loglik', ascending=False)
        best_hp = df_valid_sorted.iloc[0].to_dict()
    else:
        # No valid configs, use best overall
        best_hp = df_sorted.iloc[0].to_dict()
        if verbose:
            print("\n⚠️  No configurations with acceptable violations, using best overall")

    if verbose:
        print("\n" + "=" * 70)
        print("🏆 BEST HYPERPARAMETERS (by cross-validation)")
        print("=" * 70)
        print(f"lambda_reg:       {best_hp['lambda_reg']:.4f}")
        print(f"lambda_bound:     {best_hp['lambda_bound']:.1f}")
        print(f"learning_rate:    {best_hp['learning_rate']:.4f}")
        print(f"max_grad_norm:    {best_hp['max_grad_norm']:.1f}")
        print(f"\nCV Performance:")
        print(f"  Mean CV log-lik:  {best_hp['mean_cv_loglik']:.3f} ± "
              f"{best_hp['std_cv_loglik']:.3f}")
        print(f"  Mean violations:  {best_hp['mean_violations']:.1f}")

    return best_hp, df
