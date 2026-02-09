"""
SpectralFM v2: Classical Calibration Transfer Baselines

Implements classical methods for comparison:
- PLS: Partial Least Squares (base calibration model)
- PDS: Piecewise Direct Standardization (Wang et al. 1991)
- SBC: Slope/Bias Correction
- DS: Direct Standardization

Reference: Wang, Y., Veltkamp, D.J., Kowalski, B.R. (1991).
           Multivariate instrument standardization.
           Analytical Chemistry, 63(23), 2750-2756.
"""
import numpy as np
from typing import Dict, Optional, Tuple, List
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score
from scipy import stats
import warnings


class PLSCalibration:
    """PLS regression calibration model.

    For speed, uses a fixed number of components. Set n_components=None
    to use a reasonable default based on training set size.
    """

    def __init__(self, n_components: int = None, max_components: int = 15):
        self.n_components = n_components
        self.max_components = max_components
        self.model = None
        self.optimal_n_components = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit PLS model."""
        if self.n_components is None:
            # Use a reasonable default: min(15, n_samples-1, n_features)
            self.optimal_n_components = min(self.max_components, X.shape[0] - 1, X.shape[1])
        else:
            self.optimal_n_components = min(self.n_components, X.shape[0] - 1, X.shape[1])

        self.model = PLSRegression(n_components=self.optimal_n_components)
        self.model.fit(X, y.reshape(-1, 1) if y.ndim == 1 else y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using fitted model."""
        return self.model.predict(X).flatten()

    def fit_predict(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray) -> np.ndarray:
        """Fit and predict in one step."""
        self.fit(X_train, y_train)
        return self.predict(X_test)


class PDS:
    """Piecewise Direct Standardization (Wang et al. 1991).

    Maps each wavelength in the secondary instrument to a window of
    wavelengths in the primary instrument using local regression.

    The transfer matrix F transforms secondary spectra to match primary:
        X_primary_predicted = X_secondary @ F

    Parameters:
        window_size: Number of wavelengths in the local window (default: 11)
                    Wang et al. recommend window of +/- 5 wavelengths
        regularization: Ridge regularization parameter (default: 1e-4)
    """

    def __init__(self, window_size: int = 11, regularization: float = 1e-4):
        self.window_size = window_size
        self.regularization = regularization
        self.F = None

    def fit(self, X_primary: np.ndarray, X_secondary: np.ndarray):
        """Fit PDS transfer matrix.

        Args:
            X_primary: (N, P) spectra from primary/master instrument
            X_secondary: (N, P) spectra from secondary instrument (same samples)
        """
        N, P = X_primary.shape
        half_w = self.window_size // 2

        self.F = np.zeros((P, P))

        for j in range(P):
            # Window around wavelength j
            start = max(0, j - half_w)
            end = min(P, j + half_w + 1)

            # Local regression: predict X_primary[:, j] from X_secondary[:, start:end]
            X_local = X_secondary[:, start:end]
            y_local = X_primary[:, j]

            # Ridge regression for numerical stability
            XtX = X_local.T @ X_local + self.regularization * np.eye(end - start)
            Xty = X_local.T @ y_local
            coeffs = np.linalg.solve(XtX, Xty)

            self.F[start:end, j] = coeffs

        return self

    def transform(self, X_secondary: np.ndarray) -> np.ndarray:
        """Transform secondary spectra to primary instrument space."""
        if self.F is None:
            raise ValueError("PDS not fitted. Call fit() first.")
        return X_secondary @ self.F

    def fit_transform(self, X_primary_train: np.ndarray,
                      X_secondary_train: np.ndarray,
                      X_secondary_test: np.ndarray) -> np.ndarray:
        """Fit on transfer samples and transform test spectra."""
        self.fit(X_primary_train, X_secondary_train)
        return self.transform(X_secondary_test)


class SBC:
    """Slope/Bias Correction.

    Simple linear correction applied independently to each wavelength:
        X_corrected[:, i] = slope[i] * X_secondary[:, i] + bias[i]

    This is equivalent to DS with window_size=1.
    """

    def __init__(self):
        self.slopes = None
        self.intercepts = None

    def fit(self, X_primary: np.ndarray, X_secondary: np.ndarray):
        """Fit slope/bias for each wavelength."""
        N, P = X_primary.shape
        self.slopes = np.zeros(P)
        self.intercepts = np.zeros(P)

        for i in range(P):
            if np.std(X_secondary[:, i]) < 1e-10:
                # Constant wavelength, just use bias
                self.slopes[i] = 1.0
                self.intercepts[i] = np.mean(X_primary[:, i]) - np.mean(X_secondary[:, i])
            else:
                slope, intercept, _, _, _ = stats.linregress(
                    X_secondary[:, i], X_primary[:, i]
                )
                self.slopes[i] = slope
                self.intercepts[i] = intercept

        return self

    def transform(self, X_secondary: np.ndarray) -> np.ndarray:
        """Apply slope/bias correction."""
        if self.slopes is None:
            raise ValueError("SBC not fitted. Call fit() first.")
        return X_secondary * self.slopes + self.intercepts

    def fit_transform(self, X_primary_train: np.ndarray,
                      X_secondary_train: np.ndarray,
                      X_secondary_test: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X_primary_train, X_secondary_train)
        return self.transform(X_secondary_test)


class DS:
    """Direct Standardization.

    Computes a global transfer matrix using least squares:
        X_primary = X_secondary @ F + residual
        F = (X_secondary.T @ X_secondary + lambda*I)^-1 @ X_secondary.T @ X_primary

    This is a full-rank linear transformation (unlike PDS which is sparse).

    Parameters:
        regularization: Ridge regularization parameter (default: 1e-2)
    """

    def __init__(self, regularization: float = 1e-2):
        self.regularization = regularization
        self.F = None

    def fit(self, X_primary: np.ndarray, X_secondary: np.ndarray):
        """Fit global transfer matrix."""
        N, P = X_secondary.shape

        # Ridge regression: F = (X'X + λI)^-1 X' Y
        XtX = X_secondary.T @ X_secondary + self.regularization * np.eye(P)
        XtY = X_secondary.T @ X_primary

        self.F = np.linalg.solve(XtX, XtY)

        return self

    def transform(self, X_secondary: np.ndarray) -> np.ndarray:
        """Transform secondary spectra."""
        if self.F is None:
            raise ValueError("DS not fitted. Call fit() first.")
        return X_secondary @ self.F

    def fit_transform(self, X_primary_train: np.ndarray,
                      X_secondary_train: np.ndarray,
                      X_secondary_test: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X_primary_train, X_secondary_train)
        return self.transform(X_secondary_test)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics: R², RMSEP, RPD, bias."""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)

    # RMSEP (Root Mean Square Error of Prediction)
    rmsep = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # RPD (Ratio of Performance to Deviation)
    std_true = np.std(y_true)
    rpd = std_true / (rmsep + 1e-10)

    # Bias
    bias = np.mean(y_pred - y_true)

    # Slope
    if np.std(y_true) > 1e-10:
        slope, _, _, _, _ = stats.linregress(y_true, y_pred)
    else:
        slope = 1.0

    return {
        "r2": float(r2),
        "rmsep": float(rmsep),
        "rpd": float(rpd),
        "bias": float(bias),
        "slope": float(slope),
    }


def run_baseline_comparison(
    X_source_train: np.ndarray,
    X_target_train: np.ndarray,
    X_source_test: np.ndarray,
    X_target_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    pls_components: int = None,
) -> Dict[str, Dict[str, float]]:
    """Run all baseline methods and return comparison.

    Args:
        X_source_train: (N_train, P) source instrument spectra (transfer standards)
        X_target_train: (N_train, P) target instrument spectra (same samples)
        X_source_test: (N_test, P) source instrument test spectra
        X_target_test: (N_test, P) target instrument test spectra
        y_train: (N_train,) training targets
        y_test: (N_test,) test targets
        pls_components: Number of PLS components (None = auto-select)

    Returns:
        Dict mapping method_name -> metrics dict
    """
    results = {}

    # 1. No Transfer Baseline: Train PLS on source, test on target (no correction)
    pls_no_transfer = PLSCalibration(n_components=pls_components)
    y_pred = pls_no_transfer.fit_predict(X_source_train, y_train, X_target_test)
    results["No_Transfer"] = compute_metrics(y_test, y_pred)
    results["No_Transfer"]["n_components"] = pls_no_transfer.optimal_n_components

    # 2. Upper Bound: Train PLS directly on target instrument
    pls_upper = PLSCalibration(n_components=pls_components)
    y_pred = pls_upper.fit_predict(X_target_train, y_train, X_target_test)
    results["Target_Direct"] = compute_metrics(y_test, y_pred)
    results["Target_Direct"]["n_components"] = pls_upper.optimal_n_components

    # 3. PDS + PLS
    # Learn target → source mapping using transfer standards, then correct target test data
    # PLS is trained on source, so we transform target to source space
    pds = PDS(window_size=11)  # ±5 wavelengths
    pds.fit(X_source_train, X_target_train)  # primary=source, secondary=target
    X_target_test_corrected = pds.transform(X_target_test)
    pls_pds = PLSCalibration(n_components=pls_components)
    pls_pds.fit(X_source_train, y_train)
    y_pred = pls_pds.predict(X_target_test_corrected)
    results["PDS"] = compute_metrics(y_test, y_pred)
    results["PDS"]["n_components"] = pls_pds.optimal_n_components

    # 4. SBC + PLS
    # Same approach: transform target to source space
    sbc = SBC()
    sbc.fit(X_source_train, X_target_train)  # primary=source, secondary=target
    X_target_test_corrected = sbc.transform(X_target_test)
    pls_sbc = PLSCalibration(n_components=pls_components)
    pls_sbc.fit(X_source_train, y_train)
    y_pred = pls_sbc.predict(X_target_test_corrected)
    results["SBC"] = compute_metrics(y_test, y_pred)
    results["SBC"]["n_components"] = pls_sbc.optimal_n_components

    # 5. DS + PLS
    # Same approach: transform target to source space
    ds = DS(regularization=1e-2)
    ds.fit(X_source_train, X_target_train)  # primary=source, secondary=target
    X_target_test_corrected = ds.transform(X_target_test)
    pls_ds = PLSCalibration(n_components=pls_components)
    pls_ds.fit(X_source_train, y_train)
    y_pred = pls_ds.predict(X_target_test_corrected)
    results["DS"] = compute_metrics(y_test, y_pred)
    results["DS"]["n_components"] = pls_ds.optimal_n_components

    return results


def print_results_table(results: Dict[str, Dict[str, float]], title: str = ""):
    """Pretty-print results as a table."""
    if title:
        print(f"\n{title}")
        print("=" * 70)

    print(f"{'Method':<15} {'R²':>8} {'RMSEP':>10} {'RPD':>8} {'Bias':>10} {'Slope':>8}")
    print("-" * 70)

    for method, metrics in results.items():
        print(f"{method:<15} {metrics['r2']:>8.4f} {metrics['rmsep']:>10.4f} "
              f"{metrics['rpd']:>8.2f} {metrics['bias']:>10.4f} {metrics['slope']:>8.4f}")

    print("-" * 70)
