"""Production prediction-powered inference (PPI) engine.

The statistical core of the system: estimators (classical, rectified,
power-tuned), analytic and bootstrap confidence intervals, inverse-
propensity weighting for actively-selected labeled samples, coverage
simulation, and the chunked experiment runner.

Numerical contract: with uniform weights, every estimator in this
package matches the golden reference in ``reference/ppi_reference.py``
within the tolerances stated in ``python/tests/test_against_reference.py``.
"""

from ppi_core.estimators import (
    logistic_classical,
    logistic_ppi,
    logistic_ppi_power_tuned,
    mean_classical,
    mean_lam_star,
    mean_ppi,
    mean_ppi_power_tuned,
    ols_classical,
    ols_ppi,
    ols_ppi_power_tuned,
    quantile_classical,
    quantile_ppi,
)

__all__ = [
    "mean_lam_star",
    "mean_classical",
    "mean_ppi",
    "mean_ppi_power_tuned",
    "quantile_classical",
    "quantile_ppi",
    "ols_classical",
    "ols_ppi",
    "ols_ppi_power_tuned",
    "logistic_classical",
    "logistic_ppi",
    "logistic_ppi_power_tuned",
]
