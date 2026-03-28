"""Rolling correlation and factor-exposure monitoring.

Detects concentration of risk when nominally "independent" strategies or
positions are all exposed to the same underlying factor or market driver.
Caps aggregate exposure to highly correlated clusters.

Design:
    - Correlations are computed on rolling windows of returns.
    - Hierarchical clustering groups instruments into correlated clusters.
    - Aggregate notional per cluster is checked against limits.
    - Factor exposures are tracked via rolling regression against
      a small set of risk factors (market, sector, rates, etc.).

Assumptions:
    - Returns are daily and aligned across instruments.
    - Factor returns are available for the same period.
    - Cluster assignments are updated on a configurable schedule.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CorrelationConfig:
    """Configuration for correlation monitoring.

    Attributes:
        rolling_window: Window size for rolling correlation (days).
        high_corr_threshold: Absolute correlation above which two
            instruments are considered highly correlated.
        max_cluster_notional: Maximum aggregate notional exposure
            allowed in a single correlated cluster.
        recompute_interval_days: How often to recompute the correlation
            matrix and clusters.
        factor_names: Names of risk factors to track.
        max_factor_exposure_pct: Maximum allowed exposure to any single
            factor as a percentage of NAV.
    """

    rolling_window: int = 60
    high_corr_threshold: float = 0.70
    max_cluster_notional: float = 1_000_000_000.0
    recompute_interval_days: int = 5
    factor_names: list[str] = field(
        default_factory=lambda: ["market", "size", "value", "momentum"]
    )
    max_factor_exposure_pct: float = 0.30


@dataclass
class ClusterInfo:
    """Information about a correlated cluster of instruments."""

    cluster_id: int
    members: list[str]
    avg_correlation: float
    aggregate_notional: float
    within_limit: bool


@dataclass
class FactorExposure:
    """Factor exposure for a portfolio or instrument."""

    factor: str
    beta: float
    notional_exposure: float
    pct_of_nav: float
    within_limit: bool


class CorrelationMonitor:
    """Monitor correlations and factor exposures across instruments.

    Usage::

        monitor = CorrelationMonitor(config)

        # Update with new return data
        monitor.update_returns(returns_df)

        # Check correlations
        clusters = monitor.get_clusters()
        violations = monitor.check_cluster_limits(positions, prices)

        # Check factor exposures
        exposures = monitor.compute_factor_exposures(
            portfolio_returns, factor_returns
        )
    """

    def __init__(self, config: CorrelationConfig | None = None) -> None:
        self.config = config or CorrelationConfig()
        self._returns_history: pd.DataFrame | None = None
        self._correlation_matrix: pd.DataFrame | None = None
        self._clusters: list[ClusterInfo] = []
        self._last_recompute: pd.Timestamp | None = None

    def update_returns(self, returns: pd.DataFrame) -> None:
        """Update the return history with new data.

        Args:
            returns: DataFrame with instruments as columns and dates
                as the index.  Values are daily returns.
        """
        if self._returns_history is None:
            self._returns_history = returns
        else:
            self._returns_history = pd.concat(
                [self._returns_history, returns]
            ).sort_index()
            # Keep only recent data
            max_rows = self.config.rolling_window * 3
            if len(self._returns_history) > max_rows:
                self._returns_history = self._returns_history.iloc[-max_rows:]

    def compute_correlation_matrix(self) -> pd.DataFrame:
        """Compute the rolling correlation matrix.

        Returns:
            Correlation matrix as a symmetric DataFrame.
        """
        if self._returns_history is None or self._returns_history.empty:
            return pd.DataFrame()

        window = min(self.config.rolling_window, len(self._returns_history))
        recent = self._returns_history.iloc[-window:]
        self._correlation_matrix = recent.corr()
        return self._correlation_matrix

    def get_clusters(self) -> list[ClusterInfo]:
        """Group instruments into correlated clusters.

        Uses a simple greedy approach: instruments with pairwise
        correlation above the threshold are placed in the same cluster.
        """
        if self._correlation_matrix is None:
            self.compute_correlation_matrix()

        if self._correlation_matrix is None or self._correlation_matrix.empty:
            return []

        corr = self._correlation_matrix
        symbols = list(corr.columns)
        assigned: set[str] = set()
        clusters: list[ClusterInfo] = []
        cluster_id = 0

        for sym in symbols:
            if sym in assigned:
                continue

            # Find all symbols highly correlated with this one
            members = [sym]
            for other in symbols:
                if other == sym or other in assigned:
                    continue
                val = corr.loc[sym, other]
                if np.isfinite(val) and abs(val) >= self.config.high_corr_threshold:
                    members.append(other)

            assigned.update(members)

            # Average pairwise correlation within cluster
            if len(members) > 1:
                pairwise = []
                for i, m1 in enumerate(members):
                    for m2 in members[i + 1 :]:
                        val = corr.loc[m1, m2]
                        if np.isfinite(val):
                            pairwise.append(abs(val))
                avg_corr = float(np.mean(pairwise)) if pairwise else 0.0
            else:
                avg_corr = 1.0

            clusters.append(ClusterInfo(
                cluster_id=cluster_id,
                members=members,
                avg_correlation=avg_corr,
                aggregate_notional=0.0,
                within_limit=True,
            ))
            cluster_id += 1

        self._clusters = clusters
        return clusters

    def check_cluster_limits(
        self,
        positions: dict[str, float],
        prices: dict[str, float],
    ) -> list[ClusterInfo]:
        """Check if any correlated cluster exceeds notional limits.

        Args:
            positions: Symbol → quantity (signed).
            prices: Symbol → current price.

        Returns:
            List of clusters with updated notional and limit status.
        """
        if not self._clusters:
            self.get_clusters()

        for cluster in self._clusters:
            notional = sum(
                abs(positions.get(sym, 0) * prices.get(sym, 0))
                for sym in cluster.members
            )
            cluster.aggregate_notional = notional
            cluster.within_limit = notional <= self.config.max_cluster_notional

            if not cluster.within_limit:
                logger.warning(
                    "Cluster %d LIMIT BREACH: notional=%.0f > limit=%.0f, "
                    "members=%s",
                    cluster.cluster_id,
                    notional,
                    self.config.max_cluster_notional,
                    cluster.members,
                )

        return self._clusters

    def compute_factor_exposures(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
        nav: float = 1.0,
    ) -> list[FactorExposure]:
        """Compute portfolio factor exposures via rolling regression.

        Args:
            portfolio_returns: Portfolio return series.
            factor_returns: Factor return DataFrame (columns = factors).
            nav: Current net asset value for percentage calculation.

        Returns:
            List of ``FactorExposure`` objects.
        """
        aligned_port, aligned_factors = portfolio_returns.align(
            factor_returns, join="inner"
        )
        if len(aligned_port) < 20:
            return []

        window = min(self.config.rolling_window, len(aligned_port))
        recent_port = aligned_port.iloc[-window:]
        recent_factors = aligned_factors.iloc[-window:]

        exposures: list[FactorExposure] = []

        for factor_col in recent_factors.columns:
            factor_vals = recent_factors[factor_col].values
            port_vals = recent_port.values

            if len(factor_vals) < 10:
                continue

            # Simple OLS beta
            cov = np.cov(port_vals, factor_vals)
            beta = float(cov[0, 1] / cov[1, 1]) if cov.shape == (2, 2) and cov[1, 1] > 0 else 0.0

            notional_exposure = beta * nav
            pct_of_nav = abs(beta)

            within_limit = pct_of_nav <= self.config.max_factor_exposure_pct

            exposures.append(FactorExposure(
                factor=factor_col,
                beta=beta,
                notional_exposure=notional_exposure,
                pct_of_nav=pct_of_nav,
                within_limit=within_limit,
            ))

            if not within_limit:
                logger.warning(
                    "Factor exposure LIMIT: %s beta=%.3f (%.1f%% of NAV, "
                    "limit=%.1f%%)",
                    factor_col, beta, pct_of_nav * 100,
                    self.config.max_factor_exposure_pct * 100,
                )

        return exposures

    def summary(self) -> dict:
        """Return a summary of the current monitoring state."""
        return {
            "correlation_matrix_size": (
                self._correlation_matrix.shape
                if self._correlation_matrix is not None
                else (0, 0)
            ),
            "num_clusters": len(self._clusters),
            "clusters_in_breach": sum(
                1 for c in self._clusters if not c.within_limit
            ),
            "return_history_length": (
                len(self._returns_history) if self._returns_history is not None else 0
            ),
        }
