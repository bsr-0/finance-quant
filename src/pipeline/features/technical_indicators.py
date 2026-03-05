"""Technical indicators for time-series features."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculate technical indicators for price data."""

    @staticmethod
    def sma(prices: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average."""
        return prices.rolling(window=window, min_periods=window).mean()

    @staticmethod
    def ema(prices: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average."""
        return prices.ewm(span=window, adjust=False, min_periods=window).mean()

    @staticmethod
    def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=window).mean()

        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def macd(
        prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """MACD indicator."""
        ema_fast = TechnicalIndicators.ema(prices, fast)
        ema_slow = TechnicalIndicators.ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(
        prices: pd.Series, window: int = 20, num_std: float = 2.0
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands."""
        sma = TechnicalIndicators.sma(prices, window)
        std = prices.rolling(window=window, min_periods=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, sma, lower

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range."""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)

        return true_range.rolling(window=window, min_periods=window).mean()

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume."""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]

        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]

        return obv

    @staticmethod
    def stochastic(
        high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3
    ) -> tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_window, min_periods=k_window).min()
        highest_high = high.rolling(window=k_window, min_periods=k_window).max()

        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        k = k.replace([np.inf, -np.inf], np.nan)
        d = k.rolling(window=d_window, min_periods=d_window).mean()

        return k, d

    @staticmethod
    def momentum(prices: pd.Series, window: int = 10) -> pd.Series:
        """Price Momentum."""
        return prices.diff(window)

    @staticmethod
    def roc(prices: pd.Series, window: int = 10) -> pd.Series:
        """Rate of Change."""
        return (prices / prices.shift(window) - 1) * 100

    @staticmethod
    def williams_r(
        high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
    ) -> pd.Series:
        """Williams %R."""
        highest_high = high.rolling(window=window, min_periods=window).max()
        lowest_low = low.rolling(window=window, min_periods=window).min()

        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def calculate_all(
        df: pd.DataFrame,
        price_col: str = "close",
        high_col: str = "high",
        low_col: str = "low",
        volume_col: str = "volume",
    ) -> pd.DataFrame:
        """Calculate all technical indicators."""
        result = df.copy()
        prices = df[price_col]

        # Moving averages
        result["sma_10"] = TechnicalIndicators.sma(prices, 10)
        result["sma_20"] = TechnicalIndicators.sma(prices, 20)
        result["sma_50"] = TechnicalIndicators.sma(prices, 50)
        result["ema_12"] = TechnicalIndicators.ema(prices, 12)
        result["ema_26"] = TechnicalIndicators.ema(prices, 26)

        # Momentum indicators
        result["rsi_14"] = TechnicalIndicators.rsi(prices, 14)
        result["momentum_10"] = TechnicalIndicators.momentum(prices, 10)
        result["roc_10"] = TechnicalIndicators.roc(prices, 10)

        # MACD
        macd, signal, hist = TechnicalIndicators.macd(prices)
        result["macd"] = macd
        result["macd_signal"] = signal
        result["macd_hist"] = hist

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(prices)
        result["bb_upper"] = bb_upper
        result["bb_middle"] = bb_middle
        result["bb_lower"] = bb_lower
        result["bb_width"] = (bb_upper - bb_lower) / bb_middle
        result["bb_position"] = (prices - bb_lower) / (bb_upper - bb_lower)

        # Volatility
        if high_col in df.columns and low_col in df.columns:
            result["atr_14"] = TechnicalIndicators.atr(df[high_col], df[low_col], prices, 14)

            # Stochastic
            k, d = TechnicalIndicators.stochastic(df[high_col], df[low_col], prices)
            result["stoch_k"] = k
            result["stoch_d"] = d

            # Williams %R
            result["williams_r"] = TechnicalIndicators.williams_r(df[high_col], df[low_col], prices)

        # Volume indicators
        if volume_col in df.columns:
            result["obv"] = TechnicalIndicators.obv(prices, df[volume_col])
            result["volume_sma_20"] = TechnicalIndicators.sma(df[volume_col], 20)

        return result


class ContractFeatureEngineer:
    """Feature engineering for prediction market contracts."""

    @staticmethod
    def calculate_price_features(prices: pd.Series) -> pd.DataFrame:
        """Calculate price-based features for contracts."""
        features = pd.DataFrame(index=prices.index)

        # Price levels
        features["price"] = prices
        features["price_log"] = np.log(prices.replace(0, 0.001))

        # Price changes
        features["price_change_1h"] = prices.diff(1)
        features["price_change_24h"] = prices.diff(24)
        features["price_return_1h"] = prices.pct_change(1)
        features["price_return_24h"] = prices.pct_change(24)

        # Moving averages
        features["price_sma_6h"] = prices.rolling(6, min_periods=None).mean()
        features["price_sma_24h"] = prices.rolling(24, min_periods=None).mean()

        # Volatility
        features["price_volatility_6h"] = prices.rolling(6, min_periods=None).std()
        features["price_volatility_24h"] = prices.rolling(24, min_periods=None).std()

        # Extremes
        features["price_max_24h"] = prices.rolling(24, min_periods=None).max()
        features["price_min_24h"] = prices.rolling(24, min_periods=None).min()
        features["price_range_24h"] = features["price_max_24h"] - features["price_min_24h"]

        # Momentum
        features["price_momentum"] = prices.diff(6)

        return features

    @staticmethod
    def calculate_liquidity_features(trades_df: pd.DataFrame, window: str = "24h") -> pd.DataFrame:
        """Calculate liquidity features from trades."""
        if trades_df.empty:
            return pd.DataFrame()

        # Resample to hourly
        trades_df = trades_df.copy()
        trades_df["ts"] = pd.to_datetime(trades_df["ts"])
        trades_df.set_index("ts", inplace=True)

        hourly = trades_df.resample("1h").agg(
            {"size": ["sum", "count", "mean"], "price": ["std", "min", "max"]}
        )

        hourly.columns = [
            "volume",
            "trade_count",
            "avg_trade_size",
            "price_std",
            "price_min",
            "price_max",
        ]

        # Calculate rolling metrics
        window_map = {"1h": 1, "6h": 6, "24h": 24}
        w = window_map.get(window, 24)

        features = pd.DataFrame(index=hourly.index)
        features["volume_rolling"] = hourly["volume"].rolling(w, min_periods=None).sum()
        features["trade_count_rolling"] = hourly["trade_count"].rolling(w, min_periods=None).sum()
        features["avg_trade_size"] = hourly["avg_trade_size"]
        features["price_volatility"] = hourly["price_std"]
        features["bid_ask_spread_est"] = hourly["price_max"] - hourly["price_min"]

        return features

    @staticmethod
    def calculate_market_features(
        contract_prices: pd.DataFrame, macro_data: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Calculate market-wide features."""
        features = pd.DataFrame(index=contract_prices.index)

        # Correlation with market (if multiple contracts)
        if len(contract_prices.columns) > 1:
            features["market_correlation"] = contract_prices.corr().mean()

        # Market volatility
        features["market_volatility"] = contract_prices.std(axis=1)

        # Market momentum
        features["market_momentum"] = contract_prices.pct_change().mean(axis=1)

        # Macro correlation (if provided)
        if macro_data is not None and not macro_data.empty:
            for col in macro_data.columns:
                features[f"macro_{col}"] = macro_data[col]

        return features
