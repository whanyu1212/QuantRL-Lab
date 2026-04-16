# Technical Indicators for RL Agents

This directory contains a suite of technical indicators designed to enrich the observation space of Reinforcement Learning (RL) agents. By transforming raw OHLCV (Open, High, Low, Close, Volume) data into structured features, we provide the agent with a "richer" state representation, allowing it to detect patterns, trends, and market regimes more effectively than from raw price action alone.

## Why Indicators Matter for RL

In Financial Reinforcement Learning, the **Markov Property** implies that the current state should contain all necessary information to make an optimal decision. Raw price history alone is often noisy and non-stationary. Technical indicators help by:

1.  **Stationarity**: Many indicators (RSI, CCI, ADX) are bounded or mean-reverting, making them easier for neural networks to normalize and learn from compared to unbounded raw prices.
2.  **Feature Extraction**: They pre-calculate complex relationships (momentum, volatility expansion) that a dense layer might struggle to infer from raw sequence data immediately.
3.  **Regime Detection**: Indicators like ADX or ATR help the agent distinguish between trending vs. ranging markets or high vs. low volatility environments, enabling dynamic strategy adaptation.

---

## Available Indicators

### 1. Trend & Regime
*Helping the agent identify direction, persistence, channels, and structural regime.*

- **SMA**, **EMA**, **MACD**: baseline moving-average trend features and crossover dynamics.
- **ADX**: trend-strength measure independent of direction.
- **CCI**: price deviation from recent typical-price mean.
- **AROON**: how recently highs and lows occurred inside the rolling window.
- **VORTEX**: competing upward vs downward trend pressure.
- **SUPERTREND**: ATR-based trailing trend line with bullish / bearish regime state.
- **DONCHIAN**: breakout channels based on recent highs and lows.
- **KELTNER**: EMA centerline with ATR-based envelopes.

### 2. Momentum & Oscillators
*Helping the agent identify acceleration, exhaustion, and mean-reversion setups.*

- **RSI**: bounded momentum oscillator on a 0-100 scale.
- **STOCH**: close position inside the recent trading range.
- **WILLR**: Williams %R on a -100 to 0 scale.
- **ROC**: percentage price momentum over a configurable lookback.
- **PPO**: percentage spread between fast and slow EMAs.
- **TRIX**: triple-smoothed rate-of-change oscillator.
- **TSI**: double-smoothed momentum strength oscillator.

### 3. Volatility & Range
*Helping the agent estimate activity level, stop distance, and breakout context.*

- **ATR**: absolute volatility via true range smoothing.
- **BB**: Bollinger Bands for relative volatility and squeeze / expansion behavior.
- **NATR**: ATR normalized as a percentage of closing price.

### 4. Volume & Money Flow
*Helping the agent confirm moves with participation and accumulation pressure.*

- **OBV**: cumulative up-volume vs down-volume pressure.
- **MFI**: volume-weighted RSI.
- **CMF**: rolling Chaikin money flow over the chosen window.
- **ADL**: cumulative accumulation / distribution line.
- **CHO**: Chaikin oscillator from the fast / slow EMA spread of ADL.

---

## Usage Example

To add these indicators to your dataframe using the `IndicatorRegistry`:

```python
from quantrl_lab.data.indicators import IndicatorRegistry

# Apply specific indicators
df = IndicatorRegistry.apply("SMA", df, window=20)
df = IndicatorRegistry.apply("RSI", df, window=14)
df = IndicatorRegistry.apply("CMF", df, window=20)
df = IndicatorRegistry.apply("SUPERTREND", df, window=10, multiplier=3.0)

# Or use within a VectorizedStrategy
```
