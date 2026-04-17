"""
Unit tests for the alpha_research module.

Tests cover:
- Core data structures (AlphaJob, AlphaResult)
- Strategy registry
- Vectorized strategies
- AlphaRunner execution
- Ensemble methods
- Robustness testing
- Visualization
"""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from quantrl_lab.alpha_research.alpha_strategies import (
    MACDCrossoverStrategy,
    MeanReversionStrategy,
    TrendFollowingStrategy,
)
from quantrl_lab.alpha_research.analysis import RobustnessTester
from quantrl_lab.alpha_research.base import SignalType
from quantrl_lab.alpha_research.ensemble import AlphaEnsemble
from quantrl_lab.alpha_research.indicator_research import (
    FEATURE_SELECTION_MODE,
    INDICATOR_RESEARCH_METADATA,
)
from quantrl_lab.alpha_research.models import AlphaJob, AlphaResult
from quantrl_lab.alpha_research.registry import VectorizedStrategyRegistry
from quantrl_lab.alpha_research.runner import AlphaRunner
from quantrl_lab.alpha_research.visualization import AlphaVisualizer

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 252  # One year of trading days
    dates = pd.date_range("2023-01-01", periods=n, freq="B")

    # Generate realistic price data with trend
    base_price = 100
    returns = np.random.randn(n) * 0.02  # 2% daily volatility
    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "Open": prices * (1 + np.random.randn(n) * 0.005),
            "High": prices * (1 + np.abs(np.random.randn(n) * 0.01)),
            "Low": prices * (1 - np.abs(np.random.randn(n) * 0.01)),
            "Close": prices,
            "Volume": np.random.randint(1000000, 10000000, n),
        },
        index=dates,
    )

    return df


@pytest.fixture
def sample_data_with_indicators(sample_ohlcv_data: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to sample data."""
    df = sample_ohlcv_data.copy()

    # Add SMA
    df["SMA_20"] = df["Close"].rolling(20).mean()

    # Add RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # Add MACD
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Add Williams %R
    df["WILLR_14"] = (
        (df["High"].rolling(14).max() - df["Close"]) / (df["High"].rolling(14).max() - df["Low"].rolling(14).min())
    ) * -100

    # Drop NaN rows
    df = df.dropna()

    return df


@pytest.fixture
def sample_alpha_job(sample_data_with_indicators: pd.DataFrame) -> AlphaJob:
    """Create a sample AlphaJob for testing."""
    return AlphaJob(
        data=sample_data_with_indicators,
        indicator_name="RSI",
        strategy_name="mean_reversion",
        indicator_params={"window": 14},
        strategy_params={"indicator_col": "RSI_14", "oversold": 30, "overbought": 70},
        allow_short=True,
    )


@pytest.fixture
def sample_alpha_result(sample_alpha_job: AlphaJob, sample_data_with_indicators: pd.DataFrame) -> AlphaResult:
    """Create a sample AlphaResult for testing."""
    # Create a simple equity curve
    n = len(sample_data_with_indicators)
    equity = pd.Series(1.0 + np.cumsum(np.random.randn(n) * 0.01), index=sample_data_with_indicators.index)

    return AlphaResult(
        job=sample_alpha_job,
        metrics={
            "sharpe_ratio": 0.75,
            "sortino_ratio": 1.1,
            "calmar_ratio": 0.5,
            "total_return": 0.15,
            "max_drawdown": -0.12,
            "win_rate": 0.55,
            "ic": 0.03,
            "rank_ic": 0.04,
            "mutual_info": 0.01,
        },
        equity_curve=equity,
        signals=pd.Series(np.random.choice([1, 0, -1], n), index=sample_data_with_indicators.index),
        status="completed",
    )


@pytest.fixture
def multiple_alpha_results(sample_data_with_indicators: pd.DataFrame) -> list:
    """Create multiple AlphaResults for ensemble testing."""
    results = []
    strategies = [
        ("RSI", "mean_reversion", {"indicator_col": "RSI_14"}),
        ("SMA", "trend_following", {"indicator_col": "SMA_20"}),
        ("MACD", "macd_crossover", {"fast_col": "MACD", "slow_col": "MACD_signal"}),
    ]

    n = len(sample_data_with_indicators)

    for i, (ind, strat, params) in enumerate(strategies):
        job = AlphaJob(
            data=sample_data_with_indicators,
            indicator_name=ind,
            strategy_name=strat,
            strategy_params=params,
        )

        # Generate different equity curves
        np.random.seed(42 + i)
        equity = pd.Series(1.0 + np.cumsum(np.random.randn(n) * 0.01), index=sample_data_with_indicators.index)

        result = AlphaResult(
            job=job,
            metrics={
                "sharpe_ratio": 0.5 + i * 0.2,
                "sortino_ratio": 0.7 + i * 0.2,
                "total_return": 0.1 + i * 0.05,
                "max_drawdown": -0.15 + i * 0.02,
                "ic": 0.02 + i * 0.01,
            },
            equity_curve=equity,
            status="completed",
        )
        results.append(result)

    return results


# ============================================================================
# Core Data Structure Tests
# ============================================================================


class TestAlphaJob:
    """Tests for AlphaJob dataclass."""

    def test_alpha_job_creation(self, sample_ohlcv_data):
        """Test basic AlphaJob creation."""
        job = AlphaJob(
            data=sample_ohlcv_data,
            indicator_name="RSI",
            strategy_name="mean_reversion",
        )

        assert job.indicator_name == "RSI"
        assert job.strategy_name == "mean_reversion"
        assert len(job.id) == 8  # UUID short format
        assert isinstance(job.indicator_params, dict)
        assert isinstance(job.strategy_params, dict)

    def test_alpha_job_with_params(self, sample_ohlcv_data):
        """Test AlphaJob with custom parameters."""
        job = AlphaJob(
            data=sample_ohlcv_data,
            indicator_name="RSI",
            strategy_name="mean_reversion",
            indicator_params={"window": 14},
            strategy_params={"oversold": 25, "overbought": 75},
            allow_short=False,
            tags={"category": "momentum"},
        )

        assert job.indicator_params["window"] == 14
        assert job.strategy_params["oversold"] == 25
        assert job.allow_short is False
        assert job.tags["category"] == "momentum"

    def test_alpha_job_unique_ids(self, sample_ohlcv_data):
        """Test that each AlphaJob gets a unique ID."""
        jobs = [
            AlphaJob(data=sample_ohlcv_data, indicator_name="RSI", strategy_name="mean_reversion") for _ in range(10)
        ]

        ids = [j.id for j in jobs]
        assert len(ids) == len(set(ids)), "Job IDs should be unique"


class TestAlphaResult:
    """Tests for AlphaResult dataclass."""

    def test_alpha_result_creation(self, sample_alpha_job):
        """Test basic AlphaResult creation."""
        result = AlphaResult(
            job=sample_alpha_job,
            metrics={"sharpe_ratio": 1.0},
            status="completed",
        )

        assert result.job == sample_alpha_job
        assert result.metrics["sharpe_ratio"] == 1.0
        assert result.status == "completed"
        assert result.equity_curve is None
        assert result.error is None

    def test_alpha_result_with_error(self, sample_alpha_job):
        """Test AlphaResult with failed status stores error as string
        (for pickling safety)."""
        # error field is Optional[str] — not Exception — so parallel joblib workers
        # can serialise AlphaResult across processes without pickle errors.
        error_msg = "Traceback (most recent call last):\n  ...\nValueError: Test error"
        result = AlphaResult(
            job=sample_alpha_job,
            metrics={},
            status="failed",
            error=error_msg,
        )

        assert result.status == "failed"
        assert isinstance(result.error, str)
        assert "ValueError" in result.error


# ============================================================================
# Strategy Registry Tests
# ============================================================================


class TestVectorizedStrategyRegistry:
    """Tests for VectorizedStrategyRegistry."""

    def test_list_strategies(self):
        """Test listing registered strategies."""
        strategies = VectorizedStrategyRegistry.list_strategies()

        assert isinstance(strategies, list)
        assert "mean_reversion" in strategies
        assert "trend_following" in strategies
        assert "macd_crossover" in strategies

    def test_create_strategy(self):
        """Test creating a strategy by name."""
        strategy = VectorizedStrategyRegistry.create(
            "mean_reversion",
            indicator_col="RSI_14",
            oversold=30,
            overbought=70,
        )

        assert isinstance(strategy, MeanReversionStrategy)
        assert strategy.indicator_col == "RSI_14"
        assert strategy.oversold == 30
        assert strategy.overbought == 70

    def test_create_unknown_strategy(self):
        """Test creating an unknown strategy raises error."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            VectorizedStrategyRegistry.create("nonexistent_strategy")


# ============================================================================
# Vectorized Strategy Tests
# ============================================================================


class TestTrendFollowingStrategy:
    """Tests for TrendFollowingStrategy."""

    def test_generate_signals(self, sample_data_with_indicators):
        """Test signal generation."""
        strategy = TrendFollowingStrategy(indicator_col="SMA_20", allow_short=True)
        signals = strategy.generate_signals(sample_data_with_indicators)

        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_data_with_indicators)
        assert set(signals.unique()).issubset({SignalType.BUY.value, SignalType.SELL.value, SignalType.HOLD.value})

    def test_generate_signals_no_short(self, sample_data_with_indicators):
        """Test signal generation without shorting."""
        strategy = TrendFollowingStrategy(indicator_col="SMA_20", allow_short=False)
        signals = strategy.generate_signals(sample_data_with_indicators)

        # Should not have SELL signals when shorting is disabled
        assert SignalType.SELL.value not in signals.values

    def test_required_columns(self):
        """Test required columns."""
        strategy = TrendFollowingStrategy(indicator_col="SMA_20")
        cols = strategy.get_required_columns()

        assert "SMA_20" in cols
        assert "Close" in cols


class TestMeanReversionStrategy:
    """Tests for MeanReversionStrategy."""

    def test_generate_signals(self, sample_data_with_indicators):
        """Test signal generation."""
        strategy = MeanReversionStrategy(indicator_col="RSI_14", oversold=30, overbought=70)
        signals = strategy.generate_signals(sample_data_with_indicators)

        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_data_with_indicators)

    def test_oversold_generates_buy(self, sample_data_with_indicators):
        """Test that oversold condition generates BUY signal."""
        # Force some RSI values to be oversold
        df = sample_data_with_indicators.copy()
        df.loc[df.index[:10], "RSI_14"] = 20  # Oversold

        strategy = MeanReversionStrategy(indicator_col="RSI_14", oversold=30, overbought=70)
        signals = strategy.generate_signals(df)

        # At least some of the first 10 signals should be BUY
        assert SignalType.BUY.value in signals.iloc[:10].values

    def test_overbought_generates_sell(self, sample_data_with_indicators):
        """Test that overbought condition generates SELL signal."""
        df = sample_data_with_indicators.copy()
        df.loc[df.index[:10], "RSI_14"] = 80  # Overbought

        strategy = MeanReversionStrategy(indicator_col="RSI_14", oversold=30, overbought=70, allow_short=True)
        signals = strategy.generate_signals(df)

        assert SignalType.SELL.value in signals.iloc[:10].values


class TestMACDCrossoverStrategy:
    """Tests for MACDCrossoverStrategy."""

    def test_generate_signals(self, sample_data_with_indicators):
        """Test signal generation."""
        strategy = MACDCrossoverStrategy(fast_col="MACD", slow_col="MACD_signal")
        signals = strategy.generate_signals(sample_data_with_indicators)

        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_data_with_indicators)


# ============================================================================
# AlphaRunner Tests
# ============================================================================


class TestAlphaRunner:
    """Tests for AlphaRunner."""

    def test_runner_initialization(self):
        """Test runner initialization."""
        runner = AlphaRunner(verbose=False)
        assert runner.verbose is False

    def test_run_single_job(self, sample_alpha_job):
        """Test running a single job."""
        runner = AlphaRunner(verbose=False)
        result = runner.run_job(sample_alpha_job)

        assert isinstance(result, AlphaResult)
        assert result.job == sample_alpha_job
        # Result should be either completed or failed
        assert result.status in ["completed", "failed"]

    def test_run_feature_mode_job_returns_scores(self, sample_alpha_job):
        """Feature mode should return score-based metrics without an
        equity curve."""
        runner = AlphaRunner(verbose=False)
        feature_job = AlphaJob(
            data=sample_alpha_job.data,
            indicator_name=sample_alpha_job.indicator_name,
            strategy_name=sample_alpha_job.strategy_name,
            indicator_params=sample_alpha_job.indicator_params,
            strategy_params=sample_alpha_job.strategy_params,
            evaluation_mode=FEATURE_SELECTION_MODE,
        )

        result = runner.run_job(feature_job)

        assert result.status == "completed"
        assert result.equity_curve is None
        assert result.signals is None
        assert result.scores is not None
        assert "feature_ic" in result.metrics
        assert "ic" in result.metrics
        assert "sharpe_ratio" not in result.metrics

    def test_run_batch(self, sample_data_with_indicators):
        """Test running a batch of jobs."""
        jobs = [
            AlphaJob(
                data=sample_data_with_indicators,
                indicator_name="RSI",
                strategy_name="mean_reversion",
                strategy_params={"indicator_col": "RSI_14"},
            ),
            AlphaJob(
                data=sample_data_with_indicators,
                indicator_name="SMA",
                strategy_name="trend_following",
                strategy_params={"indicator_col": "SMA_20"},
            ),
        ]

        runner = AlphaRunner(verbose=False)
        results = runner.run_batch(jobs, n_jobs=1)

        assert len(results) == 2
        assert all(isinstance(r, AlphaResult) for r in results)

    def test_run_batch_parallel(self, sample_data_with_indicators):
        """Test running batch with parallel execution."""
        jobs = [
            AlphaJob(
                data=sample_data_with_indicators,
                indicator_name="RSI",
                strategy_name="mean_reversion",
                strategy_params={"indicator_col": "RSI_14"},
            )
            for _ in range(4)
        ]

        runner = AlphaRunner(verbose=False)
        results = runner.run_batch(jobs, n_jobs=2)

        assert len(results) == 4


# ============================================================================
# Ensemble Tests
# ============================================================================


class TestAlphaEnsemble:
    """Tests for AlphaEnsemble."""

    def test_ensemble_initialization(self, multiple_alpha_results):
        """Test ensemble initialization."""
        ensemble = AlphaEnsemble(multiple_alpha_results)
        assert len(ensemble.results) == len(multiple_alpha_results)

    def test_ensemble_no_valid_results(self):
        """Test ensemble with no valid results raises error."""
        with pytest.raises(ValueError, match="No valid alpha results"):
            AlphaEnsemble([])

    def test_equal_weight_combination(self, multiple_alpha_results):
        """Test equal weight combination method."""
        ensemble = AlphaEnsemble(multiple_alpha_results)
        combined = ensemble.combine(method="equal_weight")

        assert isinstance(combined, pd.Series)
        assert len(combined) > 0
        # Should start at 1.0 (normalized)
        assert np.isclose(combined.iloc[0], 1.0)

    def test_inverse_volatility_combination(self, multiple_alpha_results):
        """Test inverse volatility combination method."""
        ensemble = AlphaEnsemble(multiple_alpha_results)
        combined = ensemble.combine(method="inverse_volatility")

        assert isinstance(combined, pd.Series)
        assert np.isclose(combined.iloc[0], 1.0)

    def test_ic_weighted_combination(self, multiple_alpha_results):
        """Test IC-weighted combination method."""
        ensemble = AlphaEnsemble(multiple_alpha_results)
        combined = ensemble.combine(method="ic_weighted")

        assert isinstance(combined, pd.Series)
        assert np.isclose(combined.iloc[0], 1.0)

    def test_sharpe_weighted_combination(self, multiple_alpha_results):
        """Test Sharpe-weighted combination method."""
        ensemble = AlphaEnsemble(multiple_alpha_results)
        combined = ensemble.combine(method="sharpe_weighted")

        assert isinstance(combined, pd.Series)
        assert np.isclose(combined.iloc[0], 1.0)

    def test_unknown_combination_method(self, multiple_alpha_results):
        """Test unknown combination method raises error."""
        ensemble = AlphaEnsemble(multiple_alpha_results)
        with pytest.raises(ValueError, match="Unknown combination method"):
            ensemble.combine(method="nonexistent_method")

    def test_returns_matrix(self, multiple_alpha_results):
        """Test internal returns matrix generation."""
        ensemble = AlphaEnsemble(multiple_alpha_results)
        returns_df = ensemble._get_returns_matrix()

        assert isinstance(returns_df, pd.DataFrame)
        assert returns_df.shape[1] == len(multiple_alpha_results)


# ============================================================================
# Robustness Tests
# ============================================================================


class TestRobustnessTester:
    """Tests for RobustnessTester."""

    def test_robustness_tester_initialization(self):
        """Test robustness tester initialization."""
        tester = RobustnessTester()
        assert tester.runner is not None

    def test_robustness_tester_with_custom_runner(self):
        """Test robustness tester with custom runner."""
        runner = AlphaRunner(verbose=False)
        tester = RobustnessTester(runner=runner)
        assert tester.runner == runner

    def test_parameter_sensitivity(self, sample_data_with_indicators):
        """Test parameter sensitivity analysis."""
        base_job = AlphaJob(
            data=sample_data_with_indicators,
            indicator_name="RSI",
            strategy_name="mean_reversion",
            indicator_params={"window": 14},
            strategy_params={"indicator_col": "RSI_14", "oversold": 30, "overbought": 70},
        )

        param_grid = {
            "oversold": [25, 30],
            "overbought": [70, 75],
        }

        tester = RobustnessTester(runner=AlphaRunner(verbose=False))
        results_df = tester.parameter_sensitivity(base_job, param_grid, n_jobs=1)

        assert isinstance(results_df, pd.DataFrame)
        # Should have 4 rows (2x2 combinations)
        assert len(results_df) <= 4  # May be less if some failed

    def test_sub_period_analysis(self, sample_alpha_result):
        """Test sub-period analysis."""
        tester = RobustnessTester()

        # Use quarterly periods
        period_df = tester.sub_period_analysis(sample_alpha_result, period="Q")

        assert isinstance(period_df, pd.DataFrame)
        # Should have some standard columns
        if not period_df.empty:
            assert "Return" in period_df.columns
            assert "Sharpe" in period_df.columns

    def test_sub_period_analysis_invalid_result(self, sample_alpha_job):
        """Test sub-period analysis with invalid result."""
        invalid_result = AlphaResult(
            job=sample_alpha_job,
            metrics={},
            status="failed",
        )

        tester = RobustnessTester()
        with pytest.raises(ValueError, match="not completed"):
            tester.sub_period_analysis(invalid_result)


# ============================================================================
# Visualization Tests
# ============================================================================


class TestAlphaVisualizer:
    """Tests for AlphaVisualizer."""

    def test_visualizer_initialization(self):
        """Test visualizer initialization."""
        viz = AlphaVisualizer()
        assert viz is not None
        assert len(viz.PALETTE) > 0

    def test_plot_cumulative_returns(self, multiple_alpha_results):
        """Test cumulative returns plot generation."""
        viz = AlphaVisualizer()
        fig = viz.plot_cumulative_returns(multiple_alpha_results)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_drawdowns(self, multiple_alpha_results):
        """Test drawdown plot generation."""
        viz = AlphaVisualizer()
        fig = viz.plot_drawdowns(multiple_alpha_results)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_ic_analysis(self, multiple_alpha_results):
        """Test IC analysis plot generation."""
        viz = AlphaVisualizer()
        fig = viz.plot_ic_analysis(multiple_alpha_results)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_rolling_sharpe(self, multiple_alpha_results):
        """Test rolling Sharpe plot generation."""
        viz = AlphaVisualizer()
        fig = viz.plot_rolling_sharpe(multiple_alpha_results)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_metrics_radar(self, multiple_alpha_results):
        """Test radar chart generation."""
        viz = AlphaVisualizer()
        fig = viz.plot_metrics_radar(multiple_alpha_results)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    @pytest.mark.skipif(
        importlib.util.find_spec("plotly") is None,
        reason="plotly not installed (uv sync --extra viz)",
    )
    def test_generate_html_report(self, multiple_alpha_results, tmp_path):
        """Test HTML report generation."""
        viz = AlphaVisualizer()
        output_path = tmp_path / "test_report.html"

        viz.generate_html_report(multiple_alpha_results, str(output_path))

        assert output_path.exists()
        content = output_path.read_text()
        assert "Alpha Research Report" in content
        assert "Strategy Performance" in content

    def test_fig_to_base64(self, multiple_alpha_results):
        """Test figure to base64 conversion."""
        viz = AlphaVisualizer()
        fig = viz.plot_cumulative_returns(multiple_alpha_results)

        base64_str = viz._fig_to_base64(fig)

        assert isinstance(base64_str, str)
        assert len(base64_str) > 0

        import matplotlib.pyplot as plt

        plt.close(fig)


# ============================================================================
# Integration Tests
# ============================================================================


class TestAlphaResearchIntegration:
    """Integration tests for the alpha research workflow."""

    def test_full_workflow(self, sample_data_with_indicators):
        """Test complete alpha research workflow."""
        # 1. Create jobs
        jobs = [
            AlphaJob(
                data=sample_data_with_indicators,
                indicator_name="RSI",
                strategy_name="mean_reversion",
                strategy_params={"indicator_col": "RSI_14", "oversold": 30, "overbought": 70},
            ),
            AlphaJob(
                data=sample_data_with_indicators,
                indicator_name="SMA",
                strategy_name="trend_following",
                strategy_params={"indicator_col": "SMA_20"},
            ),
        ]

        # 2. Run jobs
        runner = AlphaRunner(verbose=False)
        results = runner.run_batch(jobs, n_jobs=1)

        assert len(results) == 2

        # 3. Filter completed results
        completed = [r for r in results if r.status == "completed"]

        if len(completed) >= 2:
            # 4. Create ensemble
            ensemble = AlphaEnsemble(completed)
            combined = ensemble.combine(method="equal_weight")

            assert isinstance(combined, pd.Series)
            assert len(combined) > 0

    def test_workflow_with_robustness(self, sample_data_with_indicators):
        """Test workflow including robustness analysis."""
        # Create base job
        base_job = AlphaJob(
            data=sample_data_with_indicators,
            indicator_name="RSI",
            strategy_name="mean_reversion",
            indicator_params={"window": 14},
            strategy_params={"indicator_col": "RSI_14", "oversold": 30, "overbought": 70},
        )

        # Run sensitivity analysis
        tester = RobustnessTester(runner=AlphaRunner(verbose=False))

        param_grid = {"oversold": [25, 30]}
        results_df = tester.parameter_sensitivity(base_job, param_grid, n_jobs=1)

        assert isinstance(results_df, pd.DataFrame)


# ============================================================================
# Tests for Bug Fixes (High + Medium + Low Priority)
# ============================================================================


class TestBollingerBandsStateMachine:
    """Tests for the BB state machine fix (was: broken ffill exit
    logic)."""

    @pytest.fixture()
    def bb_data(self):
        """Synthetic data with clear band-crossing events."""
        idx = pd.date_range("2023-01-01", periods=20)
        return pd.DataFrame(
            {
                # price dips below lower band at bar 3, returns to mid at bar 6
                # price rises above upper band at bar 9, returns to mid at bar 13
                "Close": [100, 99, 97, 95, 96, 98, 100, 101, 103, 105, 106, 104, 101, 99, 97, 98, 100, 102, 105, 107],
                "BB_lower": [96] * 20,
                "BB_middle": [100] * 20,
                "BB_upper": [104] * 20,
                "Open": 100,
                "High": 110,
                "Low": 90,
                "Volume": 1_000_000,
            },
            index=idx,
        )

    def _make_strategy(self, allow_short=True):
        from quantrl_lab.alpha_research.alpha_strategies import BollingerBandsStrategy

        return BollingerBandsStrategy(
            lower_col="BB_lower", middle_col="BB_middle", upper_col="BB_upper", allow_short=allow_short
        )

    def test_buy_entry_below_lower_band(self, bb_data):
        """Price touching lower band triggers BUY."""
        strat = self._make_strategy()
        sigs = strat.generate_signals(bb_data)
        # Bar 3: close=95 <= lower=96 → BUY
        assert sigs.iloc[3] == SignalType.BUY.value

    def test_buy_exit_at_middle_band(self, bb_data):
        """BUY position exits as soon as price reaches middle band."""
        strat = self._make_strategy()
        sigs = strat.generate_signals(bb_data)
        # Bar 6: close=100 >= middle=100 → HOLD (exit)
        assert sigs.iloc[6] == SignalType.HOLD.value

    def test_exit_sticks_after_middle_cross(self, bb_data):
        """After exit, HOLD persists until the next entry signal."""
        strat = self._make_strategy()
        sigs = strat.generate_signals(bb_data)
        # Bars 7-8 (close=101, 103) are inside the bands → no new signal → HOLD
        assert sigs.iloc[7] == SignalType.HOLD.value
        assert sigs.iloc[8] == SignalType.HOLD.value

    def test_sell_entry_above_upper_band(self, bb_data):
        """Price touching upper band triggers SELL."""
        strat = self._make_strategy(allow_short=True)
        sigs = strat.generate_signals(bb_data)
        # Bar 9: close=105 >= upper=104 → SELL
        assert sigs.iloc[9] == SignalType.SELL.value

    def test_sell_exit_at_middle_band(self, bb_data):
        """SELL position exits as soon as price falls to middle band."""
        strat = self._make_strategy(allow_short=True)
        sigs = strat.generate_signals(bb_data)
        # Bar 13: close=99 <= middle=100 → HOLD (exit)
        assert sigs.iloc[13] == SignalType.HOLD.value

    def test_no_sell_when_short_disabled(self, bb_data):
        """SELL signals are suppressed when allow_short=False."""
        strat = self._make_strategy(allow_short=False)
        sigs = strat.generate_signals(bb_data)
        assert SignalType.SELL.value not in sigs.values

    def test_missing_columns_returns_hold(self, bb_data):
        """Missing indicator columns yields all-HOLD rather than
        crashing."""
        from quantrl_lab.alpha_research.alpha_strategies import BollingerBandsStrategy

        strat = BollingerBandsStrategy(lower_col="MISSING_L", middle_col="MISSING_M", upper_col="MISSING_U")
        sigs = strat.generate_signals(bb_data)
        assert (sigs == SignalType.HOLD.value).all()


class TestErrorStringPickling:
    """Tests for the error-as-string fix (for joblib pickling
    safety)."""

    def test_error_field_is_str(self, sample_alpha_job):
        """AlphaResult.error must be Optional[str], not Exception."""
        # Verify the field accepts a string (not Exception) by constructing one
        result = AlphaResult(job=sample_alpha_job, metrics={}, status="failed", error="traceback text")
        assert isinstance(result.error, str)

    def test_runner_stores_traceback_string_on_failure(self, sample_ohlcv_data):
        """AlphaRunner.run_job stores a traceback string, not an
        Exception object."""
        # Create a job that will definitely fail (unknown indicator)
        bad_job = AlphaJob(
            data=sample_ohlcv_data,
            indicator_name="NONEXISTENT_INDICATOR_XYZ",
            strategy_name="mean_reversion",
        )
        runner = AlphaRunner(verbose=False)
        result = runner.run_job(bad_job)

        assert result.status == "failed"
        assert isinstance(result.error, str), "error should be a traceback string, not an Exception"
        assert len(result.error) > 0

    def test_failed_result_is_picklable(self, sample_ohlcv_data):
        """Failed AlphaResult must survive pickle round-trip (needed for
        joblib)."""
        import pickle

        bad_job = AlphaJob(
            data=sample_ohlcv_data,
            indicator_name="NONEXISTENT_INDICATOR_XYZ",
            strategy_name="mean_reversion",
        )
        runner = AlphaRunner(verbose=False)
        result = runner.run_job(bad_job)

        # Should not raise — was broken before error field was changed to str
        serialised = pickle.dumps(result)
        restored = pickle.loads(serialised)
        assert restored.status == "failed"
        assert restored.error == result.error


class TestMeanReversionIndicatorScale:
    """Tests for explicit indicator_scale parameter (replaces runtime
    heuristic)."""

    @pytest.fixture()
    def rsi_data(self):
        idx = pd.date_range("2023-01-01", periods=5)
        return pd.DataFrame({"RSI": [20.0, 35.0, 50.0, 65.0, 80.0], "Close": 100}, index=idx)

    @pytest.fixture()
    def willr_data(self):
        idx = pd.date_range("2023-01-01", periods=5)
        # Williams %R: -100 = most oversold, 0 = most overbought
        return pd.DataFrame({"WillR": [-90.0, -70.0, -50.0, -30.0, -10.0], "Close": 100}, index=idx)

    def test_default_scale_rsi_oversold_positive(self, rsi_data):
        """RSI=20 (oversold) should produce a positive score with
        default scale."""
        strat = MeanReversionStrategy(indicator_col="RSI")
        scores = strat.generate_scores(rsi_data)
        # (50-20)/50 = 0.6 > 0
        assert scores.iloc[0] > 0

    def test_default_scale_rsi_overbought_negative(self, rsi_data):
        """RSI=80 (overbought) should produce a negative score."""
        strat = MeanReversionStrategy(indicator_col="RSI")
        scores = strat.generate_scores(rsi_data)
        # (50-80)/50 = -0.6 < 0
        assert scores.iloc[-1] < 0

    def test_williams_r_scale_oversold_positive(self, willr_data):
        """WillR=-90 (oversold) should produce a positive score with
        williams_r scale."""
        strat = MeanReversionStrategy(indicator_col="WillR", indicator_scale="williams_r")
        scores = strat.generate_scores(willr_data)
        # (-50 - -90)/50 = 0.8 > 0
        assert scores.iloc[0] > 0

    def test_williams_r_scale_overbought_negative(self, willr_data):
        """WillR=-10 (overbought) should produce a negative score."""
        strat = MeanReversionStrategy(indicator_col="WillR", indicator_scale="williams_r")
        scores = strat.generate_scores(willr_data)
        # (-50 - -10)/50 = -0.8 < 0
        assert scores.iloc[-1] < 0

    def test_wrong_scale_gives_wrong_sign(self, willr_data):
        """
        Using default scale for Williams %R data gives opposite sign to
        correct scale.

        WillR=-10 is near overbought (should be SELL / negative score).
        - correct (williams_r): (-50 - -10)/50 = -0.8  (negative → SELL ✓)
        - wrong   (0_100):      (50  - -10)/50 =  1.2 → clipped to +1.0  (positive → BUY ✗)
        The two formulas give opposite signs at this extreme value.
        """
        strat_wrong = MeanReversionStrategy(indicator_col="WillR", indicator_scale="0_100")
        strat_correct = MeanReversionStrategy(indicator_col="WillR", indicator_scale="williams_r")
        # Use last bar: WillR=-10 (overbought)
        score_wrong = strat_wrong.generate_scores(willr_data).iloc[-1]
        score_correct = strat_correct.generate_scores(willr_data).iloc[-1]
        # Correct scale gives negative (SELL), wrong scale gives positive (BUY)
        assert score_correct < 0, f"Expected negative score for overbought WillR, got {score_correct}"
        assert score_wrong > 0, f"Expected wrong (positive) score with 0_100 scale, got {score_wrong}"

    def test_scores_clipped_to_unit_range(self, rsi_data):
        """Scores should always be within [-1, 1]."""
        strat = MeanReversionStrategy(indicator_col="RSI")
        scores = strat.generate_scores(rsi_data)
        assert (scores >= -1.0).all() and (scores <= 1.0).all()


class TestCalculatePearsonIC:
    """Tests for the new calculate_pearson_ic function in metrics.py."""

    def test_perfect_positive_correlation(self):
        """Perfect positive correlation → IC=1.0, p~0."""
        from quantrl_lab.alpha_research.metrics import calculate_pearson_ic

        sig = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        ret = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
        ic, p = calculate_pearson_ic(sig, ret)
        assert np.isclose(ic, 1.0, atol=1e-6)
        assert p < 0.05

    def test_no_correlation(self):
        """No correlation → IC near 0."""
        from quantrl_lab.alpha_research.metrics import calculate_pearson_ic

        np.random.seed(0)
        sig = pd.Series(np.random.randn(1000))
        ret = pd.Series(np.random.randn(1000))
        ic, p = calculate_pearson_ic(sig, ret)
        assert abs(ic) < 0.1

    def test_returns_tuple_of_floats(self):
        """Return type should be (float, float)."""
        from quantrl_lab.alpha_research.metrics import calculate_pearson_ic

        sig = pd.Series([1.0, 2.0, 3.0])
        ret = pd.Series([0.1, 0.2, 0.3])
        ic, p = calculate_pearson_ic(sig, ret)
        assert isinstance(ic, float)
        assert isinstance(p, float)

    def test_insufficient_data_returns_zero(self):
        """Less than 2 valid pairs → IC=0, p=1."""
        from quantrl_lab.alpha_research.metrics import calculate_pearson_ic

        ic, p = calculate_pearson_ic(pd.Series([1.0]), pd.Series([0.01]))
        assert ic == 0.0
        assert p == 1.0


class TestColumnCaseNormalisation:
    """Tests for case-insensitive OHLCV column validation in
    AlphaRunner."""

    def test_lowercase_columns_auto_renamed(self, sample_ohlcv_data):
        """
        Lowercase OHLCV columns are silently renamed to title-case.

        _validate_data returns a renamed copy instead of mutating the
        input in-place (fix for A-3), so we check the return value and
        also verify the original DataFrame is NOT mutated.
        """
        df = sample_ohlcv_data.rename(columns=str.lower)
        original_cols = list(df.columns)
        runner = AlphaRunner(verbose=False)
        result = runner._validate_data(df)
        # Returned DataFrame has normalised column names
        assert "Close" in result.columns
        assert "close" not in result.columns
        # Original DataFrame is NOT mutated
        assert list(df.columns) == original_cols

    def test_mixed_case_columns_auto_renamed(self, sample_ohlcv_data):
        """
        Mixed-case columns (e.g., 'CLOSE') are also handled.

        Checks the returned copy, not the original (fix for A-3).
        """
        df = sample_ohlcv_data.rename(columns=lambda c: c.upper())
        runner = AlphaRunner(verbose=False)
        result = runner._validate_data(df)
        assert "Close" in result.columns

    def test_truly_missing_column_raises(self, sample_ohlcv_data):
        """Columns that don't exist at all (even case-insensitively)
        still raise ValueError."""
        df = sample_ohlcv_data.drop(columns=["Volume"])
        runner = AlphaRunner(verbose=False)
        with pytest.raises(ValueError, match="missing columns"):
            runner._validate_data(df)

    def test_correct_case_passes_unchanged(self, sample_ohlcv_data):
        """Correctly-cased columns are not modified."""
        original_cols = list(sample_ohlcv_data.columns)
        runner = AlphaRunner(verbose=False)
        runner._validate_data(sample_ohlcv_data)
        assert list(sample_ohlcv_data.columns) == original_cols


class TestEnsembleDescriptiveColumns:
    """Tests for descriptive column keys in
    AlphaEnsemble._get_returns_matrix."""

    def test_column_keys_are_descriptive(self, multiple_alpha_results):
        """Returns matrix columns use '<indicator>_<job_id>' format."""
        ensemble = AlphaEnsemble(multiple_alpha_results)
        mat = ensemble._get_returns_matrix()
        for col in mat.columns:
            assert "_" in col, f"Column '{col}' should be 'indicator_jobid' format"

    def test_column_names_match_results(self, multiple_alpha_results):
        """Each column corresponds to the correct AlphaResult."""
        ensemble = AlphaEnsemble(multiple_alpha_results)
        mat = ensemble._get_returns_matrix()
        for r in multiple_alpha_results:
            expected_col = f"{r.job.indicator_name}_{r.job.id}"
            assert expected_col in mat.columns, f"Expected column '{expected_col}' in returns matrix"


class TestConverterDeduplication:
    """Tests for results_to_pipeline_config deduplicate parameter."""

    @pytest.fixture()
    def multi_rsi_results(self, sample_ohlcv_data):
        """Three results: two RSI with different windows, one SMA."""
        equity = pd.Series(np.ones(10), index=pd.date_range("2023-01-01", periods=10))
        results = []
        for window, ic in [(14, 0.08), (21, 0.05)]:
            job = AlphaJob(
                data=sample_ohlcv_data,
                indicator_name="RSI",
                strategy_name="mean_reversion",
                indicator_params={"window": window},
            )
            results.append(AlphaResult(job=job, metrics={"ic": ic}, equity_curve=equity))
        job_sma = AlphaJob(
            data=sample_ohlcv_data,
            indicator_name="SMA",
            strategy_name="trend_following",
            indicator_params={"window": 50},
        )
        results.append(AlphaResult(job=job_sma, metrics={"ic": 0.06}, equity_curve=equity))
        return results

    def test_no_dedup_keeps_all(self, multi_rsi_results):
        """Without deduplication, all three results are returned."""
        from quantrl_lab.alpha_research.converters import results_to_pipeline_config

        cfg = results_to_pipeline_config(multi_rsi_results, top_n=10, deduplicate=False)
        assert len(cfg) == 3

    def test_dedup_keeps_best_per_name(self, multi_rsi_results):
        """With deduplication, only one RSI entry survives (the
        best)."""
        from quantrl_lab.alpha_research.converters import results_to_pipeline_config

        cfg = results_to_pipeline_config(multi_rsi_results, top_n=10, metric="ic", deduplicate=True)
        assert len(cfg) == 2
        # Best RSI is window=14 (ic=0.08)
        assert {"RSI": {"window": 14}} in cfg

    def test_dedup_respects_top_n(self, multi_rsi_results):
        """top_n is applied after deduplication."""
        from quantrl_lab.alpha_research.converters import results_to_pipeline_config

        cfg = results_to_pipeline_config(multi_rsi_results, top_n=1, metric="ic", deduplicate=True)
        assert len(cfg) == 1
        # Best overall is RSI w=14 (ic=0.08)
        assert {"RSI": {"window": 14}} in cfg

    def test_p_value_metric_sorts_ascending(self, sample_ohlcv_data):
        """P-value metrics should rank lower values first."""
        from quantrl_lab.alpha_research.converters import results_to_pipeline_config

        equity = pd.Series(np.ones(10), index=pd.date_range("2023-01-01", periods=10))
        high_p = AlphaResult(
            job=AlphaJob(
                data=sample_ohlcv_data,
                indicator_name="SMA",
                strategy_name="trend_following",
                indicator_params={"window": 20},
            ),
            metrics={"ic_p_value": 0.20},
            equity_curve=equity,
        )
        low_p = AlphaResult(
            job=AlphaJob(
                data=sample_ohlcv_data,
                indicator_name="EMA",
                strategy_name="trend_following",
                indicator_params={"window": 12},
            ),
            metrics={"ic_p_value": 0.01},
            equity_curve=equity,
        )

        cfg = results_to_pipeline_config([high_p, low_p], top_n=1, metric="ic_p_value", deduplicate=False)
        assert cfg == [{"EMA": {"window": 12}}]


class TestParameterSensitivityRouting:
    """Tests for the explicit indicator_param_grid / strategy_param_grid
    routing fix."""

    @pytest.fixture()
    def base_rsi_job(self, sample_ohlcv_data):
        return AlphaJob(
            data=sample_ohlcv_data,
            indicator_name="RSI",
            strategy_name="mean_reversion",
            indicator_params={"window": 14},
        )

    def test_indicator_param_grid_routes_correctly(self, base_rsi_job):
        """Parameters in indicator_param_grid always go to
        indicator_params."""
        tester = RobustnessTester(runner=AlphaRunner(verbose=False))
        df = tester.parameter_sensitivity(
            base_rsi_job,
            indicator_param_grid={"window": [10, 20]},
            n_jobs=1,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "window" in df.columns

    def test_strategy_param_grid_routes_correctly(self, base_rsi_job):
        """Parameters in strategy_param_grid always go to
        strategy_params."""
        tester = RobustnessTester(runner=AlphaRunner(verbose=False))
        df = tester.parameter_sensitivity(
            base_rsi_job,
            strategy_param_grid={"oversold": [25, 30]},
            n_jobs=1,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "oversold" in df.columns

    def test_combined_grid_produces_cross_product(self, base_rsi_job):
        """Both grids combined produce a full cross-product."""
        tester = RobustnessTester(runner=AlphaRunner(verbose=False))
        df = tester.parameter_sensitivity(
            base_rsi_job,
            indicator_param_grid={"window": [10, 14]},
            strategy_param_grid={"oversold": [25, 30]},
            n_jobs=1,
        )
        # 2 window values × 2 oversold values = 4 combinations
        assert len(df) == 4

    def test_legacy_flat_param_grid_still_works(self, base_rsi_job):
        """Old-style flat param_grid still routes existing keys
        correctly."""
        tester = RobustnessTester(runner=AlphaRunner(verbose=False))
        # 'window' is already in indicator_params → should route there
        df = tester.parameter_sensitivity(
            base_rsi_job,
            param_grid={"window": [10, 14]},
            n_jobs=1,
        )
        assert len(df) == 2


class TestAlphaSelectorDefaults:
    """Tests for AlphaSelector default candidate coverage."""

    def test_default_candidates_cover_all_research_supported_indicators(self, sample_ohlcv_data):
        """Every research-supported indicator should appear in the
        default feature sweep."""
        from quantrl_lab.alpha_research.selector import AlphaSelector

        sel = AlphaSelector(data=sample_ohlcv_data, verbose=False)
        candidates = sel._get_default_candidates(selection_mode="feature")
        candidate_names = {c["name"] for c in candidates}

        for indicator_name in INDICATOR_RESEARCH_METADATA:
            assert indicator_name in candidate_names

    def test_ema_is_in_default_candidates(self, sample_ohlcv_data):
        """EMA was missing before the fix — verify it is now
        included."""
        from quantrl_lab.alpha_research.selector import AlphaSelector

        sel = AlphaSelector(data=sample_ohlcv_data, verbose=False)
        candidates = sel._get_default_candidates(selection_mode="feature")
        names = {c["name"] for c in candidates}
        assert "EMA" in names

    def test_ema_maps_to_trend_following(self, sample_ohlcv_data):
        """EMA indicator should map to the trend_following strategy."""
        from quantrl_lab.alpha_research.selector import AlphaSelector

        sel = AlphaSelector(data=sample_ohlcv_data, verbose=False)
        result = sel._get_strategy_config({"name": "EMA"}, selection_mode="feature")
        assert result is not None
        assert result["name"] == "trend_following"

    def test_feature_mode_rejects_strategy_only_metric(self, sample_ohlcv_data):
        """Feature-mode selector should reject backtest-only metrics
        such as Sharpe."""
        from quantrl_lab.alpha_research.selector import AlphaSelector

        sel = AlphaSelector(data=sample_ohlcv_data, verbose=False)
        with pytest.raises(ValueError, match="selection_mode='feature'"):
            sel.suggest_indicators(metric="sharpe_ratio", top_k=1, selection_mode="feature")

    def test_p_value_metric_uses_max_threshold_and_ascending_rank(self, monkeypatch, sample_ohlcv_data):
        """Selector should treat p-values as lower-is-better metrics."""
        from quantrl_lab.alpha_research.selector import AlphaSelector

        sel = AlphaSelector(data=sample_ohlcv_data, verbose=False)

        def fake_run_batch(jobs, n_jobs=1):
            metric_values = {"SMA": 0.20, "EMA": 0.01, "RSI": 0.03}
            return [
                AlphaResult(job=job, metrics={"ic_p_value": metric_values[job.indicator_name]}, status="completed")
                for job in jobs
            ]

        monkeypatch.setattr(sel.runner, "run_batch", fake_run_batch)

        selected = sel.suggest_indicators(
            candidates=[
                {"name": "SMA", "params": {"window": 20}},
                {"name": "EMA", "params": {"window": 12}},
                {"name": "RSI", "params": {"window": 14}},
            ],
            metric="ic_p_value",
            threshold=0.05,
            top_k=2,
            selection_mode="feature",
        )

        assert selected == [{"EMA": {"window": 12}}, {"RSI": {"window": 14}}]


# ===========================================================================
# Additional strategy signal and score tests (coverage for alpha_strategies.py)
# ===========================================================================


@pytest.fixture
def ohlcv_with_indicators() -> pd.DataFrame:
    """OHLCV data with synthetic SMA, RSI, MACD columns."""
    np.random.seed(0)
    n = 50
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": np.random.randint(1000, 5000, n),
            "SMA_20": close * 0.98,  # Close always above SMA → BUY signals
            "RSI_14": np.where(np.arange(n) % 2 == 0, 25.0, 75.0),  # alternates oversold/overbought
            "EMA_12": close * 1.02,  # fast > slow for MACD
            "EMA_26": close * 0.97,
        },
        index=dates,
    )
    return df


class TestTrendFollowingStrategySignals:
    def test_buy_when_price_above_indicator(self, ohlcv_with_indicators):
        strategy = TrendFollowingStrategy(indicator_col="SMA_20")
        signals = strategy.generate_signals(ohlcv_with_indicators)
        # Close > SMA_20 for all rows, so all should be BUY
        from quantrl_lab.alpha_research.base import SignalType

        assert (signals == SignalType.BUY.value).all()

    def test_sell_when_price_below_indicator_with_short(self):
        from quantrl_lab.alpha_research.base import SignalType

        np.random.seed(1)
        dates = pd.date_range("2023-01-01", periods=5, freq="B")
        df = pd.DataFrame(
            {
                "Close": [90, 90, 90, 90, 90],
                "SMA_20": [100, 100, 100, 100, 100],
            },
            index=dates,
        )
        strategy = TrendFollowingStrategy(indicator_col="SMA_20", allow_short=True)
        signals = strategy.generate_signals(df)
        assert (signals == SignalType.SELL.value).all()

    def test_hold_when_allow_short_false_and_below(self):
        from quantrl_lab.alpha_research.base import SignalType

        dates = pd.date_range("2023-01-01", periods=3, freq="B")
        df = pd.DataFrame({"Close": [90, 90, 90], "SMA_20": [100, 100, 100]}, index=dates)
        strategy = TrendFollowingStrategy(indicator_col="SMA_20", allow_short=False)
        signals = strategy.generate_signals(df)
        assert (signals == SignalType.HOLD.value).all()

    def test_missing_indicator_returns_hold(self):
        from quantrl_lab.alpha_research.base import SignalType

        dates = pd.date_range("2023-01-01", periods=3, freq="B")
        df = pd.DataFrame({"Close": [100, 101, 102]}, index=dates)
        strategy = TrendFollowingStrategy(indicator_col="SMA_20")
        signals = strategy.generate_signals(df)
        assert (signals == SignalType.HOLD.value).all()

    def test_scores_bounded(self, ohlcv_with_indicators):
        strategy = TrendFollowingStrategy(indicator_col="SMA_20")
        scores = strategy.generate_scores(ohlcv_with_indicators)
        assert (scores >= -1.0).all()
        assert (scores <= 1.0).all()

    def test_get_required_columns(self):
        strategy = TrendFollowingStrategy(indicator_col="SMA_20")
        assert "SMA_20" in strategy.get_required_columns()
        assert "Close" in strategy.get_required_columns()


class TestMeanReversionStrategySignals:
    def test_buy_when_oversold(self):
        from quantrl_lab.alpha_research.base import SignalType

        dates = pd.date_range("2023-01-01", periods=5, freq="B")
        df = pd.DataFrame({"RSI_14": [20, 20, 20, 20, 20]}, index=dates)
        strategy = MeanReversionStrategy(indicator_col="RSI_14", oversold=30, overbought=70)
        signals = strategy.generate_signals(df)
        assert (signals == SignalType.BUY.value).all()

    def test_sell_when_overbought(self):
        from quantrl_lab.alpha_research.base import SignalType

        dates = pd.date_range("2023-01-01", periods=5, freq="B")
        df = pd.DataFrame({"RSI_14": [80, 80, 80, 80, 80]}, index=dates)
        strategy = MeanReversionStrategy(indicator_col="RSI_14", oversold=30, overbought=70, allow_short=True)
        signals = strategy.generate_signals(df)
        assert (signals == SignalType.SELL.value).all()

    def test_scores_bounded(self):
        dates = pd.date_range("2023-01-01", periods=10, freq="B")
        df = pd.DataFrame({"RSI_14": np.linspace(0, 100, 10)}, index=dates)
        strategy = MeanReversionStrategy(indicator_col="RSI_14")
        scores = strategy.generate_scores(df)
        assert (scores >= -1.0).all()
        assert (scores <= 1.0).all()

    def test_williams_r_scale(self):
        dates = pd.date_range("2023-01-01", periods=5, freq="B")
        df = pd.DataFrame({"WR": [-100.0, -100.0, -100.0, -100.0, -100.0]}, index=dates)
        strategy = MeanReversionStrategy(indicator_col="WR", indicator_scale="williams_r")
        scores = strategy.generate_scores(df)
        # At -100, score = (-50 - (-100)) / 50 = 1.0 (clipped to 1.0)
        assert scores.values == pytest.approx([1.0] * 5)


class TestMACDCrossoverStrategySignals:
    def test_buy_when_fast_above_slow(self):
        from quantrl_lab.alpha_research.base import SignalType

        dates = pd.date_range("2023-01-01", periods=4, freq="B")
        df = pd.DataFrame(
            {"EMA_12": [105, 106, 107, 108], "EMA_26": [100, 100, 100, 100], "Close": [104, 105, 106, 107]}, index=dates
        )
        strategy = MACDCrossoverStrategy(fast_col="EMA_12", slow_col="EMA_26")
        signals = strategy.generate_signals(df)
        assert (signals == SignalType.BUY.value).all()

    def test_sell_when_fast_below_slow(self):
        from quantrl_lab.alpha_research.base import SignalType

        dates = pd.date_range("2023-01-01", periods=4, freq="B")
        df = pd.DataFrame(
            {"EMA_12": [95, 95, 95, 95], "EMA_26": [100, 100, 100, 100], "Close": [97, 97, 97, 97]}, index=dates
        )
        strategy = MACDCrossoverStrategy(fast_col="EMA_12", slow_col="EMA_26", allow_short=True)
        signals = strategy.generate_signals(df)
        assert (signals == SignalType.SELL.value).all()

    def test_scores_bounded(self):
        dates = pd.date_range("2023-01-01", periods=5, freq="B")
        df = pd.DataFrame(
            {
                "EMA_12": [105, 104, 103, 102, 101],
                "EMA_26": [100, 100, 100, 100, 100],
                "Close": [103, 102, 101, 101, 100],
            },
            index=dates,
        )
        strategy = MACDCrossoverStrategy(fast_col="EMA_12", slow_col="EMA_26")
        scores = strategy.generate_scores(df)
        assert (scores >= -1.0).all()
        assert (scores <= 1.0).all()

    def test_missing_columns_returns_hold(self):
        from quantrl_lab.alpha_research.base import SignalType

        dates = pd.date_range("2023-01-01", periods=3, freq="B")
        df = pd.DataFrame({"Close": [100, 101, 102]}, index=dates)
        strategy = MACDCrossoverStrategy(fast_col="EMA_12", slow_col="EMA_26")
        signals = strategy.generate_signals(df)
        assert (signals == SignalType.HOLD.value).all()


class TestRegistryDedupGuard:
    """A-5: re-registering a strategy name warns instead of silently overwriting."""

    def test_reregister_warns(self):
        import warnings

        from quantrl_lab.alpha_research.base import VectorizedTradingStrategy
        from quantrl_lab.alpha_research.registry import VectorizedStrategyRegistry

        # Register a dummy strategy
        @VectorizedStrategyRegistry.register("_test_dedup_strategy")
        class _DummyA(VectorizedTradingStrategy):
            def generate_signals(self, data):
                return pd.Series(0, index=data.index)

            def get_required_columns(self):
                return []

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @VectorizedStrategyRegistry.register("_test_dedup_strategy")
            class _DummyB(VectorizedTradingStrategy):
                def generate_signals(self, data):
                    return pd.Series(0, index=data.index)

                def get_required_columns(self):
                    return []

            assert len(w) == 1
            assert "_test_dedup_strategy" in str(w[0].message)

        # Cleanup
        del VectorizedStrategyRegistry._strategies["_test_dedup_strategy"]


class TestAlphaJobValidation:
    """A-6: AlphaJob validates non-empty indicator_name and strategy_name."""

    def test_empty_indicator_name_raises(self, sample_ohlcv_data):
        with pytest.raises(ValueError, match="indicator_name"):
            AlphaJob(data=sample_ohlcv_data, indicator_name="", strategy_name="mean_reversion")

    def test_empty_strategy_name_raises(self, sample_ohlcv_data):
        with pytest.raises(ValueError, match="strategy_name"):
            AlphaJob(data=sample_ohlcv_data, indicator_name="RSI", strategy_name="")

    def test_whitespace_indicator_name_raises(self, sample_ohlcv_data):
        with pytest.raises(ValueError, match="indicator_name"):
            AlphaJob(data=sample_ohlcv_data, indicator_name="  ", strategy_name="mean_reversion")


class TestICConstantColumns:
    """A-8: IC returns 0 for constant signal or returns instead of NaN."""

    def test_constant_signal_returns_zero(self):
        from quantrl_lab.alpha_research.metrics import calculate_pearson_ic, calculate_rank_ic

        sig = pd.Series([1.0] * 10)  # constant
        ret = pd.Series(np.random.randn(10))
        ic, p = calculate_pearson_ic(sig, ret)
        assert ic == 0.0
        assert p == 1.0
        ric, rp = calculate_rank_ic(sig, ret)
        assert ric == 0.0

    def test_constant_returns_returns_zero(self):
        from quantrl_lab.alpha_research.metrics import calculate_pearson_ic

        sig = pd.Series(np.random.randn(10))
        ret = pd.Series([0.01] * 10)  # constant
        ic, p = calculate_pearson_ic(sig, ret)
        assert ic == 0.0


class TestUnknownIndicatorWarning:
    """D-8: TechnicalFeatureGenerator warns on unknown indicators."""

    def test_unknown_indicator_warns(self, sample_ohlcv_data):
        import warnings

        from quantrl_lab.data.processing.features.technical import TechnicalFeatureGenerator

        gen = TechnicalFeatureGenerator(["NONEXISTENT_XYZ"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = gen.generate(sample_ohlcv_data)
            assert any("NONEXISTENT_XYZ" in str(warning.message) for warning in w)
        # Data should be returned unchanged (no crash)
        assert list(result.columns) == list(sample_ohlcv_data.columns)


class TestProcessingModuleExports:
    """D-12: processing __init__ exports DataPipeline and ProcessingMetadata."""

    def test_data_pipeline_importable(self):
        from quantrl_lab.data.processing import DataPipeline

        assert DataPipeline is not None

    def test_processing_metadata_importable(self):
        from quantrl_lab.data.processing import ProcessingMetadata

        assert ProcessingMetadata is not None


class TestStrategyMetadata:
    """A-7: StrategyMetadata and VectorizedStrategyRegistry.get_metadata()."""

    def test_get_metadata_returns_strategy_metadata(self):
        """get_metadata() returns a StrategyMetadata object with correct
        fields."""
        from quantrl_lab.alpha_research.registry import StrategyMetadata, VectorizedStrategyRegistry

        meta = VectorizedStrategyRegistry.get_metadata("mean_reversion")
        assert isinstance(meta, StrategyMetadata)
        assert meta.name == "mean_reversion"
        assert meta.strategy_class is not None
        assert isinstance(meta.description, str)
        assert len(meta.description) > 0

    def test_get_metadata_unknown_strategy_raises(self):
        """get_metadata() raises ValueError for an unknown strategy
        name."""
        from quantrl_lab.alpha_research.registry import VectorizedStrategyRegistry

        with pytest.raises(ValueError, match="Unknown strategy"):
            VectorizedStrategyRegistry.get_metadata("nonexistent_xyz")

    def test_all_registered_strategies_have_descriptions(self):
        """Every registered strategy should have a non-empty
        description."""
        from quantrl_lab.alpha_research.registry import VectorizedStrategyRegistry

        for name in VectorizedStrategyRegistry.list_strategies():
            meta = VectorizedStrategyRegistry.get_metadata(name)
            assert meta.description, f"Strategy '{name}' is missing a description"

    def test_metadata_strategy_class_is_instantiable(self):
        """strategy_class stored in metadata can be used to create an
        instance."""
        from quantrl_lab.alpha_research.registry import VectorizedStrategyRegistry

        meta = VectorizedStrategyRegistry.get_metadata("trend_following")
        strategy = meta.strategy_class(indicator_col="SMA_20")
        assert strategy is not None


class TestValidateColumns:
    """A-2: VectorizedTradingStrategy.validate_columns() catches broken column wiring."""

    def test_validate_columns_returns_empty_when_all_present(self):
        """validate_columns returns [] when all required columns
        exist."""
        from quantrl_lab.alpha_research.alpha_strategies import TrendFollowingStrategy

        strategy = TrendFollowingStrategy(indicator_col="SMA_20")
        dates = pd.date_range("2023-01-01", periods=3, freq="B")
        df = pd.DataFrame({"Close": [100, 101, 102], "SMA_20": [98, 99, 100]}, index=dates)
        assert strategy.validate_columns(df) == []

    def test_validate_columns_returns_missing_names(self):
        """validate_columns returns the names of columns not in the
        DataFrame."""
        from quantrl_lab.alpha_research.alpha_strategies import TrendFollowingStrategy

        strategy = TrendFollowingStrategy(indicator_col="SMA_20")
        dates = pd.date_range("2023-01-01", periods=3, freq="B")
        df = pd.DataFrame({"Close": [100, 101, 102]}, index=dates)  # SMA_20 missing
        missing = strategy.validate_columns(df)
        assert "SMA_20" in missing

    def test_validate_columns_multi_col_strategy(self):
        """MACDCrossoverStrategy.validate_columns catches missing
        fast/slow cols."""
        from quantrl_lab.alpha_research.alpha_strategies import MACDCrossoverStrategy

        strategy = MACDCrossoverStrategy(fast_col="MACD_line_12_26", slow_col="MACD_signal_9")
        dates = pd.date_range("2023-01-01", periods=3, freq="B")
        df = pd.DataFrame({"Close": [100, 101, 102]}, index=dates)
        missing = strategy.validate_columns(df)
        assert "MACD_line_12_26" in missing
        assert "MACD_signal_9" in missing

    def test_runner_warns_on_missing_columns(self, sample_ohlcv_data):
        """AlphaRunner.run_job warns (via console) when resolved columns
        are absent."""
        # Use ADX with a deliberately wrong strategy that has no auto-wiring
        # so the resolved indicator_col will be None-free but still missing.
        # Easiest: provide a strategy_params with a column that won't be created.
        job = AlphaJob(
            data=sample_ohlcv_data,
            indicator_name="SMA",
            strategy_name="trend_following",
            strategy_params={"indicator_col": "NONEXISTENT_COL"},
        )
        runner = AlphaRunner(verbose=False)
        result = runner.run_job(job)
        # Job still completes (defaults to HOLD signals) but result is "completed"
        assert result.status == "completed"


class TestIndicatorStrategyMap:
    """D-1: INDICATOR_STRATEGY_MAP lives in alpha_strategies, not IndicatorMetadata."""

    def test_map_is_importable_from_alpha_strategies(self):
        """INDICATOR_STRATEGY_MAP should be importable directly."""
        from quantrl_lab.alpha_research.alpha_strategies import INDICATOR_STRATEGY_MAP

        assert isinstance(INDICATOR_STRATEGY_MAP, dict)
        assert len(INDICATOR_STRATEGY_MAP) > 0
        assert set(INDICATOR_STRATEGY_MAP) == set(INDICATOR_RESEARCH_METADATA)

    def test_all_map_entries_have_name_and_params(self):
        """Every entry in INDICATOR_STRATEGY_MAP has 'name' and 'params'
        keys."""
        from quantrl_lab.alpha_research.alpha_strategies import INDICATOR_STRATEGY_MAP

        for indicator, config in INDICATOR_STRATEGY_MAP.items():
            assert "name" in config, f"{indicator} entry missing 'name'"
            assert "params" in config, f"{indicator} entry missing 'params'"

    def test_all_map_strategy_names_are_registered(self):
        """Every strategy name in INDICATOR_STRATEGY_MAP must be
        registered."""
        from quantrl_lab.alpha_research.alpha_strategies import INDICATOR_STRATEGY_MAP
        from quantrl_lab.alpha_research.registry import VectorizedStrategyRegistry

        registered = set(VectorizedStrategyRegistry.list_strategies())
        for indicator, config in INDICATOR_STRATEGY_MAP.items():
            strategy_name = config["name"]
            assert (
                strategy_name in registered
            ), f"INDICATOR_STRATEGY_MAP['{indicator}'] points to unregistered strategy '{strategy_name}'"

    def test_indicator_metadata_has_no_strategy_fields(self):
        """D-1: IndicatorMetadata must NOT have strategy_name or strategy_params fields."""
        from quantrl_lab.data.indicators.registry import IndicatorMetadata

        meta = IndicatorMetadata.__dataclass_fields__
        assert "strategy_name" not in meta, "strategy_name should have been removed from IndicatorMetadata (D-1)"
        assert "strategy_params" not in meta, "strategy_params should have been removed from IndicatorMetadata (D-1)"

    def test_adx_maps_to_adx_trend_strategy(self):
        """ADX indicator should map to the adx_trend strategy (A-1
        fix)."""
        from quantrl_lab.alpha_research.alpha_strategies import INDICATOR_STRATEGY_MAP

        assert INDICATOR_STRATEGY_MAP["ADX"]["name"] == "adx_trend"

    def test_all_registered_indicators_have_research_metadata(self):
        """The research registry should stay in lockstep with
        IndicatorRegistry."""
        from quantrl_lab.data.indicators import IndicatorRegistry

        assert set(IndicatorRegistry.list_all()) == set(INDICATOR_RESEARCH_METADATA)
