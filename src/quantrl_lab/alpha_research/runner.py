import traceback
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rich.console import Console
from rich.progress import track
from sklearn.feature_selection import mutual_info_regression

from quantrl_lab.data.indicators.registry import IndicatorRegistry

from .metrics import (
    calculate_forward_returns,
    calculate_pearson_ic,
    calculate_rank_ic,
)
from .models import AlphaJob, AlphaResult
from .registry import VectorizedStrategyRegistry

console = Console()


class AlphaRunner:
    """
    Executor for alpha research jobs.

    Runs vectorized backtests and statistical analysis for signals.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def run_job(self, job: AlphaJob) -> AlphaResult:
        """
        Execute a single alpha research job.

        Args:
            job (AlphaJob): The job configuration.

        Returns:
            AlphaResult: The results of the backtest and signal analysis.
        """
        if self.verbose:
            console.print(
                f"[cyan]Running Alpha Job: {job.id} "
                f"(Indicator: {job.indicator_name}, Strategy: {job.strategy_name})[/cyan]"
            )

        try:
            # 1. Validate Data
            self._validate_data(job.data)

            # 2. Calculate Indicators
            old_cols = set(job.data.columns)
            data_with_indicators = self._calculate_indicators(job.data, job.indicator_name, job.indicator_params)
            new_cols = list(set(data_with_indicators.columns) - old_cols)

            # 3. Create Strategy
            # Filter out runner-specific parameters that are not for strategy init
            strategy_params = job.strategy_params.copy()
            for param in ["ic_horizon", "initial_capital", "transaction_cost"]:
                if param in strategy_params:
                    del strategy_params[param]

            # Resolve strategy specific column arguments (pass indicator_name for registry lookup)
            strategy_cls = VectorizedStrategyRegistry.get(job.strategy_name)
            resolved_args = self._resolve_strategy_args(
                strategy_cls, new_cols, strategy_params, indicator_name=job.indicator_name
            )
            strategy_params.update(resolved_args)

            strategy = VectorizedStrategyRegistry.create(
                job.strategy_name, allow_short=job.allow_short, **strategy_params
            )

            # 4. Generate Signals
            signals = strategy.generate_signals(data_with_indicators)

            # 5. Simulate Portfolio
            portfolio_results = self._simulate_portfolio(
                data_with_indicators,
                signals,
                initial_capital=job.strategy_params.get("initial_capital", 100000.0),
                transaction_cost=job.strategy_params.get("transaction_cost", 0.001),
            )

            # 6. Statistical Signal Analysis (IC, etc.)
            signal_analysis = self._analyze_signal_predictive_power(
                data_with_indicators, signals, horizon=job.strategy_params.get("ic_horizon", 5)
            )

            # 7. Metrics
            metrics = self._calculate_metrics(portfolio_results)
            metrics.update(signal_analysis)

            return AlphaResult(
                job=job,
                metrics=metrics,
                equity_curve=portfolio_results["portfolio_values"],
                signals=signals,
                status="completed",
            )

        except Exception as e:
            tb = traceback.format_exc()
            if self.verbose:
                console.print(f"[red]Job Failed: {e}[/red]")
                console.print(tb)
            return AlphaResult(job=job, metrics={}, status="failed", error=tb)

    def _resolve_strategy_args(
        self,
        strategy_cls: Any,
        new_cols: List[str],
        current_params: Dict[str, Any],
        indicator_name: str = "",
    ) -> Dict[str, Any]:
        """
        Resolve column arguments for strategy from indicator-generated
        columns.

        Uses ``IndicatorRegistry`` metadata (``output_columns``) as the
        authoritative source of column names, so that strategy wiring does not
        break when the registry changes its naming conventions.  Hardcoded
        substring matching is kept only as a last-resort fallback for
        indicators registered without explicit ``output_columns``.

        Args:
            strategy_cls: The strategy class to instantiate.
            new_cols: Columns added to the DataFrame by the indicator calculation.
            current_params: Strategy parameters already supplied by the user
                (these are never overridden).
            indicator_name: Registry name of the indicator, used to fetch
                ``IndicatorMetadata.output_columns`` (optional but recommended).

        Returns:
            Dict[str, Any]: Additional keyword arguments to pass to the strategy.
        """
        import inspect

        init_signature = inspect.signature(strategy_cls.__init__)
        init_params = init_signature.parameters
        resolved = {}

        # --- Registry-aware column lookup ---
        # Prefer metadata.output_columns over substring guessing so the wiring
        # stays valid even if the registry renames its output columns.
        registry_cols: List[str] = []
        if indicator_name:
            try:
                meta = IndicatorRegistry.get_metadata(indicator_name)
                # Only keep columns that were actually added to the DataFrame
                registry_cols = [c for c in meta.output_columns if c in new_cols]
            except KeyError:
                pass  # Unknown indicator — fall back to substring heuristics

        def find_col(substring: str) -> Any:
            """Fallback: find first new column whose name contains
            substring."""
            matches = [c for c in new_cols if substring in c]
            if not matches and self.verbose:
                console.print(
                    f"[yellow]Warning: could not resolve column for '{substring}' "
                    f"in indicator '{indicator_name}'. New columns: {new_cols}[/yellow]"
                )
            return matches[0] if matches else None

        def registry_col(substring: str) -> Any:
            """Check registry columns first, then fall back to substring
            search."""
            matches = [c for c in registry_cols if substring in c]
            return matches[0] if matches else find_col(substring)

        # 1. Generic 'indicator_col' — single-output indicators (RSI, SMA, ATR, CCI…)
        if "indicator_col" in init_params and "indicator_col" not in current_params:
            if registry_cols:
                resolved["indicator_col"] = registry_cols[0]
            elif new_cols:
                resolved["indicator_col"] = new_cols[0]

        # 2. MACD: fast_col = MACD line, slow_col = Signal line
        if "fast_col" in init_params and "slow_col" in init_params:
            if "fast_col" not in current_params:
                resolved["fast_col"] = registry_col("MACD_line")
            if "slow_col" not in current_params:
                resolved["slow_col"] = registry_col("MACD_signal")

        # 3. Bollinger Bands: upper / lower / middle bands
        if "upper_col" in init_params and "lower_col" in init_params:
            if "upper_col" not in current_params:
                resolved["upper_col"] = registry_col("BB_upper")
            if "lower_col" not in current_params:
                resolved["lower_col"] = registry_col("BB_lower")
            if "middle_col" not in current_params and "middle_col" in init_params:
                resolved["middle_col"] = registry_col("BB_middle")

        # 4. Stochastic: %K and optional %D
        if "k_col" in init_params:
            if "k_col" not in current_params:
                resolved["k_col"] = registry_col("STOCH_%K")
            if "d_col" in init_params and "d_col" not in current_params:
                resolved["d_col"] = registry_col("STOCH_%D")

        # 5. OBV
        if "obv_col" in init_params and "obv_col" not in current_params:
            resolved["obv_col"] = registry_col("OBV")

        # 6. ADX: adx_col / pdi_col / mdi_col
        if "adx_col" in init_params and "adx_col" not in current_params:
            resolved["adx_col"] = registry_col("ADX")
        if "pdi_col" in init_params and "pdi_col" not in current_params:
            resolved["pdi_col"] = registry_col("PDI")
        if "mdi_col" in init_params and "mdi_col" not in current_params:
            resolved["mdi_col"] = registry_col("MDI")

        return resolved

    def run_batch(self, jobs: List[AlphaJob], n_jobs: int = 1) -> List[AlphaResult]:
        """
        Run a batch of jobs, optionally in parallel.

        Args:
            jobs (List[AlphaJob]): List of jobs to run.
            n_jobs (int): Number of parallel jobs. -1 for all cores, 1 for sequential.

        Returns:
            List[AlphaResult]: List of results.
        """
        if n_jobs == 1:
            # Sequential execution with progress bar
            results = []
            for job in track(jobs, description="Running Alpha Jobs...", disable=not self.verbose):
                results.append(self.run_job(job))
            return results

        # Parallel execution
        if self.verbose:
            console.print(f"[cyan]Running {len(jobs)} jobs in parallel (n_jobs={n_jobs})...[/cyan]")

        # Create a temporary runner without verbose output for parallel jobs
        runner = AlphaRunner(verbose=False)
        results = Parallel(n_jobs=n_jobs)(delayed(runner.run_job)(job) for job in jobs)

        if self.verbose:
            successful = sum(1 for r in results if r.status == "completed")
            console.print(f"[green]Completed {successful}/{len(jobs)} jobs successfully[/green]")

        return results

    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate input data for required columns, types, and quality.

        Args:
            data (pd.DataFrame): Input market data.

        Raises:
            ValueError: If validation fails.
        """
        required_columns = ["Open", "High", "Low", "Close", "Volume"]

        # Case-insensitive check first so we can offer a helpful suggestion
        col_lower_map = {c.lower(): c for c in data.columns}
        missing = []
        rename_map = {}
        for req in required_columns:
            if req in data.columns:
                continue  # exact match — fine
            elif req.lower() in col_lower_map:
                # Column exists but with different casing — auto-rename
                rename_map[col_lower_map[req.lower()]] = req
            else:
                missing.append(req)

        if missing:
            raise ValueError(f"Data missing columns: {missing}")

        if rename_map:
            if self.verbose:
                console.print(f"[yellow]Auto-normalising column names: {rename_map}[/yellow]")
            data.rename(columns=rename_map, inplace=True)
        if data.empty:
            raise ValueError("Data is empty")

        # Check for NaNs in critical columns
        nan_counts = data[required_columns].isna().sum()
        if nan_counts.any():
            if self.verbose:
                console.print(
                    f"[yellow]Warning: Data contains NaNs in OHLCV columns:\n{nan_counts[nan_counts > 0]}[/yellow]"
                )
            # Don't raise, but warn - strategies might handle it or it will fail downstream

        # Ensure numeric types
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                raise ValueError(f"Column '{col}' is not numeric (dtype: {data[col].dtype})")

        # Check for price consistency (High >= Low, etc.)
        if (data["High"] < data["Low"]).any():
            problematic_rows = (data["High"] < data["Low"]).sum()
            raise ValueError(f"Invalid data: High < Low in {problematic_rows} rows")

    def _calculate_indicators(
        self, data: pd.DataFrame, indicator_name: str, indicator_params: Dict[str, Any]
    ) -> pd.DataFrame:
        data_copy = data.copy()
        try:
            return IndicatorRegistry.apply(name=indicator_name, df=data_copy, **indicator_params)
        except Exception as e:
            raise ValueError(f"Failed to calculate indicator {indicator_name}: {e}")

    def _analyze_signal_predictive_power(
        self, data: pd.DataFrame, signals: pd.Series, horizon: int = 5
    ) -> Dict[str, float]:
        """
        Analyze the predictive power of signals using Information
        Coefficient (IC).

        Args:
            data (pd.DataFrame): Market data with Close prices.
            signals (pd.Series): Trading signals.
            horizon (int): Forward-looking horizon for returns (in periods).

        Returns:
            Dict[str, float]: Dictionary of signal quality metrics.
        """
        forward_returns = calculate_forward_returns(data["Close"], periods=horizon)

        valid_mask = signals.notna() & forward_returns.notna()
        if not valid_mask.any():
            return {"ic": 0.0, "rank_ic": 0.0, "ic_p_value": 1.0, "rank_ic_p_value": 1.0, "mutual_info": 0.0}

        s = signals[valid_mask]
        r = forward_returns[valid_mask]

        ic, ic_p = calculate_pearson_ic(s, r)
        rank_ic, rank_ic_p = calculate_rank_ic(s, r)

        # Mutual Information (captures non-linear relationships, unique to runner)
        try:
            mi = mutual_info_regression(s.values.reshape(-1, 1), r.values, discrete_features=False, random_state=42)[0]
        except Exception:
            mi = 0.0

        return {
            "ic": float(ic),
            "rank_ic": float(rank_ic),
            "ic_p_value": float(ic_p),
            "rank_ic_p_value": float(rank_ic_p),
            "mutual_info": float(mi),
        }

    def _simulate_portfolio(
        self, data: pd.DataFrame, signals: pd.Series, initial_capital: float = 100000.0, transaction_cost: float = 0.001
    ) -> Dict[str, Any]:
        """Vectorized portfolio simulation."""
        price = data["Close"]
        returns = price.pct_change().fillna(0)

        # Signals at t are acted upon at Close of t, realized at t+1
        pos = signals.shift(1).fillna(0)

        strat_returns = pos * returns

        # Transaction costs on changes in position
        trades = pos.diff().abs().fillna(0)
        costs = trades * transaction_cost

        net_returns = strat_returns - costs

        # Cumulative returns and equity curve
        cum_returns = (1 + net_returns).cumprod()
        equity_curve = initial_capital * cum_returns

        return {"portfolio_values": equity_curve, "strategy_returns": net_returns, "positions": pos, "trades": trades}

    def _calculate_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.

        Args:
            results (Dict[str, Any]): Portfolio simulation results.

        Returns:
            Dict[str, float]: Dictionary of performance metrics.
        """
        returns = results["strategy_returns"]
        trades = results["trades"]

        if len(returns) < 2:
            return {}

        total_return = (1 + returns).prod() - 1

        ann_factor = 252
        ann_return = returns.mean() * ann_factor
        volatility = returns.std() * np.sqrt(ann_factor)

        # Sharpe Ratio
        risk_free_rate = 0.02
        sharpe = (ann_return - risk_free_rate) / volatility if volatility > 0 else 0

        # Sortino Ratio (only penalize downside volatility)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(ann_factor)
        sortino = (ann_return - risk_free_rate) / downside_std if downside_std > 0 else 0

        # Drawdown analysis
        equity_curve = (1 + returns).cumprod()
        max_drawdown = (equity_curve / equity_curve.cummax() - 1).min()

        # Calmar Ratio (Return / Max Drawdown)
        calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Trade-based metrics
        trade_returns = returns[trades > 0]
        if len(trade_returns) > 0:
            win_rate = (trade_returns > 0).mean()
            winning_trades = trade_returns[trade_returns > 0]
            losing_trades = trade_returns[trade_returns < 0]

            avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
            avg_loss = abs(losing_trades.mean()) if len(losing_trades) > 0 else 0
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

            # Profit Factor
            total_wins = winning_trades.sum() if len(winning_trades) > 0 else 0
            total_losses = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
        else:
            win_rate = 0
            win_loss_ratio = 0
            profit_factor = 0

        # Turnover (average daily position change)
        turnover = trades.mean()

        return {
            "total_return": float(total_return),
            "annual_return": float(ann_return),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "calmar_ratio": float(calmar),
            "max_drawdown": float(max_drawdown),
            "volatility": float(volatility),
            "win_rate": float(win_rate),
            "win_loss_ratio": float(win_loss_ratio),
            "profit_factor": float(profit_factor),
            "turnover": float(turnover),
        }
