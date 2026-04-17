import traceback
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rich.console import Console
from rich.progress import track
from sklearn.feature_selection import mutual_info_regression

from quantrl_lab.data.indicators.registry import IndicatorRegistry

from .indicator_research import FEATURE_SELECTION_MODE
from .metrics import (
    calculate_forward_returns,
    calculate_pearson_ic,
    calculate_rank_ic,
)
from .models import AlphaJob, AlphaResult
from .registry import VectorizedStrategyRegistry

# IndicatorRegistry is used by _calculate_indicators; kept as a named import
# so the import is explicit even though resolve_columns no longer uses it.

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
            # Returns a (possibly column-normalised) copy — never the original job.data
            validated_data = self._validate_data(job.data)

            # 2. Calculate Indicators
            old_cols = set(validated_data.columns)
            data_with_indicators = self._calculate_indicators(validated_data, job.indicator_name, job.indicator_params)
            new_cols = list(set(data_with_indicators.columns) - old_cols)

            # 3. Create Strategy
            # Filter out runner-specific parameters that are not for strategy init
            strategy_params = job.strategy_params.copy()
            for param in ["ic_horizon", "initial_capital", "transaction_cost"]:
                if param in strategy_params:
                    del strategy_params[param]

            # A-7: Delegate column wiring to the strategy class itself.
            # Each strategy implements resolve_columns() so this runner no
            # longer needs a per-strategy if-chain.
            strategy_cls = VectorizedStrategyRegistry.get(job.strategy_name)
            resolved_args = strategy_cls.resolve_columns(new_cols, strategy_params)
            strategy_params.update(resolved_args)

            strategy = VectorizedStrategyRegistry.create(
                job.strategy_name, allow_short=job.allow_short, **strategy_params
            )

            # A-2: Validate that all required columns are present before generating
            # signals. Catches broken wiring early instead of silently returning
            # all-HOLD signals (which score zero IC and waste the whole job run).
            missing_cols = strategy.validate_columns(data_with_indicators)
            if missing_cols and self.verbose:
                console.print(
                    f"[yellow]Warning: {job.strategy_name} for {job.indicator_name} "
                    f"is missing columns {missing_cols}. Signals will default to HOLD.[/yellow]"
                )

            # 4. Generate Signals
            analysis_horizon = job.strategy_params.get("ic_horizon", 5)

            if job.evaluation_mode == FEATURE_SELECTION_MODE:
                scores = strategy.generate_scores(data_with_indicators)
                feature_metrics = self._analyze_predictive_power(
                    data_with_indicators,
                    scores,
                    horizon=analysis_horizon,
                    prefix="feature_",
                )
                feature_metrics.update(self._summarize_scores(scores))
                feature_metrics.update(self._alias_primary_metrics(feature_metrics, prefix="feature_"))
                return AlphaResult(
                    job=job,
                    metrics=feature_metrics,
                    scores=scores,
                    status="completed",
                )

            signals = strategy.generate_signals(data_with_indicators)

            # 5. Simulate Portfolio
            portfolio_results = self._simulate_portfolio(
                data_with_indicators,
                signals,
                initial_capital=job.strategy_params.get("initial_capital", 100000.0),
                transaction_cost=job.strategy_params.get("transaction_cost", 0.001),
            )

            # 6. Statistical Signal Analysis (IC, etc.)
            signal_analysis = self._analyze_predictive_power(
                data_with_indicators,
                signals,
                horizon=analysis_horizon,
                prefix="signal_",
            )

            # 7. Metrics
            metrics = self._calculate_metrics(portfolio_results)
            metrics.update(signal_analysis)
            metrics.update(self._alias_primary_metrics(signal_analysis, prefix="signal_"))

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

    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate input data for required columns, types, and quality.

        Returns a (possibly column-renamed) copy of the DataFrame rather than
        mutating the caller's object in-place.

        BUG FIX (A-3): The original implementation called
        ``data.rename(columns=..., inplace=True)`` which mutated the shared
        ``AlphaJob.data`` DataFrame directly. In batch runs where multiple jobs
        reference the same DataFrame, this silently renamed columns under
        subsequent jobs, causing them to fail with "missing column" errors.
        The fix returns a new DataFrame so the caller's object is never touched.

        Args:
            data (pd.DataFrame): Input market data.

        Returns:
            pd.DataFrame: Validated (and if necessary, column-normalised) copy.

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
            # Return a renamed copy — do NOT mutate the caller's DataFrame
            data = data.rename(columns=rename_map)

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

        return data

    def _calculate_indicators(
        self, data: pd.DataFrame, indicator_name: str, indicator_params: Dict[str, Any]
    ) -> pd.DataFrame:
        data_copy = data.copy()
        try:
            return IndicatorRegistry.apply(name=indicator_name, df=data_copy, **indicator_params)
        except Exception as e:
            raise ValueError(f"Failed to calculate indicator {indicator_name}: {e}")

    def _analyze_predictive_power(
        self,
        data: pd.DataFrame,
        values: pd.Series,
        horizon: int = 5,
        prefix: str = "",
    ) -> Dict[str, float]:
        """
        Analyze predictive power using Information Coefficient (IC).

        Args:
            data (pd.DataFrame): Market data with Close prices.
            values (pd.Series): Signals or continuous feature scores.
            horizon (int): Forward-looking horizon for returns (in periods).
            prefix (str): Prefix for returned metric names.

        Returns:
            Dict[str, float]: Dictionary of signal quality metrics.
        """
        forward_returns = calculate_forward_returns(data["Close"], periods=horizon)

        valid_mask = values.notna() & forward_returns.notna()
        if not valid_mask.any():
            return {
                f"{prefix}ic": 0.0,
                f"{prefix}rank_ic": 0.0,
                f"{prefix}ic_p_value": 1.0,
                f"{prefix}rank_ic_p_value": 1.0,
                f"{prefix}mutual_info": 0.0,
            }

        s = values[valid_mask]
        r = forward_returns[valid_mask]

        ic, ic_p = calculate_pearson_ic(s, r)
        rank_ic, rank_ic_p = calculate_rank_ic(s, r)

        # Mutual Information (captures non-linear relationships, unique to runner)
        try:
            mi = mutual_info_regression(s.values.reshape(-1, 1), r.values, discrete_features=False, random_state=42)[0]
        except Exception:
            mi = 0.0

        return {
            f"{prefix}ic": float(ic),
            f"{prefix}rank_ic": float(rank_ic),
            f"{prefix}ic_p_value": float(ic_p),
            f"{prefix}rank_ic_p_value": float(rank_ic_p),
            f"{prefix}mutual_info": float(mi),
        }

    def _alias_primary_metrics(self, metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
        """Expose the requested mode's predictive metrics under the
        legacy names."""
        aliased: Dict[str, float] = {}
        for key in ["ic", "rank_ic", "ic_p_value", "rank_ic_p_value", "mutual_info"]:
            prefixed_key = f"{prefix}{key}"
            if prefixed_key in metrics:
                aliased[key] = float(metrics[prefixed_key])
        return aliased

    def _summarize_scores(self, scores: pd.Series) -> Dict[str, float]:
        """Return lightweight diagnostics for feature-mode score
        distributions."""
        clean_scores = scores.dropna()
        if clean_scores.empty:
            return {"score_mean": 0.0, "score_std": 0.0, "score_abs_mean": 0.0}

        return {
            "score_mean": float(clean_scores.mean()),
            "score_std": float(clean_scores.std()),
            "score_abs_mean": float(clean_scores.abs().mean()),
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
