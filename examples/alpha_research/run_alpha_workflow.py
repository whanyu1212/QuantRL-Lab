"""
Alpha Research Workflow.

Demonstrates the full alpha research pipeline:
1. Load real market data via YFinance
2. Use AlphaSelector to discover the best indicators for the symbol
3. Run AlphaEnsemble to combine top strategies
4. Run RobustnessTester for parameter sensitivity and sub-period stability
5. Generate an interactive HTML report via AlphaVisualizer
"""

from rich.console import Console
from rich.table import Table

from quantrl_lab.alpha_research import (
    AlphaEnsemble,
    AlphaJob,
    AlphaResult,
    AlphaRunner,
    AlphaSelector,
    AlphaVisualizer,
    RobustnessTester,
)
from quantrl_lab.data.sources.yfinance_loader import YFinanceDataLoader

console = Console()


def main():
    # ── 1. Load data ─────────────────────────────────────────────────────────
    console.rule("[bold cyan]Alpha Research Workflow[/bold cyan]")
    console.print("[cyan]Loading data...[/cyan]")
    loader = YFinanceDataLoader()
    data = loader.get_historical_ohlcv_data(
        symbols="AAPL",
        start="2022-01-01",
        end="2023-12-31",
        timeframe="1d",
    )
    console.print(f"Loaded {len(data)} days of AAPL data\n")

    # ── 2. Select best indicators via AlphaSelector ───────────────────────────
    console.rule("[bold cyan]Step 1: Indicator Selection[/bold cyan]")
    selector = AlphaSelector(data)
    selected = selector.suggest_indicators(metric="ic", threshold=0.02, top_k=5, selection_mode="feature")

    if not selected:
        console.print("[yellow]No indicators passed the IC threshold. Using defaults.[/yellow]")
        selected = [
            {"RSI": {"window": 14}},
            {"SMA": {"window": 50}},
            {"ATR": {"window": 14}},
            {"BB": {"window": 20, "num_std": 2.0}},
        ]

    # ── 3. Run full batch with the default candidate grid for richer results ──
    # AlphaSelector internally runs a batch — we reuse the runner for the
    # ensemble and robustness steps so we need the raw AlphaResult objects.
    console.rule("[bold cyan]Step 2: Full Candidate Evaluation[/bold cyan]")
    runner = AlphaRunner(verbose=True)

    # Reconstruct jobs from the default candidate grid so we have AlphaResult
    # objects with equity curves (AlphaSelector doesn't expose them directly).
    candidate_map = {
        "RSI": {"strategy": "mean_reversion", "params_list": [{"window": 7}, {"window": 14}, {"window": 21}]},
        "SMA": {"strategy": "trend_following", "params_list": [{"window": 20}, {"window": 50}]},
        "EMA": {"strategy": "trend_following", "params_list": [{"window": 12}, {"window": 26}]},
        "MACD": {"strategy": "macd_crossover", "params_list": [{"fast": 12, "slow": 26, "signal": 9}]},
        "BB": {"strategy": "bollinger_bands", "params_list": [{"window": 20, "num_std": 2.0}]},
        "ATR": {"strategy": "volatility_breakout", "params_list": [{"window": 14}]},
        "CCI": {"strategy": "cci_reversal", "params_list": [{"window": 20}]},
        "STOCH": {"strategy": "stochastic", "params_list": [{"k_window": 14, "d_window": 3}]},
        "OBV": {"strategy": "obv_trend", "params_list": [{}]},
        "ADX": {"strategy": "adx_trend", "params_list": [{"window": 14}]},
    }

    jobs = []
    for indicator, cfg in candidate_map.items():
        for params in cfg["params_list"]:
            jobs.append(
                AlphaJob(
                    data=data,
                    indicator_name=indicator,
                    strategy_name=cfg["strategy"],
                    indicator_params=params,
                    strategy_params={"ic_horizon": 5},
                )
            )

    results = runner.run_batch(jobs, n_jobs=2)
    completed = [r for r in results if r.status == "completed"]
    console.print(f"\n[green]{len(completed)}/{len(jobs)} jobs completed successfully[/green]")

    # ── 4. Results summary table ──────────────────────────────────────────────
    console.rule("[bold cyan]Step 3: Results Summary[/bold cyan]")
    table = Table(title="Alpha Research Results")
    for col, style in [
        ("Indicator", "magenta"),
        ("Strategy", "yellow"),
        ("Sharpe", None),
        ("Sortino", None),
        ("Calmar", None),
        ("IC", None),
        ("Rank IC", None),
        ("Mut. Info", None),
        ("Win Rate", None),
        ("Status", "green"),
    ]:
        table.add_column(col, style=style, justify="right" if style is None else "left")

    for r in results:
        m = r.metrics
        if r.status == "completed":
            table.add_row(
                r.job.indicator_name,
                r.job.strategy_name,
                f"{m.get('sharpe_ratio', 0):.2f}",
                f"{m.get('sortino_ratio', 0):.2f}",
                f"{m.get('calmar_ratio', 0):.2f}",
                f"{m.get('ic', 0):.3f}",
                f"{m.get('rank_ic', 0):.3f}",
                f"{m.get('mutual_info', 0):.3f}",
                f"{m.get('win_rate', 0):.2%}",
                r.status,
            )
        else:
            table.add_row(
                r.job.indicator_name,
                r.job.strategy_name,
                *(["-"] * 7),
                f"[red]{r.status}[/red]",
            )
    console.print(table)

    if not completed:
        console.print("[red]No completed results — aborting.[/red]")
        return

    # ── 5. Ensemble ───────────────────────────────────────────────────────────
    console.rule("[bold cyan]Step 4: Strategy Ensembles[/bold cyan]")
    ensemble = AlphaEnsemble(completed)
    ensemble_results = []

    for method in ["equal_weight", "inverse_volatility", "ic_weighted", "sharpe_weighted"]:
        try:
            equity_curve = ensemble.combine(method=method)
            returns = equity_curve.pct_change().dropna()
            ann = 252
            ann_ret = returns.mean() * ann
            vol = returns.std() * (ann**0.5)
            sharpe = (ann_ret - 0.02) / vol if vol > 0 else 0
            max_dd = (equity_curve / equity_curve.cummax() - 1).min()

            dummy_job = AlphaJob(
                data=data,
                indicator_name="Ensemble",
                strategy_name=method,
                indicator_params={},
                strategy_params={},
            )
            ensemble_results.append(
                AlphaResult(
                    job=dummy_job,
                    status="completed",
                    equity_curve=equity_curve,
                    metrics={
                        "total_return": float((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1),
                        "sharpe_ratio": float(sharpe),
                        "max_drawdown": float(max_dd),
                        "ic": 0.0,
                        "rank_ic": 0.0,
                        "mutual_info": 0.0,
                    },
                )
            )
            console.print(f"  {method:<22} Sharpe={sharpe:.2f}  MaxDD={max_dd:.2%}")
        except Exception as e:
            console.print(f"[red]  {method} failed: {e}[/red]")

    # ── 6. Robustness checks ──────────────────────────────────────────────────
    console.rule("[bold cyan]Step 5: Robustness Analysis[/bold cyan]")
    robustness = RobustnessTester(runner)

    # Parameter sensitivity on RSI (first job in list)
    rsi_job = next(j for j in jobs if j.indicator_name == "RSI" and j.indicator_params.get("window") == 14)
    console.print("[cyan]Parameter sensitivity: RSI window × oversold threshold[/cyan]")
    sensitivity_df = robustness.parameter_sensitivity(
        rsi_job,
        indicator_param_grid={"window": [7, 10, 14, 21]},
        strategy_param_grid={"oversold": [20, 25, 30, 35]},
        n_jobs=2,
    )
    console.print(
        sensitivity_df.sort_values("sharpe_ratio", ascending=False)
        .head(5)[["window", "oversold", "sharpe_ratio", "ic", "total_return"]]
        .to_string(index=False)
    )

    # Sub-period stability on best strategy
    best = max(completed, key=lambda r: r.metrics.get("sharpe_ratio", -999))
    console.print(f"\n[cyan]Quarterly stability: {best.job.indicator_name} | {best.job.strategy_name}[/cyan]")
    period_df = robustness.sub_period_analysis(best, period="Q")
    console.print(period_df[["Return", "Sharpe", "MaxDrawdown"]].to_string())

    # ── 7. HTML report ────────────────────────────────────────────────────────
    console.rule("[bold cyan]Step 6: HTML Report[/bold cyan]")
    visualizer = AlphaVisualizer()
    all_results = completed + ensemble_results
    visualizer.generate_html_report(all_results, "alpha_research_report.html")


if __name__ == "__main__":
    main()
