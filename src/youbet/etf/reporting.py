"""Strategy comparison reports and tear sheets."""

from __future__ import annotations

from youbet.etf.backtester import BacktestResult


def comparison_table(results: list[BacktestResult]) -> str:
    """Format a comparison table across multiple strategy backtest results."""
    if not results:
        return "No results to compare."

    header = (
        f"{'Strategy':<25} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} "
        f"{'CVaR95':>8} {'IR':>8} {'Corr':>6} {'Turn':>6} {'ExSh':>8}"
    )
    sep = "-" * len(header)

    lines = [header, sep]
    for r in results:
        m = r.overall_metrics
        lines.append(
            f"{r.strategy_name:<25} "
            f"{m.sharpe_ratio:>8.3f} "
            f"{m.sortino_ratio:>8.3f} "
            f"{m.max_drawdown:>8.3f} "
            f"{m.cvar_95:>8.4f} "
            f"{m.information_ratio:>8.3f} "
            f"{m.correlation_to_benchmark:>6.3f} "
            f"{m.annual_turnover:>6.2f} "
            f"{r.excess_sharpe:>+8.3f}"
        )

    return "\n".join(lines)


def strategy_tearsheet(result: BacktestResult) -> str:
    """Generate a text tear sheet for a single strategy."""
    lines = [
        f"{'=' * 60}",
        f"STRATEGY TEAR SHEET: {result.strategy_name}",
        f"{'=' * 60}",
        "",
        f"Period: {result.overall_returns.index[0].date()} to "
        f"{result.overall_returns.index[-1].date()}",
        f"Folds:  {len(result.fold_results)}",
        "",
        "--- Performance ---",
        result.overall_metrics.summary(),
        "",
        f"Excess Sharpe vs Benchmark: {result.excess_sharpe:+.4f}",
        f"Total Cost Drag:            {result.total_cost_drag:.6f}",
        "",
        "--- Per-Fold Breakdown ---",
    ]

    for f in result.fold_results:
        lines.append(
            f"  {f.fold_name}: "
            f"test {f.test_start.date()}-{f.test_end.date()} | "
            f"Sharpe {f.metrics.sharpe_ratio:+.3f} | "
            f"MaxDD {f.metrics.max_drawdown:.3f} | "
            f"Turnover {f.total_turnover:.3f}"
        )

    return "\n".join(lines)
