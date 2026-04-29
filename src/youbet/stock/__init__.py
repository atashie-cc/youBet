"""Individual-stock strategy evaluation engine.

Extends the ETF backtesting infrastructure to cross-sectional stock picking:
point-in-time fundamentals (SEC EDGAR), historical index membership with
delisting handling, mcap-bucketed costs, and a backtester that supports
500+ tickers with changing universes per rebalance.
"""
