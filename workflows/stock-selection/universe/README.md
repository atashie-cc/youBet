# Universe Files

## sp500_membership.csv (seed)

**This is a SEED file for testing — not a full historical membership list.**

The seed contains ~35 well-known current S&P 500 constituents with approximate
inclusion dates, sufficient to smoke-test the universe / backtester / PIT
plumbing. Before running any confirmatory phase (Phase 1+), replace this
seed with a full historical membership series so survivorship-bias
correctness is real, not nominal.

Sources that can produce a proper file:
- Wikipedia "List of S&P 500 companies" + its "Selected changes" history
  table (scrape via `pandas.read_html`)
- Public GitHub archives, e.g., `fja05680/sp500`
- WRDS/CRSP (paid)

Required schema:
```
ticker, name, gics_sector, gics_subindustry, start_date, end_date, cik, notes
```

One row per *membership interval*. A ticker added, removed, and re-added
should have two rows. `end_date` is empty for still-current members.
`cik` is the zero-padded 10-digit SEC EDGAR CIK.

## sp600_membership.csv

Placeholder for Phase 5 (small-cap OOS replication). Same schema.

## delisting_returns.csv

One row per delisted ticker with the terminal one-day total return.
- Acquisition: typically positive (premium over prior close).
- Bankruptcy / liquidation: often -1.0.
- Ticker-change only: ~0.

Schema:
```
ticker, delist_date, delist_return, reason
```
