"""Cross-sectional stock-selection strategies.

All strategies subclass `CrossSectionalStrategy` and are consumed by
`youbet.stock.backtester.StockBacktester`.
"""

from youbet.stock.strategies.base import (
    CrossSectionalStrategy,
    EqualWeightBenchmark,
    BuyAndHoldETF,
    equal_weight,
    inverse_vol_weight,
    top_decile_select,
)
