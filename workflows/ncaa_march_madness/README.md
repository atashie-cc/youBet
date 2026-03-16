# NCAA March Madness Bracket Prediction

Predict NCAA Men's Basketball Tournament outcomes using XGBoost on team stat differentials with isotonic calibration.

## Quick Start

```bash
python scripts/collect_data.py       # Download Kaggle data
python scripts/build_features.py     # Build 18 stat differentials
python scripts/train.py              # Train XGBoost + calibrate
python scripts/predict.py            # Predict tournament matchups
python scripts/generate_bracket.py   # Monte Carlo bracket simulation
python scripts/backtest.py           # Validate on 2003-2025 tournaments
```

## Target Performance
- Log loss: < 0.55 (tournament games)
- Accuracy: > 72% (tournament games)
- Calibration: within 5% of diagonal on calibration curve
