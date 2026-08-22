# Stock Market Regime & Signal Detection

End-to-end NIFTY 50 machine-learning research and paper-trading pipeline.

## Research pipeline

```text
Yahoo Finance
    ↓
data_loader.py → features.py → walk_forward_hmm.py
    ↓
walk_forward_regimes.csv
    ↓
walk_forward.py
    ↓
walk_forward_results.csv
    ↓
risk_manager.py → performance.py
```

The production research model is a 5-trading-day XGBoost return regressor combined with a three-state Gaussian HMM and calibrated volatility-aware position sizing.

Frozen strategy configuration:

- Prediction threshold: 70th percentile
- Risk threshold: 65th percentile
- Holding period: 5 trading days
- Regime multipliers: Regime 0 = 1.00, Regime 1 = 0.25, Regime 2 = 0.70
- Transaction-cost assumption: 0.1% in backtests

## Validation

The repository contains controlled A/B testing, frozen out-of-sample validation, and rolling robustness tests. Historical performance is not evidence of future profitability.

## Live daily signal

The live engine downloads the latest NIFTY 50 daily market data, rebuilds the feature vector, loads the persisted HMM regime mapping, retrains the return model on available historical data, and applies the frozen risk rules.

```powershell
python -m src.live_predict
```

The output includes:

- latest price
- market regime
- predicted 5-day return
- prediction percentile
- calibrated signal/risk cutoffs
- BUY/HOLD/SELL signal
- position size
- target price
- ATR-based stop
- expected profit for ₹100,000 capital
- expected holding window

The current implementation is daily-bar based. It is not a tick-level or guaranteed real-time execution system.

## Paper trading

```powershell
python -m src.paper_trader
```

Each run records the current model snapshot to `data/processed/paper_trades.csv`. No broker order is sent.

## Dashboard

```powershell
streamlit run src/dashboard.py
```

The dashboard refreshes cached live inference every five minutes and displays the current signal, trade plan, historical equity curve, and paper-trade ledger.

## Full research pipeline

```powershell
python -m src.data_loader
python -m src.features
python -m src.walk_forward_hmm
python -m src.walk_forward
python -m src.risk_manager
python -m src.performance
python -m src.out_of_sample
python -m src.robustness_test
python -m src.rolling_robustness
```

## Safety boundary

This project is a research and paper-trading system. It does not place live broker orders. Predictions are estimates and can be wrong; they are not guaranteed returns or financial advice.
