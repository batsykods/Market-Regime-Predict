# Stock Market Regime & Signal Detection

An end-to-end NIFTY 50 machine-learning pipeline for market-regime detection and BUY/HOLD/SELL signal generation.

## Canonical pipeline

```text
Yahoo Finance
    ↓
data_loader.py
    ↓
features.py
    ↓
walk_forward_hmm.py
    ↓
walk_forward_regimes.csv
    ↓
walk_forward.py
    ↓
walk_forward_results.csv
    ↓
risk_manager.py
    ↓
risk_adjusted_backtest.csv
    ↓
performance.py
    ↓
FastAPI / Streamlit / live_predict.py
```

## Models

- Gaussian HMM: three market regimes.
- XGBoost: BUY/HOLD/SELL classification.
- Walk-forward validation: predictions use historical data only.
- Risk management: confidence- and volatility-based position sizing.

## Important live-inference rule

HMM state numbers have no intrinsic meaning. The training pipeline persists the state-to-regime mapping in `data/models/regime_mapping.pkl`. Live inference loads that mapping instead of assuming `state 0 = regime 0`.

## Run the pipeline

```powershell
python -m src.data_loader
python -m src.features
python -m src.walk_forward_hmm
python -m src.walk_forward
python -m src.risk_manager
python -m src.performance
```

## Live prediction

After the models have been trained:

```powershell
python -m src.live_data
python -m src.live_predict
```

## Dashboard

```powershell
streamlit run src/dashboard.py
```

## API

```powershell
uvicorn src.api:app --reload
```

## Validation

The production evaluation path is the walk-forward backtest. Do not use in-sample model accuracy as evidence that the trading strategy is profitable.

`return_model.py` is an experimental return-regression model and is not part of the production signal pipeline.
