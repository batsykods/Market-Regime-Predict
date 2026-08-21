# Stock Market Regime & Signal Detection

An end-to-end machine learning system that detects stock-market regimes and generates BUY, HOLD, and SELL signals using historical market data.

## Overview

The system combines:

- Feature engineering
- Hidden Markov Model (HMM) for market-regime detection
- XGBoost for signal classification
- Walk-forward validation
- Confidence-based position sizing
- Volatility-based risk management
- Backtesting
- Performance evaluation
- FastAPI
- Streamlit dashboard

## Architecture

```text
Market Data
     ↓
Feature Engineering
     ↓
Walk-Forward HMM
     ↓
Market Regime
     ↓
Walk-Forward XGBoost
     ↓
BUY / HOLD / SELL
     ↓
Risk Management
     ↓
Backtesting
     ↓
Performance Metrics
     ↓
FastAPI + Streamlit