# Indian ETF Sector Rotation System 🚀

A professional-grade quantitative trading system that uses Machine Learning and Technical Analysis to rotate capital monthly across top-performing Indian sector ETFs.

## 📊 Overview

This system implements a sophisticated **monthly sector rotation strategy** for the Indian market, combining:

- **XGBoost ML Models** for predicting next-month ETF outperformance
- **Technical Indicators** (RSI, MACD, ADX, Bollinger Bands) for momentum scoring
- **Walk-Forward Training** with rolling 36-month windows
- **Risk Management** rules (ADX filters, RSI limits, stop-loss)
- **Automated Backtesting** from 2015 to present
- **Professional Reporting** with Excel exports and interactive charts
- **Streamlit Web UI** for visualization and analysis

---

## 🎯 Strategy Logic

### Universe
- **Benchmark**: NIFTYBEES
- **Sector ETFs**: BANKBEES, ITBEES, PHARMABEES, PSUBANKBEES, FMCGBEES, METALBEES, CONSUMPTIONBEES, MIDCAPBEES, SMALLCAPBEES

### Process (Monthly)
1. **Feature Engineering**: Calculate 19+ features from daily OHLCV data
2. **ML Prediction**: XGBoost predicts probability of outperformance
3. **Technical Scoring**: Combine RSI, MACD, ADX, BB into 0-1 score
4. **Final Score**: `0.6 × ML_Probability + 0.4 × TechScore`
5. **Selection**: Pick top 2-3 ETFs with highest Final Score
6. **Allocation**: Equal weight allocation (90% invested, 10% cash)
7. **Hold**: Hold for one month
8. **Rebalance**: Repeat next month

### Risk Management
- ✅ Maximum 2% risk per ETF
- ✅ Stop-loss at -10%
- ✅ ADX filter: Skip if ADX < 15 (weak trend)
- ✅ RSI filter: Skip if RSI > 80 (overbought)
- ✅ 10% cash reserve maintained

---

## 📁 Project Structure

```
etf_rotation_system/
│
├── data/
│   └── fetch_data.py              # Data fetching from yfinance
│
├── features/
│   ├── indicators.py              # Technical indicator calculations
│   └── monthly_features.py        # Monthly feature aggregation
│
├── models/
│   ├── train_model.py             # XGBoost training with walk-forward
│   ├── predict.py                 # Generate predictions
│   └── saved_models/              # Trained model storage
│
├── strategies/
│   ├── technical_score.py         # Technical momentum scoring
│   └── sector_rotation.py         # Allocation logic
│
├── backtest/
│   ├── portfolio.py               # Portfolio tracking
│   ├── performance.py             # Performance metrics
│   └── backtest_runner.py         # Complete backtest orchestration
│
├── reports/
│   ├── excel_report.py            # Excel report generation
│   ├── charts.py                  # Chart generation
│   └── [output files]             # Generated reports
│
├── app/
│   └── dashboard.py               # Streamlit web interface
│
├── main.py                        # Main execution script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip

### Steps

1. **Clone or download this project**

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

That's it! The system will automatically:
- Download historical data
- Train ML models
- Run backtest
- Generate reports

---

## 💻 Usage

### Method 1: Run Complete Backtest (Command Line)

Execute the main script to run the entire pipeline:

```bash
python main.py
```

This will:
1. ✅ Fetch 10 years of daily data for all ETFs
2. ✅ Calculate technical indicators
3. ✅ Generate monthly features
4. ✅ Train XGBoost models with walk-forward validation
5. ✅ Generate ML predictions
6. ✅ Calculate technical scores
7. ✅ Run backtest simulation
8. ✅ Generate Excel reports (saved to `reports/`)
9. ✅ Generate charts (saved to `reports/charts/`)
10. ✅ Display current allocation recommendation

**Expected Runtime**: 3-5 minutes (first run), 30-60 seconds (cached)

### Method 2: Launch Web Dashboard

Start the interactive Streamlit dashboard:

```bash
streamlit run app/dashboard.py
```

Then open your browser to `http://localhost:8501`

**Dashboard Features**:
- 📊 Interactive performance charts
- 📈 Equity curve vs benchmark
- 🎯 Current allocation visualization
- 📋 Complete trade history
- 📥 Download Excel reports
- ⚙️ Configurable backtest parameters

---

## 📊 Output Files

### Excel Reports (`reports/`)

1. **Backtest_Report_[timestamp].xlsx**
   - Summary sheet with all metrics
   - Performance metrics
   - Complete trade log
   - Monthly returns
   - Allocation history
   - Current recommendations
   - Feature importance
   - Equity curve data

2. **Trade_Log.xlsx**
   - Simplified trade log
   - Entry/exit prices
   - P&L per trade
   - Holding periods

### Charts (`reports/charts/`)

- `equity_curve.png` - Portfolio vs benchmark
- `drawdown.png` - Drawdown over time
- `monthly_returns_heatmap.png` - Monthly returns by year
- `sector_rotation.png` - Allocation changes over time
- `feature_importance.png` - Top ML features
- `monthly_performance.png` - Monthly bar chart
- `rolling_sharpe.png` - 6-month rolling Sharpe

---

## 📈 Key Features

### Machine Learning
- **Model**: XGBoost Classifier
- **Target**: Binary (above/below median next-month return)
- **Training**: Walk-forward with 36-month rolling window
- **Retraining**: Every 1 month
- **Features**: 19 technical and momentum features

### Technical Analysis
- **RSI(14)**: Mean reversion + trend
- **MACD Histogram**: Momentum direction
- **ADX(14)**: Trend strength
- **Bollinger %B**: Position in range
- **ATR%**: Volatility measure
- **Volume indicators**: Spike detection

### Performance Metrics
- CAGR, Sharpe, Sortino, Calmar ratios
- Maximum drawdown
- Win rate (monthly & trades)
- Alpha vs benchmark
- Hit rate (months beating NIFTYBEES)
- Trade statistics

---

## 🎓 Example Output

```
========================================
PERFORMANCE SUMMARY
========================================

📈 Returns:
   Total Return:          127.45%
   CAGR:                   14.23%
   Benchmark CAGR:         10.87%
   Alpha:                   3.36%

📉 Risk:
   Annual Volatility:      18.52%
   Max Drawdown:          -22.34%
   Sharpe Ratio:            0.72
   Sortino Ratio:           1.05

🎯 Win Rates:
   Monthly Win Rate:       64.20%
   vs Benchmark:           58.50%
   Trade Win Rate:         61.30%

CURRENT RECOMMENDED ALLOCATION
========================================
   ITBEES              : 30.00%  (Score: 0.847)
   BANKBEES            : 30.00%  (Score: 0.782)
   MIDCAPBEES          : 30.00%  (Score: 0.756)
   CASH                : 10.00%
```

---

## 🔧 Configuration

### Backtest Parameters (main.py)

```python
START_DATE = '2015-01-01'      # Backtest start
END_DATE = None                # Today
TOP_N_ETFS = 3                 # Number of ETFs to hold
INITIAL_CAPITAL = 1000000      # ₹10 Lakh
```

### Strategy Parameters (strategies/sector_rotation.py)

```python
top_n = 3                      # Top N ETFs
max_exposure = 0.90            # 90% max invested
cash_reserve = 0.10            # 10% cash
```

### Model Parameters (models/train_model.py)

```python
train_window_months = 36       # Training window
retrain_frequency = 1          # Retrain every N months
n_estimators = 100            # XGBoost trees
max_depth = 4                 # Tree depth
```

---

## 🛠️ Customization

### Add New ETFs

Edit `data/fetch_data.py`:

```python
self.tickers = {
    'NIFTYBEES': 'NIFTYBEES.NS',
    'BANKBEES': 'BANKBEES.NS',
    # Add new ETF here:
    'NEWEEETF': 'NEWETF.NS',
    ...
}
```

### Add New Features

Edit `features/monthly_features.py` → `_create_monthly_features_for_etf()`

### Modify ML Model

Edit `models/train_model.py` → `walk_forward_train()`

### Change Technical Score Logic

Edit `strategies/technical_score.py` → `calculate_technical_score()`

---

## 📚 Dependencies

Key libraries:
- `pandas`, `numpy` - Data manipulation
- `yfinance` - Market data
- `ta` - Technical indicators
- `xgboost` - Machine learning
- `scikit-learn` - ML utilities
- `matplotlib`, `seaborn`, `plotly` - Visualization
- `streamlit` - Web dashboard
- `openpyxl` - Excel export

See `requirements.txt` for complete list.

---

## ⚠️ Disclaimer

**This is for educational and research purposes only.**

- Past performance does not guarantee future results
- This system does not constitute investment advice
- Trading involves risk of loss
- Always do your own research
- Test thoroughly before deploying with real capital
- The author is not responsible for any financial losses

---

## 📝 Notes

- **Data Source**: Yahoo Finance (yfinance)
- **Data Frequency**: Daily OHLCV
- **Rebalancing**: Monthly (on month-end)
- **Transaction Costs**: 0.1% per trade
- **Currency**: Indian Rupees (₹)

---

## 🤝 Support

For questions or issues:
1. Check the code comments
2. Review the Streamlit dashboard
3. Examine generated Excel reports
4. Verify data quality in `data/cache/`

---

## 📄 License

This project is provided as-is for educational purposes.

---

## 🎯 Quick Start Checklist

- [ ] Install Python 3.8+
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Run backtest: `python main.py`
- [ ] View reports in `reports/` directory
- [ ] Launch dashboard: `streamlit run app/dashboard.py`
- [ ] Analyze results and performance
- [ ] Customize for your needs

---

**Built with ❤️ for the Indian markets**

*Professional-grade quantitative trading system*
*Machine Learning + Technical Analysis + Risk Management*
