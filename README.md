# Trader Performance vs Market Sentiment
### Primetrade.ai — Data Science Intern Assignment

---

## Overview

This project analyzes how **Bitcoin market sentiment (Fear/Greed Index)** relates to **trader behavior and performance** on the Hyperliquid decentralized exchange.

**Datasets used:**
- `historical_data.csv` — 211,224 trades across 32 accounts and 246 coins (Dec 2024 – Apr 2025)
- `fear_greed_index.csv` — Daily Fear/Greed classification and score (2018–2025)

---

## Project Structure

```
primetrade_project/
├── analysis.ipynb          # Full Jupyter notebook (Parts A, B, C + Bonus)
├── dashboard.py            # Streamlit interactive dashboard
├── historical_data.csv     # Trader data (place here)
├── fear_greed_index.csv    # Fear/Greed data (place here)
├── outputs/                # Auto-generated charts & CSVs
│   ├── chart1_performance_vs_sentiment.png
│   ├── chart2_behavior_vs_sentiment.png
│   ├── chart3_segment_performance.png
│   ├── chart4_pnl_distribution.png
│   ├── chart5_long_ratio_timeline.png
│   ├── chart6_top_traders.png
│   ├── chart7_feature_importance.png
│   ├── account_summary.csv
│   └── daily_merged.csv
└── README.md
```

---

## Setup

### Requirements
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn streamlit plotly
```

### Run the Notebook
```bash
jupyter notebook analysis.ipynb
```
> Run all cells top to bottom. Charts are saved to `outputs/`.

### Run the Dashboard
```bash
streamlit run dashboard.py
```
> Opens a browser at `http://localhost:8501`

---

## Methodology

### Data Preparation (Part A)
1. Loaded both datasets; confirmed **no missing values** in either
2. Parsed trader timestamps (`dd-mm-yyyy HH:MM` IST format) and aligned to daily granularity
3. Merged on `date` → **inner join** preserving only overlapping days
4. Engineered key metrics: daily PnL per account, win rate, long/short ratio, approximate leverage, trade frequency, cumulative PnL, and drawdown proxy

### Analysis (Part B)
- Compared performance metrics (PnL, win rate, drawdown) across all 5 sentiment states
- Ran Mann-Whitney U tests to assess statistical significance of Fear vs Greed differences
- Segmented traders into 3 dimensions:
  - **High vs Low Leverage** (split at median leverage)
  - **Frequent vs Infrequent** (split at median trades/day)
  - **Consistent Winners vs Inconsistent** (positive total PnL + win rate > 50%)
- Produced 6 charts backing 3 key insights

### Bonus: Predictive Model
- Random Forest classifier predicting **next-day profitability**
- Features: sentiment score, trade count, win rate, leverage, long ratio, trade size, daily PnL
- Evaluated with classification report; feature importance plotted

---

## Key Insights

### Insight 1 — Greed Days Drive Higher PnL and Win Rate
Traders earn more and win more frequently on Greed and Extreme Greed days. Mean daily PnL is significantly higher during Greed sentiment compared to Fear, with win rates also elevated. The difference is statistically significant (Mann-Whitney U, p < 0.05).

### Insight 2 — High-Leverage Traders Are Hit Hardest on Fear Days
High-leverage traders show the steepest decline in PnL on Fear days — often posting negative mean daily PnL — while low-leverage traders remain relatively stable. This suggests leverage amplifies downside risk during fearful markets more than it amplifies upside during Greed.

### Insight 3 — Consistent Winners Use Lower Leverage and Better Directional Alignment
Traders classified as "Consistent Winners" (positive total PnL + win rate > 50%) operate with lower average leverage and better-aligned long/short bias relative to sentiment. Inconsistent traders tend to over-leverage regardless of market conditions.

---

## Strategy Recommendations (Part C)

### Strategy 1 — Sentiment-Gated Leverage Control
> *"On Fear days, cap leverage at 5x. On Greed days, high-frequency traders may scale to 10x."*

High-leverage traders suffer the deepest drawdowns on Fear days. A sentiment-aware leverage ceiling prevents catastrophic losses while preserving upside on bullish days.

### Strategy 2 — Directional Bias Alignment with Sentiment
> *"During Fear, favour short positions or reduce long exposure. During Greed, maintain or increase long bias."*

Long/short ratios naturally rise on Greed days and fall on Fear days. Consistent Winners already do this instinctively. Codifying this rule helps Inconsistent traders mirror top-performer behaviour.

---

## Reproducibility

All random seeds are fixed (`random_state=42`). The notebook runs end-to-end with a single "Run All" command. Output files are deterministic given the same input CSVs.

---

## 🚀 Live Dashboard

👉 Click here to explore the interactive dashboard (https://primetradeai-assignment-krgubf24e5mwlrberyu4pf.streamlit.app)

```

