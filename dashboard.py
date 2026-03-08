"""
Primetrade.ai — Trader Performance vs Market Sentiment
Streamlit Dashboard
Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trader Performance vs Market Sentiment",
    page_icon="📈",
    layout="wide",
)

# ── Color palette ────────────────────────────────────────────────────────────
SENTIMENT_COLORS = {
    'Extreme Fear': '#d62728',
    'Fear':         '#ff7f0e',
    'Neutral':      '#9467bd',
    'Greed':        '#2ca02c',
    'Extreme Greed':'#1f77b4',
}
BINARY_MAP = {
    'Extreme Fear': 'Fear', 'Fear': 'Fear',
    'Neutral': 'Neutral',
    'Greed': 'Greed', 'Extreme Greed': 'Greed',
}
CATS = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']


# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    trader = pd.read_csv('historical_data.csv')
    fg     = pd.read_csv('fear_greed_index.csv')

    # Parse timestamps
    trader['datetime'] = pd.to_datetime(trader['Timestamp IST'], format='%d-%m-%Y %H:%M', dayfirst=True)
    trader['date']     = trader['datetime'].dt.date.astype(str)

    fg['date'] = fg['date'].astype(str)
    fg['binary_sentiment'] = fg['classification'].map(BINARY_MAP)

    # Merge
    df = trader.merge(fg[['date', 'classification', 'binary_sentiment', 'value']], on='date', how='inner')

    # Features
    df['is_win']          = df['Closed PnL'] > 0
    df['is_long']         = df['Side'] == 'BUY'
    df['leverage_approx'] = np.where(
        df['Start Position'].abs() > 0,
        (df['Size USD'] / df['Start Position'].abs()).clip(0, 200),
        np.nan
    )

    # Daily aggregation
    daily = df.groupby(['Account', 'date', 'classification', 'binary_sentiment', 'value']).agg(
        daily_pnl       = ('Closed PnL', 'sum'),
        n_trades        = ('Trade ID', 'count'),
        win_count       = ('is_win', 'sum'),
        avg_trade_size  = ('Size USD', 'mean'),
        long_count      = ('is_long', 'sum'),
        avg_leverage    = ('leverage_approx', 'mean'),
    ).reset_index()

    daily['win_rate']   = daily['win_count'] / daily['n_trades']
    daily['long_ratio'] = daily['long_count'] / daily['n_trades']

    # Cumulative PnL & drawdown
    daily = daily.sort_values(['Account', 'date'])
    daily['cum_pnl']     = daily.groupby('Account')['daily_pnl'].cumsum()
    daily['rolling_max'] = daily.groupby('Account')['cum_pnl'].cummax()
    daily['drawdown']    = daily['cum_pnl'] - daily['rolling_max']

    # Account-level segments
    acct = daily.groupby('Account').agg(
        total_pnl       = ('daily_pnl', 'sum'),
        mean_win_rate   = ('win_rate', 'mean'),
        mean_leverage   = ('avg_leverage', 'mean'),
        total_trades    = ('n_trades', 'sum'),
        trading_days    = ('date', 'nunique'),
    ).reset_index()

    acct['trades_per_day']      = acct['total_trades'] / acct['trading_days']
    lev_med                      = acct['mean_leverage'].median()
    freq_med                     = acct['trades_per_day'].median()
    acct['leverage_segment']    = np.where(acct['mean_leverage'] > lev_med, 'High Leverage', 'Low Leverage')
    acct['frequency_segment']   = np.where(acct['trades_per_day'] > freq_med, 'Frequent', 'Infrequent')
    acct['winner_segment']      = np.where(
        (acct['total_pnl'] > 0) & (acct['mean_win_rate'] > 0.5), 'Consistent Winner', 'Inconsistent'
    )

    daily = daily.merge(acct[['Account','leverage_segment','frequency_segment','winner_segment']], on='Account', how='left')
    return df, daily, acct


df, daily, acct = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🔍 Filters")

selected_sentiment = st.sidebar.multiselect(
    "Sentiment Filter",
    options=CATS,
    default=CATS
)

selected_accounts = st.sidebar.multiselect(
    "Filter by Account",
    options=sorted(daily['Account'].unique()),
    default=[]
)

date_min = pd.to_datetime(daily['date']).min().date()
date_max = pd.to_datetime(daily['date']).max().date()
date_range = st.sidebar.date_input("Date Range", [date_min, date_max])

# Apply filters
mask = daily['classification'].isin(selected_sentiment)
if selected_accounts:
    mask &= daily['Account'].isin(selected_accounts)
if len(date_range) == 2:
    mask &= (pd.to_datetime(daily['date']).dt.date >= date_range[0]) & \
            (pd.to_datetime(daily['date']).dt.date <= date_range[1])

filtered = daily[mask]

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("📈 Trader Performance vs Market Sentiment")
st.caption("Hyperliquid Trader Data × Bitcoin Fear/Greed Index | Primetrade.ai Internship Project")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Trades",     f"{len(df[df['date'].isin(filtered['date'].unique())]):,}")
c2.metric("Unique Traders",   filtered['Account'].nunique())
c3.metric("Days Covered",     filtered['date'].nunique())
c4.metric("Mean Daily PnL",   f"${filtered['daily_pnl'].mean():,.0f}")
c5.metric("Avg Win Rate",     f"{filtered['win_rate'].mean()*100:.1f}%")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# ROW 1 — Performance by Sentiment
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📊 Performance by Sentiment")

perf = filtered.groupby('classification').agg(
    mean_daily_pnl = ('daily_pnl', 'mean'),
    mean_win_rate  = ('win_rate', 'mean'),
    mean_drawdown  = ('drawdown', 'mean'),
).reindex([c for c in CATS if c in filtered['classification'].unique()]).reset_index()

col1, col2, col3 = st.columns(3)

with col1:
    fig = px.bar(perf, x='classification', y='mean_daily_pnl',
                 color='classification',
                 color_discrete_map=SENTIMENT_COLORS,
                 title='Mean Daily PnL (USD)', labels={'classification': '', 'mean_daily_pnl': 'USD'},
                 category_orders={'classification': CATS})
    fig.add_hline(y=0, line_dash='dash', line_color='black')
    fig.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    perf['win_pct'] = perf['mean_win_rate'] * 100
    fig = px.bar(perf, x='classification', y='win_pct',
                 color='classification',
                 color_discrete_map=SENTIMENT_COLORS,
                 title='Mean Win Rate (%)', labels={'classification': '', 'win_pct': '%'},
                 category_orders={'classification': CATS})
    fig.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig, use_container_width=True)

with col3:
    fig = px.bar(perf, x='classification', y='mean_drawdown',
                 color='classification',
                 color_discrete_map=SENTIMENT_COLORS,
                 title='Mean Drawdown (USD)', labels={'classification': '', 'mean_drawdown': 'USD'},
                 category_orders={'classification': CATS})
    fig.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 2 — Behavioral Metrics
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🔄 Trader Behavior by Sentiment")

behavior = filtered.groupby('classification').agg(
    avg_trades     = ('n_trades', 'mean'),
    avg_leverage   = ('avg_leverage', 'mean'),
    avg_long_ratio = ('long_ratio', 'mean'),
    avg_size       = ('avg_trade_size', 'mean'),
).reindex([c for c in CATS if c in filtered['classification'].unique()]).reset_index()

col1, col2, col3, col4 = st.columns(4)
metrics_list = [
    (col1, 'avg_trades',     'Avg Trades/Day', 'Count'),
    (col2, 'avg_leverage',   'Avg Leverage',   'x'),
    (col3, 'avg_long_ratio', 'Long Bias Ratio','ratio'),
    (col4, 'avg_size',       'Avg Trade Size', 'USD'),
]

for col, field, title, unit in metrics_list:
    with col:
        fig = px.bar(behavior, x='classification', y=field,
                     color='classification',
                     color_discrete_map=SENTIMENT_COLORS,
                     title=title, labels={'classification': '', field: unit},
                     category_orders={'classification': CATS})
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 3 — PnL Timeline
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📅 Daily PnL Timeline")

timeline = filtered.groupby(['date','classification'])['daily_pnl'].mean().reset_index()
timeline['date'] = pd.to_datetime(timeline['date'])

fig = px.scatter(timeline, x='date', y='daily_pnl',
                 color='classification',
                 color_discrete_map=SENTIMENT_COLORS,
                 labels={'date': 'Date', 'daily_pnl': 'Mean Daily PnL (USD)', 'classification': 'Sentiment'},
                 title='Mean Daily PnL over Time (coloured by Sentiment)')
fig.add_hline(y=0, line_dash='dash', line_color='grey')
fig.update_layout(height=380)
st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 4 — Segment Analysis
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("👥 Trader Segment Analysis")

seg_option = st.radio("Select Segment Type", ['leverage_segment','frequency_segment','winner_segment'],
                      format_func=lambda x: x.replace('_segment','').replace('_',' ').title(),
                      horizontal=True)

seg_data = filtered.groupby([seg_option, 'binary_sentiment'])['daily_pnl'].mean().reset_index()
seg_data.columns = ['Segment', 'Sentiment', 'Mean Daily PnL']

fig = px.bar(seg_data, x='Segment', y='Mean Daily PnL', color='Sentiment',
             barmode='group',
             color_discrete_map={'Fear':'#ff7f0e','Neutral':'#9467bd','Greed':'#2ca02c'},
             title=f'Mean Daily PnL by {seg_option.replace("_segment","").title()} Segment and Sentiment')
fig.add_hline(y=0, line_dash='dash', line_color='black')
fig.update_layout(height=380)
st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 5 — Account Leaderboard
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🏆 Trader Leaderboard")

show_acct = acct.copy()
show_acct['Account_short'] = show_acct['Account'].str[:14] + '...'
show_acct = show_acct.sort_values('total_pnl', ascending=False)

col1, col2 = st.columns([2, 1])

with col1:
    fig = px.bar(show_acct.head(15), x='Account_short', y='total_pnl',
                 color='winner_segment',
                 color_discrete_map={'Consistent Winner':'#2ca02c','Inconsistent':'#ff7f0e'},
                 title='Top 15 Traders by Total PnL',
                 labels={'Account_short': 'Account', 'total_pnl': 'Total PnL (USD)', 'winner_segment': 'Segment'})
    fig.update_layout(height=380, xaxis_tickangle=30)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    display_cols = ['Account_short','total_pnl','mean_win_rate','mean_leverage','winner_segment']
    st.dataframe(
        show_acct[display_cols].rename(columns={
            'Account_short': 'Account',
            'total_pnl': 'Total PnL',
            'mean_win_rate': 'Win Rate',
            'mean_leverage': 'Avg Lev',
            'winner_segment': 'Segment'
        }).head(15).style.format({
            'Total PnL': '${:,.0f}',
            'Win Rate': '{:.1%}',
            'Avg Lev': '{:.1f}x'
        }),
        height=380
    )

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("💡 Strategy Recommendations")

col1, col2 = st.columns(2)

with col1:
    st.info("""
**Strategy 1 — Sentiment-Gated Leverage Control**

*Rule:* On **Fear** days, cap leverage at **5x** for all traders.
On **Greed** days, high-frequency traders may scale to **10x**.

*Rationale:* High-leverage traders suffer the deepest losses on Fear days.
Capping leverage during fearful markets dramatically reduces max drawdown
while preserving upside during Greed.
    """)

with col2:
    st.success("""
**Strategy 2 — Directional Bias Alignment**

*Rule:* During **Fear**, favour **short** positions or cut long exposure.
During **Greed**, maintain or increase **long bias**.

*Rationale:* Long/short ratio naturally rises on Greed days and falls on
Fear days. Consistent Winners already align their directional bias to
sentiment — Inconsistent traders do not. Following this rule mimics
top-performer behaviour.
    """)

st.caption("Primetrade.ai Intern Assignment | Analysis by Ajay Vispute")
