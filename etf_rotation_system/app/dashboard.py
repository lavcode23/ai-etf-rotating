"""
Streamlit Dashboard for Indian ETF Sector Rotation System
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.backtest_runner import BacktestRunner
from reports.excel_report import generate_excel_reports

# Page config
st.set_page_config(
    page_title="Indian ETF Sector Rotation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .positive {
        color: #00c853;
    }
    .negative {
        color: #d50000;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_backtest(start_date, end_date, top_n, initial_capital):
    """Load and cache backtest results"""
    runner = BacktestRunner(
        start_date=start_date,
        end_date=end_date,
        top_n_etfs=top_n,
        initial_capital=initial_capital
    )
    runner.run_complete_backtest(force_refresh_data=False)
    return runner


def plot_equity_curve_plotly(runner):
    """Plot equity curve with Plotly"""
    equity_df = runner.portfolio.get_equity_curve()
    equity_df['Date'] = pd.to_datetime(equity_df['Date'])
    
    # Get benchmark
    benchmark_data = runner.data['NIFTYBEES']
    benchmark_aligned = benchmark_data.loc[
        equity_df['Date'].min():equity_df['Date'].max()
    ]
    benchmark_normalized = (
        (benchmark_aligned['Close'] / benchmark_aligned['Close'].iloc[0]) *
        runner.initial_capital
    )
    
    fig = go.Figure()
    
    # Strategy
    fig.add_trace(go.Scatter(
        x=equity_df['Date'],
        y=equity_df['Portfolio_Value'],
        name='Strategy',
        line=dict(color='#2E86AB', width=3)
    ))
    
    # Benchmark
    fig.add_trace(go.Scatter(
        x=benchmark_aligned.index,
        y=benchmark_normalized,
        name='NIFTYBEES',
        line=dict(color='#A23B72', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Portfolio Equity Curve',
        xaxis_title='Date',
        yaxis_title='Portfolio Value (₹)',
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    return fig


def plot_drawdown_plotly(runner):
    """Plot drawdown with Plotly"""
    equity_df = runner.portfolio.get_equity_curve()
    equity_df['Date'] = pd.to_datetime(equity_df['Date'])
    equity_df = equity_df.set_index('Date')
    
    cumulative = equity_df['Portfolio_Value']
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown,
        fill='tozeroy',
        name='Drawdown',
        line=dict(color='#E63946')
    ))
    
    fig.update_layout(
        title='Portfolio Drawdown',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    return fig


def plot_monthly_returns_plotly(runner):
    """Plot monthly returns as bars"""
    monthly_returns = runner.portfolio.calculate_monthly_returns()
    monthly_returns = monthly_returns.dropna()
    
    colors = ['#06D6A0' if x > 0 else '#EF476F' 
              for x in monthly_returns['Monthly_Return']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=monthly_returns['Date'],
        y=monthly_returns['Monthly_Return'],
        marker_color=colors,
        name='Monthly Return'
    ))
    
    fig.update_layout(
        title='Monthly Returns',
        xaxis_title='Date',
        yaxis_title='Return (%)',
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    return fig


def plot_sector_rotation_plotly(runner):
    """Plot sector rotation heatmap"""
    allocations = runner.allocations.copy()
    allocations['Date'] = pd.to_datetime(allocations['Selected_Date'])
    allocations['YearMonth'] = allocations['Date'].dt.strftime('%Y-%m')
    
    pivot_data = allocations.pivot_table(
        index='ETF',
        columns='YearMonth',
        values='Weight',
        fill_value=0
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values * 100,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='YlOrRd',
        colorbar=dict(title='Allocation (%)')
    ))
    
    fig.update_layout(
        title='Sector Rotation Over Time',
        xaxis_title='Month',
        yaxis_title='ETF',
        height=500,
        template='plotly_white'
    )
    
    return fig


def main():
    """Main dashboard"""
    
    # Header
    st.markdown('<p class="main-header">🚀 Indian ETF Sector Rotation System</p>', 
                unsafe_allow_html=True)
    st.markdown("**AI-Powered Monthly Sector Rotation Strategy**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Date range
        st.subheader("Backtest Period")
        start_date = st.date_input(
            "Start Date",
            value=datetime(2015, 1, 1)
        )
        end_date = st.date_input(
            "End Date",
            value=datetime.now()
        )
        
        # Strategy parameters
        st.subheader("Strategy Parameters")
        top_n = st.slider("Number of ETFs", 2, 5, 3)
        initial_capital = st.number_input(
            "Initial Capital (₹)",
            value=1000000,
            step=100000
        )
        
        # Run backtest button
        run_button = st.button("🚀 Run Backtest", type="primary")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This system uses:
        - **XGBoost** ML models
        - **Technical** indicators
        - **Monthly** rebalancing
        - **Risk** management
        """)
    
    # Load or run backtest
    if 'runner' not in st.session_state or run_button:
        with st.spinner("Running backtest... This may take a few minutes..."):
            try:
                runner = load_backtest(
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    top_n=top_n,
                    initial_capital=initial_capital
                )
                st.session_state.runner = runner
                st.success("✅ Backtest completed successfully!")
            except Exception as e:
                st.error(f"Error running backtest: {e}")
                return
    
    if 'runner' not in st.session_state:
        st.info("👈 Configure parameters and click 'Run Backtest' to begin")
        return
    
    runner = st.session_state.runner
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "📈 Performance",
        "🎯 Current Allocation",
        "📋 Trade History",
        "📥 Reports"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.header("Performance Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        perf = runner.performance
        
        with col1:
            total_return = perf.get('Total_Return_%', 0)
            color = "positive" if total_return > 0 else "negative"
            st.metric(
                "Total Return",
                f"{total_return:.2f}%",
                delta=f"vs {perf.get('Benchmark_Total_Return_%', 0):.2f}% (Benchmark)"
            )
        
        with col2:
            cagr = perf.get('CAGR_%', 0)
            st.metric(
                "CAGR",
                f"{cagr:.2f}%",
                delta=f"Alpha: {perf.get('Alpha_%', 0):.2f}%"
            )
        
        with col3:
            sharpe = perf.get('Sharpe_Ratio', 0)
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        
        with col4:
            max_dd = perf.get('Max_Drawdown_%', 0)
            st.metric("Max Drawdown", f"{max_dd:.2f}%")
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.plotly_chart(
                plot_equity_curve_plotly(runner),
                use_container_width=True
            )
        
        with col2:
            st.subheader("Risk Metrics")
            
            risk_metrics = {
                'Annual Volatility': f"{perf.get('Annual_Volatility_%', 0):.2f}%",
                'Sortino Ratio': f"{perf.get('Sortino_Ratio', 0):.2f}",
                'Calmar Ratio': f"{perf.get('Calmar_Ratio', 0):.2f}",
                'Win Rate': f"{perf.get('Win_Rate_%', 0):.2f}%",
                'Hit Rate': f"{perf.get('Hit_Rate_%', 0):.2f}%"
            }
            
            for metric, value in risk_metrics.items():
                st.markdown(f"**{metric}:** {value}")
        
        st.plotly_chart(
            plot_drawdown_plotly(runner),
            use_container_width=True
        )
    
    # Tab 2: Performance
    with tab2:
        st.header("Detailed Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                plot_monthly_returns_plotly(runner),
                use_container_width=True
            )
        
        with col2:
            st.plotly_chart(
                plot_sector_rotation_plotly(runner),
                use_container_width=True
            )
        
        # Performance metrics table
        st.subheader("All Metrics")
        
        metrics_df = pd.DataFrame([runner.performance]).T
        metrics_df.columns = ['Value']
        metrics_df.index.name = 'Metric'
        
        st.dataframe(metrics_df, use_container_width=True)
    
    # Tab 3: Current Allocation
    with tab3:
        st.header("Current Recommended Allocation")
        
        current = runner.get_current_allocation()
        
        if current is not None and not current.empty:
            # Pie chart
            fig = px.pie(
                current,
                values='Weight',
                names='ETF',
                title='Current Portfolio Allocation'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Table
            st.subheader("Allocation Details")
            
            display_df = current.copy()
            display_df['Weight'] = (display_df['Weight'] * 100).round(2)
            display_df.columns = ['ETF', 'Weight (%)', 'Final Score', 
                                 'ML Probability', 'Tech Score']
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Next month predictions
            st.subheader("Next Month Top Predictions")
            
            predictions = runner.get_next_month_predictions()
            if predictions is not None:
                st.dataframe(
                    predictions.head(10),
                    use_container_width=True
                )
        else:
            st.warning("No current allocation available")
    
    # Tab 4: Trade History
    with tab4:
        st.header("Trade History")
        
        trades_df = runner.portfolio.get_trades()
        
        if not trades_df.empty:
            # Summary stats
            col1, col2, col3 = st.columns(3)
            
            sell_trades = trades_df[trades_df['Action'] == 'SELL']
            
            with col1:
                st.metric("Total Trades", len(trades_df))
            
            with col2:
                if not sell_trades.empty:
                    avg_return = sell_trades['PnL_Pct'].mean()
                    st.metric("Avg Trade Return", f"{avg_return:.2f}%")
            
            with col3:
                winning_trades = len(sell_trades[sell_trades['PnL'] > 0])
                total_completed = len(sell_trades)
                win_rate = (winning_trades / total_completed * 100) if total_completed > 0 else 0
                st.metric("Trade Win Rate", f"{win_rate:.2f}%")
            
            # Filter options
            st.subheader("Filter Trades")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                action_filter = st.multiselect(
                    "Action",
                    options=trades_df['Action'].unique(),
                    default=trades_df['Action'].unique()
                )
            
            with col2:
                etf_filter = st.multiselect(
                    "ETF",
                    options=sorted(trades_df['ETF'].unique()),
                    default=sorted(trades_df['ETF'].unique())
                )
            
            # Filter data
            filtered_trades = trades_df[
                (trades_df['Action'].isin(action_filter)) &
                (trades_df['ETF'].isin(etf_filter))
            ]
            
            st.dataframe(
                filtered_trades.sort_values('Date', ascending=False),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("No trades executed")
    
    # Tab 5: Reports
    with tab5:
        st.header("Download Reports")
        
        st.markdown("""
        Generate and download comprehensive Excel reports with:
        - Performance summary
        - Trade log
        - Monthly returns
        - Allocation history
        - Current recommendations
        """)
        
        if st.button("📥 Generate Excel Reports"):
            with st.spinner("Generating reports..."):
                try:
                    reports = generate_excel_reports(runner, output_dir='reports')
                    
                    st.success("✅ Reports generated successfully!")
                    
                    # Provide download links
                    for report_type, filepath in reports.items():
                        if filepath and os.path.exists(filepath):
                            with open(filepath, 'rb') as f:
                                st.download_button(
                                    label=f"Download {report_type.replace('_', ' ').title()}",
                                    data=f,
                                    file_name=os.path.basename(filepath),
                                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                                )
                except Exception as e:
                    st.error(f"Error generating reports: {e}")


if __name__ == "__main__":
    main()
