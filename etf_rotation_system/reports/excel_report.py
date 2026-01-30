"""
Excel report generation
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os


class ExcelReporter:
    """Generate comprehensive Excel reports"""
    
    def __init__(self, backtest_runner, output_dir='reports'):
        """
        Args:
            backtest_runner: BacktestRunner object with results
            output_dir: Output directory for reports
        """
        self.runner = backtest_runner
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_full_report(self, filename=None):
        """
        Generate complete Excel report with multiple sheets
        
        Args:
            filename: Output filename (auto-generated if None)
            
        Returns:
            str: Path to generated file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'Backtest_Report_{timestamp}.xlsx'
        
        filepath = os.path.join(self.output_dir, filename)
        
        print(f"\nGenerating Excel report: {filename}")
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Sheet 1: Summary
            self._write_summary_sheet(writer)
            
            # Sheet 2: Performance Metrics
            self._write_performance_sheet(writer)
            
            # Sheet 3: Trade Log
            self._write_trade_log_sheet(writer)
            
            # Sheet 4: Monthly Returns
            self._write_monthly_returns_sheet(writer)
            
            # Sheet 5: Allocations
            self._write_allocations_sheet(writer)
            
            # Sheet 6: Current Recommendation
            self._write_current_recommendation_sheet(writer)
            
            # Sheet 7: Feature Importance
            self._write_feature_importance_sheet(writer)
            
            # Sheet 8: Equity Curve
            self._write_equity_curve_sheet(writer)
        
        print(f"   Report saved: {filepath}")
        return filepath
    
    def _write_summary_sheet(self, writer):
        """Write summary sheet"""
        summary_data = {
            'Metric': [],
            'Value': []
        }
        
        # Backtest configuration
        summary_data['Metric'].append('=== CONFIGURATION ===')
        summary_data['Value'].append('')
        
        summary_data['Metric'].append('Start Date')
        summary_data['Value'].append(self.runner.start_date)
        
        summary_data['Metric'].append('End Date')
        summary_data['Value'].append(str(self.runner.data['NIFTYBEES'].index[-1].date()))
        
        summary_data['Metric'].append('Initial Capital')
        summary_data['Value'].append(f"₹{self.runner.initial_capital:,.0f}")
        
        summary_data['Metric'].append('Top N ETFs')
        summary_data['Value'].append(self.runner.top_n_etfs)
        
        summary_data['Metric'].append('')
        summary_data['Value'].append('')
        
        # Performance metrics
        if self.runner.performance:
            summary_data['Metric'].append('=== PERFORMANCE ===')
            summary_data['Value'].append('')
            
            for metric, value in self.runner.performance.items():
                summary_data['Metric'].append(metric.replace('_', ' '))
                if isinstance(value, float):
                    if '%' in metric:
                        summary_data['Value'].append(f"{value:.2f}%")
                    else:
                        summary_data['Value'].append(f"{value:.4f}")
                else:
                    summary_data['Value'].append(value)
        
        df = pd.DataFrame(summary_data)
        df.to_excel(writer, sheet_name='Summary', index=False)
    
    def _write_performance_sheet(self, writer):
        """Write detailed performance metrics"""
        if self.runner.performance is None:
            return
        
        perf_df = pd.DataFrame([self.runner.performance])
        perf_df = perf_df.T
        perf_df.columns = ['Value']
        perf_df.index.name = 'Metric'
        
        perf_df.to_excel(writer, sheet_name='Performance Metrics')
    
    def _write_trade_log_sheet(self, writer):
        """Write trade log"""
        trades_df = self.runner.portfolio.get_trades()
        
        if trades_df.empty:
            return
        
        # Format trades
        trades_display = trades_df.copy()
        
        # Round numeric columns
        numeric_cols = trades_display.select_dtypes(include=[np.number]).columns
        trades_display[numeric_cols] = trades_display[numeric_cols].round(2)
        
        trades_display.to_excel(writer, sheet_name='Trade Log', index=False)
    
    def _write_monthly_returns_sheet(self, writer):
        """Write monthly returns"""
        monthly_returns = self.runner.portfolio.calculate_monthly_returns()
        
        if monthly_returns.empty:
            return
        
        monthly_returns.to_excel(writer, sheet_name='Monthly Returns', index=False)
        
        # Also create a returns heatmap data
        monthly_returns['Year'] = pd.to_datetime(monthly_returns['Date']).dt.year
        monthly_returns['Month'] = pd.to_datetime(monthly_returns['Date']).dt.month
        
        heatmap_data = monthly_returns.pivot(
            index='Year',
            columns='Month',
            values='Monthly_Return'
        )
        
        # Add year totals
        heatmap_data['YTD'] = heatmap_data.sum(axis=1)
        
        heatmap_data.to_excel(writer, sheet_name='Returns Heatmap')
    
    def _write_allocations_sheet(self, writer):
        """Write allocation history"""
        if self.runner.allocations is None or self.runner.allocations.empty:
            return
        
        allocations_display = self.runner.allocations.copy()
        
        # Select relevant columns
        cols = ['ETF', 'Weight', 'FinalScore', 'ML_Probability', 'TechScore',
                'RSI_14', 'MACD_Hist', 'ADX_14', 'Selected_Date']
        cols = [c for c in cols if c in allocations_display.columns]
        
        allocations_display = allocations_display[cols]
        
        # Round numeric columns
        numeric_cols = allocations_display.select_dtypes(include=[np.number]).columns
        allocations_display[numeric_cols] = allocations_display[numeric_cols].round(4)
        
        allocations_display.to_excel(writer, sheet_name='Allocations', index=False)
    
    def _write_current_recommendation_sheet(self, writer):
        """Write current allocation recommendation"""
        current = self.runner.get_current_allocation()
        
        if current is None or current.empty:
            return
        
        # Add metadata
        metadata = pd.DataFrame({
            'Metric': ['Report Date', 'Allocation Date', 'Strategy'],
            'Value': [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                str(self.runner.allocations['Selected_Date'].max()),
                f'Top {self.runner.top_n_etfs} ETF Rotation'
            ]
        })
        
        # Write metadata
        metadata.to_excel(writer, sheet_name='Current Recommendation', index=False, startrow=0)
        
        # Write allocation
        current_display = current.copy()
        current_display['Weight'] = (current_display['Weight'] * 100).round(2)
        current_display.columns = ['ETF', 'Weight (%)', 'Final Score', 'ML Probability', 'Tech Score']
        
        current_display.to_excel(
            writer,
            sheet_name='Current Recommendation',
            index=False,
            startrow=len(metadata) + 3
        )
    
    def _write_feature_importance_sheet(self, writer):
        """Write feature importance from models"""
        if self.runner.trainer is None or self.runner.models_dict is None:
            return
        
        importance_df = self.runner.trainer.get_feature_importance(self.runner.models_dict)
        importance_df.to_excel(writer, sheet_name='Feature Importance', index=False)
    
    def _write_equity_curve_sheet(self, writer):
        """Write equity curve data"""
        equity_df = self.runner.portfolio.get_equity_curve()
        
        if equity_df.empty:
            return
        
        # Add benchmark for comparison
        benchmark_data = self.runner.data['NIFTYBEES']
        
        equity_df['Date'] = pd.to_datetime(equity_df['Date'])
        
        # Align benchmark
        equity_with_bench = equity_df.set_index('Date')
        
        # Calculate benchmark normalized to initial capital
        benchmark_dates = equity_with_bench.index
        benchmark_aligned = benchmark_data.loc[benchmark_dates[0]:benchmark_dates[-1]]
        
        benchmark_returns = (benchmark_aligned['Close'] / benchmark_aligned['Close'].iloc[0]) * self.runner.initial_capital
        
        equity_with_bench['Benchmark_Value'] = benchmark_returns.reindex(equity_with_bench.index, method='ffill')
        
        equity_with_bench.to_excel(writer, sheet_name='Equity Curve')
    
    def generate_trade_log_only(self, filename='Trade_Log.xlsx'):
        """Generate simplified trade log report"""
        filepath = os.path.join(self.output_dir, filename)
        
        trades_df = self.runner.portfolio.get_trades()
        
        if trades_df.empty:
            print("No trades to export")
            return None
        
        # Enhanced trade log with calculations
        sell_trades = trades_df[trades_df['Action'] == 'SELL'].copy()
        
        if not sell_trades.empty:
            # Add holding period
            sell_trades['Entry_Date'] = pd.to_datetime(sell_trades['Entry_Date'])
            sell_trades['Date'] = pd.to_datetime(sell_trades['Date'])
            sell_trades['Holding_Days'] = (sell_trades['Date'] - sell_trades['Entry_Date']).dt.days
            
            # Reorder columns
            cols = ['Date', 'ETF', 'Entry_Date', 'Entry_Price', 'Price', 
                   'Shares', 'PnL', 'PnL_Pct', 'Holding_Days']
            cols = [c for c in cols if c in sell_trades.columns]
            
            sell_trades[cols].to_excel(filepath, index=False)
            print(f"Trade log saved: {filepath}")
            return filepath
        
        return None


def generate_excel_reports(backtest_runner, output_dir='reports'):
    """
    Generate all Excel reports
    
    Args:
        backtest_runner: BacktestRunner object
        output_dir: Output directory
        
    Returns:
        dict: Paths to generated files
    """
    reporter = ExcelReporter(backtest_runner, output_dir=output_dir)
    
    # Generate main report
    main_report = reporter.generate_full_report()
    
    # Generate trade log
    trade_log = reporter.generate_trade_log_only()
    
    return {
        'full_report': main_report,
        'trade_log': trade_log
    }


if __name__ == '__main__':
    from backtest.backtest_runner import run_backtest
    
    # Run backtest
    print("Running backtest...")
    runner = run_backtest(
        start_date='2015-01-01',
        top_n=3,
        initial_capital=1000000
    )
    
    # Generate reports
    print("\nGenerating reports...")
    reports = generate_excel_reports(runner)
    
    print("\n" + "="*80)
    print("REPORTS GENERATED")
    print("="*80)
    for report_type, filepath in reports.items():
        if filepath:
            print(f"{report_type:20s}: {filepath}")
