"""
Main execution script for Indian ETF Sector Rotation System
"""
import os
import sys
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest.backtest_runner import run_backtest
from reports.excel_report import generate_excel_reports
from reports.charts import generate_charts


def main():
    """Main execution function"""
    
    print("="*80)
    print("INDIAN ETF SECTOR ROTATION SYSTEM")
    print("AI-Powered Monthly Sector Rotation Strategy")
    print("="*80)
    print()
    
    # Configuration
    START_DATE = '2015-01-01'
    END_DATE = None  # Today
    TOP_N_ETFS = 3
    INITIAL_CAPITAL = 1000000  # ₹10 Lakh
    
    print("Configuration:")
    print(f"  Start Date:       {START_DATE}")
    print(f"  End Date:         {END_DATE or 'Today'}")
    print(f"  Top N ETFs:       {TOP_N_ETFS}")
    print(f"  Initial Capital:  ₹{INITIAL_CAPITAL:,}")
    print()
    
    # Step 1: Run backtest
    print("STEP 1: Running Complete Backtest")
    print("-" * 80)
    
    runner = run_backtest(
        start_date=START_DATE,
        end_date=END_DATE,
        top_n=TOP_N_ETFS,
        initial_capital=INITIAL_CAPITAL,
        force_refresh=False
    )
    
    print("\n✅ Backtest completed successfully!")
    
    # Step 2: Generate Excel reports
    print("\n" + "="*80)
    print("STEP 2: Generating Excel Reports")
    print("-" * 80)
    
    reports = generate_excel_reports(runner, output_dir='reports')
    
    print("\n✅ Excel reports generated:")
    for report_type, filepath in reports.items():
        if filepath:
            print(f"   {report_type:20s}: {filepath}")
    
    # Step 3: Generate charts
    print("\n" + "="*80)
    print("STEP 3: Generating Charts")
    print("-" * 80)
    
    charts = generate_charts(runner, output_dir='reports/charts')
    
    print("\n✅ Charts generated:")
    for chart_name, filepath in charts.items():
        if filepath:
            print(f"   {chart_name:25s}: {filepath}")
    
    # Step 4: Display current recommendation
    print("\n" + "="*80)
    print("CURRENT RECOMMENDED ALLOCATION")
    print("="*80)
    
    current_allocation = runner.get_current_allocation()
    
    if current_allocation is not None and not current_allocation.empty:
        print("\n📊 Portfolio Allocation:")
        print()
        
        display_df = current_allocation.copy()
        display_df['Weight'] = (display_df['Weight'] * 100).round(2)
        
        for _, row in display_df.iterrows():
            print(f"   {row['ETF']:20s}: {row['Weight']:6.2f}%  "
                  f"(Score: {row['FinalScore']:.3f})")
        
        # Add cash
        total_weight = display_df['Weight'].sum()
        cash_weight = 100 - total_weight
        print(f"   {'CASH':20s}: {cash_weight:6.2f}%")
    else:
        print("No current allocation available")
    
    # Step 5: Display next month predictions
    print("\n" + "="*80)
    print("NEXT MONTH TOP PREDICTIONS")
    print("="*80)
    
    predictions = runner.get_next_month_predictions()
    
    if predictions is not None and not predictions.empty:
        print()
        for idx, row in predictions.head(5).iterrows():
            print(f"{row['ETF']:20s}: Final Score = {row['FinalScore']:.3f}  "
                  f"(ML: {row['ML_Probability']:.3f}, Tech: {row['TechScore']:.3f})")
    
    # Summary
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    print()
    print("✅ All tasks completed successfully!")
    print()
    print("Next steps:")
    print("  1. Review Excel reports in ./reports/ directory")
    print("  2. View charts in ./reports/charts/ directory")
    print("  3. Launch Streamlit dashboard:")
    print("     → streamlit run app/dashboard.py")
    print()
    print("="*80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
