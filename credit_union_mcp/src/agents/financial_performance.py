"""
Financial Performance Agent for Credit Union Analytics

Provides comprehensive financial performance analysis including:
- NCUA 5300 call report calculations
- Profitability ratios (ROA, ROE, NIM)
- Capital adequacy analysis
- Efficiency metrics
- Peer group benchmarking
- Trend analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import math

from .base_agent import BaseAgent, AnalysisContext, AnalysisResult


class FinancialPerformanceAgent(BaseAgent):
    """
    Specialized agent for financial performance analysis.
    
    Calculates key financial metrics, ratios, and performance indicators
    used in credit union financial analysis and regulatory reporting.
    """
    
    def get_capabilities(self) -> List[str]:
        """Return list of analysis capabilities."""
        return [
            "profitability_analysis",
            "capital_analysis", 
            "efficiency_analysis",
            "ncua_5300_metrics",
            "peer_benchmarking",
            "trend_analysis",
            "roa_calculation",
            "roe_calculation", 
            "nim_calculation",
            "efficiency_ratio",
            "net_worth_analysis"
        ]
    
    def analyze(self, context: AnalysisContext) -> AnalysisResult:
        """
        Main analysis method for financial performance.
        
        Args:
            context: Analysis context with parameters
            
        Returns:
            AnalysisResult with financial performance metrics
        """
        self.log_analysis_start("financial_performance", context)
        
        try:
            analysis_type = context.parameters.get('metric_type', 'all')
            as_of_date = context.as_of_date or datetime.now().strftime('%Y-%m-%d')
            
            # Route to specific analysis based on type
            if analysis_type == 'profitability':
                result = self._analyze_profitability(context.database, as_of_date)
            elif analysis_type == 'capital':
                result = self._analyze_capital(context.database, as_of_date)
            elif analysis_type == 'efficiency':
                result = self._analyze_efficiency(context.database, as_of_date)
            elif analysis_type == 'all':
                result = self._comprehensive_analysis(context.database, as_of_date)
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
            
            self.log_analysis_complete("financial_performance", result)
            return result
            
        except Exception as e:
            return self.handle_analysis_error("financial_performance", e)
    
    def _comprehensive_analysis(self, database: str, as_of_date: str) -> AnalysisResult:
        """
        Perform comprehensive financial performance analysis.
        
        Args:
            database: Target database
            as_of_date: Analysis date
            
        Returns:
            AnalysisResult with comprehensive metrics
        """
        # Get financial data
        financial_data = self._get_financial_data(database, as_of_date)
        
        if financial_data.empty:
            return self.create_result(
                analysis_type="comprehensive",
                success=False,
                errors=["No financial data available for the specified date"]
            )
        
        # Calculate all metrics
        profitability_metrics = self._calculate_profitability_metrics(financial_data)
        capital_metrics = self._calculate_capital_metrics(financial_data)
        efficiency_metrics = self._calculate_efficiency_metrics(financial_data)
        
        # Combine all metrics
        all_metrics = {
            **profitability_metrics,
            **capital_metrics,
            **efficiency_metrics
        }
        
        # Generate analysis data
        analysis_data = {
            'profitability': {
                'roa': profitability_metrics.get('roa'),
                'roe': profitability_metrics.get('roe'),
                'nim': profitability_metrics.get('nim'),
                'non_interest_income_ratio': profitability_metrics.get('non_interest_income_ratio')
            },
            'capital': {
                'net_worth_ratio': capital_metrics.get('net_worth_ratio'),
                'leverage_ratio': capital_metrics.get('leverage_ratio'),
                'risk_based_capital_ratio': capital_metrics.get('risk_based_capital_ratio')
            },
            'efficiency': {
                'efficiency_ratio': efficiency_metrics.get('efficiency_ratio'),
                'cost_per_member': efficiency_metrics.get('cost_per_member'),
                'asset_per_employee': efficiency_metrics.get('asset_per_employee')
            },
            'as_of_date': as_of_date,
            'summary': self._generate_performance_summary(all_metrics)
        }
        
        return self.create_result(
            analysis_type="comprehensive",
            data=analysis_data,
            metrics=all_metrics,
            metadata={'calculation_date': as_of_date}
        )
    
    def _analyze_profitability(self, database: str, as_of_date: str) -> AnalysisResult:
        """Analyze profitability metrics."""
        financial_data = self._get_financial_data(database, as_of_date)
        
        if financial_data.empty:
            return self.create_result(
                analysis_type="profitability",
                success=False,
                errors=["No financial data available"]
            )
        
        metrics = self._calculate_profitability_metrics(financial_data)
        
        # Get historical data for trends
        historical_data = self._get_historical_trends(database, as_of_date, 'profitability')
        
        return self.create_result(
            analysis_type="profitability",
            data={
                'current_metrics': metrics,
                'trends': historical_data,
                'peer_comparison': self._get_peer_benchmarks('profitability')
            },
            metrics=metrics
        )
    
    def _analyze_capital(self, database: str, as_of_date: str) -> AnalysisResult:
        """Analyze capital adequacy metrics."""
        financial_data = self._get_financial_data(database, as_of_date)
        
        if financial_data.empty:
            return self.create_result(
                analysis_type="capital",
                success=False,
                errors=["No financial data available"]
            )
        
        metrics = self._calculate_capital_metrics(financial_data)
        
        # Determine capital classification
        capital_classification = self._classify_capital_adequacy(metrics)
        
        return self.create_result(
            analysis_type="capital",
            data={
                'current_metrics': metrics,
                'classification': capital_classification,
                'regulatory_requirements': self._get_regulatory_requirements(),
                'peer_comparison': self._get_peer_benchmarks('capital')
            },
            metrics=metrics
        )
    
    def _analyze_efficiency(self, database: str, as_of_date: str) -> AnalysisResult:
        """Analyze operational efficiency metrics."""
        financial_data = self._get_financial_data(database, as_of_date)
        
        if financial_data.empty:
            return self.create_result(
                analysis_type="efficiency",
                success=False,
                errors=["No financial data available"]
            )
        
        metrics = self._calculate_efficiency_metrics(financial_data)
        
        return self.create_result(
            analysis_type="efficiency",
            data={
                'current_metrics': metrics,
                'benchmarks': self._get_efficiency_benchmarks(),
                'recommendations': self._generate_efficiency_recommendations(metrics)
            },
            metrics=metrics
        )
    
    def _get_financial_data(self, database: str, as_of_date: str) -> pd.DataFrame:
        """
        Retrieve financial data for analysis.
        
        Args:
            database: Target database
            as_of_date: Analysis date
            
        Returns:
            DataFrame with financial data
        """
        query = """
        SELECT 
            -- Assets
            ISNULL(total_assets, 0) as total_assets,
            ISNULL(earning_assets, 0) as earning_assets,
            ISNULL(loans_outstanding, 0) as loans_outstanding,
            ISNULL(investments, 0) as investments,
            ISNULL(cash_equivalents, 0) as cash_equivalents,
            
            -- Liabilities  
            ISNULL(total_liabilities, 0) as total_liabilities,
            ISNULL(member_deposits, 0) as member_deposits,
            ISNULL(borrowed_funds, 0) as borrowed_funds,
            
            -- Equity
            ISNULL(total_equity, 0) as total_equity,
            ISNULL(retained_earnings, 0) as retained_earnings,
            ISNULL(undivided_earnings, 0) as undivided_earnings,
            
            -- Income Statement (YTD)
            ISNULL(interest_income, 0) as interest_income,
            ISNULL(interest_expense, 0) as interest_expense,
            ISNULL(net_interest_income, 0) as net_interest_income,
            ISNULL(fee_income, 0) as fee_income,
            ISNULL(other_income, 0) as other_income,
            ISNULL(total_income, 0) as total_income,
            ISNULL(operating_expenses, 0) as operating_expenses,
            ISNULL(provision_for_losses, 0) as provision_for_losses,
            ISNULL(net_income, 0) as net_income,
            
            -- Operational Data
            ISNULL(member_count, 0) as member_count,
            ISNULL(employee_count, 0) as employee_count,
            
            -- Date
            report_date
        FROM financial_summary 
        WHERE report_date <= ?
        ORDER BY report_date DESC
        """
        
        try:
            result = self.execute_query(query, database, params={'as_of_date': as_of_date})
            return result.head(1) if not result.empty else pd.DataFrame()
        except Exception as e:
            self.logger.warning(f"Could not retrieve financial data: {e}")
            return pd.DataFrame()
    
    def _calculate_profitability_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate profitability ratios."""
        if data.empty:
            return {}
        
        row = data.iloc[0]
        
        # Get values with safe conversion
        total_assets = float(row.get('total_assets', 0))
        total_equity = float(row.get('total_equity', 0))
        earning_assets = float(row.get('earning_assets', 0))
        net_interest_income = float(row.get('net_interest_income', 0))
        net_income = float(row.get('net_income', 0))
        total_income = float(row.get('total_income', 0))
        fee_income = float(row.get('fee_income', 0))
        other_income = float(row.get('other_income', 0))
        
        metrics = {}
        
        # Return on Assets (ROA) - annualized
        metrics['roa'] = self.safe_divide(net_income * 4, total_assets) * 100  # Quarterly to annual
        
        # Return on Equity (ROE) - annualized  
        metrics['roe'] = self.safe_divide(net_income * 4, total_equity) * 100
        
        # Net Interest Margin (NIM) - annualized
        metrics['nim'] = self.safe_divide(net_interest_income * 4, earning_assets) * 100
        
        # Non-Interest Income Ratio
        non_interest_income = fee_income + other_income
        metrics['non_interest_income_ratio'] = self.safe_divide(non_interest_income, total_income) * 100
        
        # Interest Spread
        interest_income = float(row.get('interest_income', 0))
        interest_expense = float(row.get('interest_expense', 0))
        metrics['interest_spread'] = self.safe_divide(
            (interest_income - interest_expense) * 4, earning_assets
        ) * 100
        
        # Yield on Earning Assets
        metrics['yield_on_earning_assets'] = self.safe_divide(
            interest_income * 4, earning_assets
        ) * 100
        
        return metrics
    
    def _calculate_capital_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate capital adequacy ratios."""
        if data.empty:
            return {}
        
        row = data.iloc[0]
        
        total_assets = float(row.get('total_assets', 0))
        total_equity = float(row.get('total_equity', 0))
        retained_earnings = float(row.get('retained_earnings', 0))
        
        metrics = {}
        
        # Net Worth Ratio (Primary regulatory capital ratio)
        metrics['net_worth_ratio'] = self.safe_divide(total_equity, total_assets) * 100
        
        # Leverage Ratio
        metrics['leverage_ratio'] = self.safe_divide(total_equity, total_assets) * 100
        
        # Risk-Based Capital Ratio (simplified - would need risk-weighted assets)
        # Using total assets as proxy for risk-weighted assets
        metrics['risk_based_capital_ratio'] = self.safe_divide(total_equity, total_assets) * 100
        
        # Equity to Asset Ratio
        metrics['equity_to_asset_ratio'] = self.safe_divide(total_equity, total_assets) * 100
        
        # Retained Earnings Ratio
        metrics['retained_earnings_ratio'] = self.safe_divide(retained_earnings, total_equity) * 100
        
        return metrics
    
    def _calculate_efficiency_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate operational efficiency ratios."""
        if data.empty:
            return {}
        
        row = data.iloc[0]
        
        operating_expenses = float(row.get('operating_expenses', 0))
        total_income = float(row.get('total_income', 0))
        total_assets = float(row.get('total_assets', 0))
        member_count = float(row.get('member_count', 1))  # Avoid division by zero
        employee_count = float(row.get('employee_count', 1))
        
        metrics = {}
        
        # Efficiency Ratio (Operating Expense / Total Income)
        metrics['efficiency_ratio'] = self.safe_divide(operating_expenses, total_income) * 100
        
        # Operating Expense Ratio (Operating Expense / Average Assets)
        metrics['operating_expense_ratio'] = self.safe_divide(
            operating_expenses * 4, total_assets
        ) * 100  # Annualized
        
        # Cost per Member
        metrics['cost_per_member'] = self.safe_divide(operating_expenses * 4, member_count)
        
        # Assets per Employee
        metrics['asset_per_employee'] = self.safe_divide(total_assets, employee_count)
        
        # Members per Employee
        metrics['members_per_employee'] = self.safe_divide(member_count, employee_count)
        
        # Asset Utilization (Income / Assets)
        metrics['asset_utilization'] = self.safe_divide(total_income * 4, total_assets) * 100
        
        return metrics
    
    def _classify_capital_adequacy(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """
        Classify capital adequacy based on NCUA standards.
        
        Args:
            metrics: Capital metrics dictionary
            
        Returns:
            Dictionary with capital classifications
        """
        net_worth_ratio = metrics.get('net_worth_ratio', 0)
        
        # NCUA Capital Classifications
        if net_worth_ratio >= 10.0:
            classification = "Well Capitalized"
            status = "Excellent"
        elif net_worth_ratio >= 8.0:
            classification = "Adequately Capitalized"
            status = "Good"
        elif net_worth_ratio >= 6.0:
            classification = "Undercapitalized"
            status = "Concern"
        elif net_worth_ratio >= 2.0:
            classification = "Significantly Undercapitalized"
            status = "Critical"
        else:
            classification = "Critically Undercapitalized"
            status = "Severe"
        
        return {
            'classification': classification,
            'status': status,
            'net_worth_ratio': net_worth_ratio,
            'required_minimum': 7.0,
            'well_capitalized_threshold': 10.0
        }
    
    def _get_regulatory_requirements(self) -> Dict[str, float]:
        """Get regulatory capital requirements."""
        return {
            'minimum_net_worth_ratio': 7.0,
            'well_capitalized_threshold': 10.0,
            'adequately_capitalized_threshold': 8.0,
            'undercapitalized_threshold': 6.0,
            'significantly_undercapitalized_threshold': 2.0
        }
    
    def _get_peer_benchmarks(self, metric_type: str) -> Dict[str, float]:
        """
        Get peer group benchmarks for comparison.
        
        Args:
            metric_type: Type of metrics to benchmark
            
        Returns:
            Dictionary with peer benchmarks
        """
        # Industry averages (these would typically come from NCUA data)
        benchmarks = {
            'profitability': {
                'roa_median': 0.75,
                'roe_median': 8.5,
                'nim_median': 3.2,
                'non_interest_income_ratio_median': 25.0
            },
            'capital': {
                'net_worth_ratio_median': 11.2,
                'leverage_ratio_median': 11.2,
                'equity_to_asset_ratio_median': 11.2
            },
            'efficiency': {
                'efficiency_ratio_median': 75.0,
                'cost_per_member_median': 425.0,
                'asset_per_employee_median': 4200000.0
            }
        }
        
        return benchmarks.get(metric_type, {})
    
    def _get_efficiency_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Get efficiency benchmarks by credit union size."""
        return {
            'small_cu': {  # Under $100M assets
                'efficiency_ratio': 78.0,
                'cost_per_member': 520.0,
                'asset_per_employee': 3500000.0
            },
            'medium_cu': {  # $100M - $1B assets
                'efficiency_ratio': 72.0,
                'cost_per_member': 380.0,
                'asset_per_employee': 4800000.0
            },
            'large_cu': {  # Over $1B assets
                'efficiency_ratio': 68.0,
                'cost_per_member': 310.0,
                'asset_per_employee': 6200000.0
            }
        }
    
    def _generate_efficiency_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Generate efficiency improvement recommendations."""
        recommendations = []
        
        efficiency_ratio = metrics.get('efficiency_ratio', 0)
        cost_per_member = metrics.get('cost_per_member', 0)
        
        if efficiency_ratio > 80:
            recommendations.append("Consider cost reduction initiatives - efficiency ratio above industry norms")
        
        if cost_per_member > 500:
            recommendations.append("High cost per member - evaluate member growth strategies")
        
        if efficiency_ratio < 60:
            recommendations.append("Excellent efficiency - monitor for sustainable growth opportunities")
        
        return recommendations
    
    def _get_historical_trends(self, database: str, as_of_date: str, metric_type: str) -> Dict[str, Any]:
        """Get historical trend data for metrics."""
        # This would query historical data for trend analysis
        # For now, return placeholder structure
        return {
            'periods': ['Q1', 'Q2', 'Q3', 'Q4'],
            'roa_trend': [0.65, 0.72, 0.78, 0.75],
            'roe_trend': [7.8, 8.2, 8.6, 8.4],
            'nim_trend': [3.1, 3.2, 3.3, 3.2]
        }
    
    def _generate_performance_summary(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """Generate overall performance summary."""
        roa = metrics.get('roa', 0)
        net_worth_ratio = metrics.get('net_worth_ratio', 0)
        efficiency_ratio = metrics.get('efficiency_ratio', 0)
        
        # Overall rating logic
        if roa > 1.0 and net_worth_ratio > 10 and efficiency_ratio < 70:
            overall_rating = "Excellent"
        elif roa > 0.75 and net_worth_ratio > 8 and efficiency_ratio < 75:
            overall_rating = "Good"
        elif roa > 0.5 and net_worth_ratio > 7 and efficiency_ratio < 80:
            overall_rating = "Satisfactory"
        else:
            overall_rating = "Needs Improvement"
        
        return {
            'overall_rating': overall_rating,
            'profitability_status': "Strong" if roa > 0.75 else "Moderate" if roa > 0.5 else "Weak",
            'capital_status': "Strong" if net_worth_ratio > 10 else "Adequate" if net_worth_ratio > 7 else "Weak",
            'efficiency_status': "Excellent" if efficiency_ratio < 70 else "Good" if efficiency_ratio < 75 else "Needs Improvement"
        }
