"""
Portfolio Risk Agent for Credit Union Analytics

Provides comprehensive loan portfolio risk analysis including:
- Concentration risk analysis (HHI index)
- Delinquency trending and forecasting
- Loss rate analysis and projections
- Stress testing (Monte Carlo simulation)
- Vintage analysis
- Credit quality metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import math
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from .base_agent import BaseAgent, AnalysisContext, AnalysisResult


class PortfolioRiskAgent(BaseAgent):
    """
    Specialized agent for loan portfolio risk analysis.
    
    Provides advanced risk analytics including concentration analysis,
    delinquency forecasting, stress testing, and credit quality assessment.
    """
    
    def get_capabilities(self) -> List[str]:
        """Return list of analysis capabilities."""
        return [
            "concentration_analysis",
            "hhi_calculation",
            "delinquency_forecasting", 
            "loss_rate_analysis",
            "stress_testing",
            "monte_carlo_simulation",
            "vintage_analysis",
            "credit_quality_assessment",
            "portfolio_composition",
            "risk_adjusted_returns",
            "var_calculation",
            "scenario_analysis"
        ]
    
    def analyze(self, context: AnalysisContext) -> AnalysisResult:
        """
        Main analysis method for portfolio risk.
        
        Args:
            context: Analysis context with parameters
            
        Returns:
            AnalysisResult with risk analysis results
        """
        self.log_analysis_start("portfolio_risk", context)
        
        try:
            analysis_type = context.parameters.get('analysis_type', 'comprehensive')
            as_of_date = context.as_of_date or datetime.now().strftime('%Y-%m-%d')
            
            # Route to specific analysis based on type
            if analysis_type == 'concentration':
                result = self._analyze_concentration(context.database, as_of_date)
            elif analysis_type == 'delinquency':
                result = self._analyze_delinquency(context.database, as_of_date)
            elif analysis_type == 'stress_test':
                result = self._perform_stress_test(context.database, as_of_date, context.parameters)
            elif analysis_type == 'comprehensive':
                result = self._comprehensive_risk_analysis(context.database, as_of_date)
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
            
            self.log_analysis_complete("portfolio_risk", result)
            return result
            
        except Exception as e:
            return self.handle_analysis_error("portfolio_risk", e)
    
    def _comprehensive_risk_analysis(self, database: str, as_of_date: str) -> AnalysisResult:
        """
        Perform comprehensive portfolio risk analysis.
        
        Args:
            database: Target database
            as_of_date: Analysis date
            
        Returns:
            AnalysisResult with comprehensive risk metrics
        """
        # Get portfolio data
        portfolio_data = self._get_portfolio_data(database, as_of_date)
        delinquency_data = self._get_delinquency_data(database, as_of_date)
        
        if portfolio_data.empty:
            return self.create_result(
                analysis_type="comprehensive",
                success=False,
                errors=["No portfolio data available for the specified date"]
            )
        
        # Calculate all risk metrics
        concentration_metrics = self._calculate_concentration_metrics(portfolio_data)
        delinquency_metrics = self._calculate_delinquency_metrics(delinquency_data)
        credit_quality_metrics = self._calculate_credit_quality_metrics(portfolio_data)
        loss_metrics = self._calculate_loss_metrics(portfolio_data)
        
        # Combine all metrics
        all_metrics = {
            **concentration_metrics,
            **delinquency_metrics, 
            **credit_quality_metrics,
            **loss_metrics
        }
        
        # Generate risk forecast
        risk_forecast = self._generate_risk_forecast(database, as_of_date)
        
        # Calculate risk score
        risk_score = self._calculate_overall_risk_score(all_metrics)
        
        analysis_data = {
            'concentration': {
                'hhi_index': concentration_metrics.get('hhi_index'),
                'top_10_concentration': concentration_metrics.get('top_10_concentration'),
                'geographic_concentration': concentration_metrics.get('geographic_concentration'),
                'product_concentration': concentration_metrics.get('product_concentration')
            },
            'delinquency': {
                'current_delinquency_rate': delinquency_metrics.get('current_delinquency_rate'),
                'delinquency_trend': delinquency_metrics.get('delinquency_trend'),
                'forecasted_delinquency': risk_forecast.get('delinquency_forecast')
            },
            'credit_quality': {
                'weighted_avg_fico': credit_quality_metrics.get('weighted_avg_fico'),
                'high_risk_loans_pct': credit_quality_metrics.get('high_risk_loans_pct'),
                'ltv_average': credit_quality_metrics.get('ltv_average')
            },
            'losses': {
                'net_charge_off_rate': loss_metrics.get('net_charge_off_rate'),
                'provision_coverage': loss_metrics.get('provision_coverage'),
                'recovery_rate': loss_metrics.get('recovery_rate')
            },
            'overall_assessment': {
                'risk_score': risk_score,
                'risk_rating': self._get_risk_rating(risk_score),
                'key_concerns': self._identify_key_concerns(all_metrics)
            },
            'as_of_date': as_of_date
        }
        
        return self.create_result(
            analysis_type="comprehensive",
            data=analysis_data,
            metrics=all_metrics,
            metadata={'calculation_date': as_of_date}
        )
    
    def _analyze_concentration(self, database: str, as_of_date: str) -> AnalysisResult:
        """Analyze portfolio concentration risk."""
        portfolio_data = self._get_portfolio_data(database, as_of_date)
        
        if portfolio_data.empty:
            return self.create_result(
                analysis_type="concentration",
                success=False,
                errors=["No portfolio data available"]
            )
        
        metrics = self._calculate_concentration_metrics(portfolio_data)
        
        # Calculate detailed concentration breakdowns
        breakdowns = self._calculate_concentration_breakdowns(portfolio_data)
        
        return self.create_result(
            analysis_type="concentration",
            data={
                'metrics': metrics,
                'breakdowns': breakdowns,
                'recommendations': self._generate_concentration_recommendations(metrics)
            },
            metrics=metrics
        )
    
    def _analyze_delinquency(self, database: str, as_of_date: str) -> AnalysisResult:
        """Analyze delinquency trends and forecast."""
        delinquency_data = self._get_delinquency_data(database, as_of_date)
        
        if delinquency_data.empty:
            return self.create_result(
                analysis_type="delinquency",
                success=False,
                errors=["No delinquency data available"]
            )
        
        metrics = self._calculate_delinquency_metrics(delinquency_data)
        forecast = self._forecast_delinquency(delinquency_data)
        
        return self.create_result(
            analysis_type="delinquency",
            data={
                'current_metrics': metrics,
                'forecast': forecast,
                'trends': self._analyze_delinquency_trends(delinquency_data)
            },
            metrics=metrics
        )
    
    def _perform_stress_test(self, database: str, as_of_date: str, parameters: Dict[str, Any]) -> AnalysisResult:
        """Perform portfolio stress testing."""
        portfolio_data = self._get_portfolio_data(database, as_of_date)
        
        if portfolio_data.empty:
            return self.create_result(
                analysis_type="stress_test",
                success=False,
                errors=["No portfolio data available for stress testing"]
            )
        
        # Get stress test parameters
        unemployment_shock = parameters.get('unemployment_shock', 2.0)  # 2% increase
        house_price_shock = parameters.get('house_price_shock', -15.0)  # 15% decline
        interest_rate_shock = parameters.get('interest_rate_shock', 3.0)  # 3% increase
        
        stress_results = self._run_stress_scenarios(
            portfolio_data, 
            unemployment_shock, 
            house_price_shock, 
            interest_rate_shock
        )
        
        return self.create_result(
            analysis_type="stress_test",
            data=stress_results,
            metrics=stress_results.get('summary_metrics', {})
        )
    
    def _get_portfolio_data(self, database: str, as_of_date: str) -> pd.DataFrame:
        """
        Retrieve loan portfolio data for analysis.
        
        Args:
            database: Target database
            as_of_date: Analysis date
            
        Returns:
            DataFrame with portfolio data
        """
        query = """
        SELECT 
            loan_id,
            member_id,
            loan_type,
            original_balance,
            current_balance,
            interest_rate,
            origination_date,
            maturity_date,
            payment_status,
            days_delinquent,
            fico_score,
            loan_to_value_ratio,
            debt_to_income_ratio,
            geographic_region,
            collateral_type,
            collateral_value,
            charge_off_amount,
            recovery_amount,
            provision_amount
        FROM loan_portfolio lp
        WHERE lp.as_of_date <= ?
        AND lp.loan_status IN ('Active', 'Delinquent', 'Charged Off')
        """
        
        try:
            result = self.execute_query(query, database, params={'as_of_date': as_of_date})
            return result
        except Exception as e:
            self.logger.warning(f"Could not retrieve portfolio data: {e}")
            return pd.DataFrame()
    
    def _get_delinquency_data(self, database: str, as_of_date: str) -> pd.DataFrame:
        """Retrieve historical delinquency data."""
        query = """
        SELECT 
            report_date,
            delinquency_30_days,
            delinquency_60_days,
            delinquency_90_days,
            total_delinquent_amount,
            total_loan_balance,
            charge_offs_month,
            recoveries_month
        FROM delinquency_summary
        WHERE report_date <= ?
        ORDER BY report_date DESC
        """
        
        try:
            result = self.execute_query(query, database, params={'as_of_date': as_of_date})
            return result.head(24)  # Last 24 months
        except Exception as e:
            self.logger.warning(f"Could not retrieve delinquency data: {e}")
            return pd.DataFrame()
    
    def _calculate_concentration_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate portfolio concentration metrics."""
        if data.empty:
            return {}
        
        metrics = {}
        
        # HHI Index calculation
        if 'current_balance' in data.columns:
            balances = data['current_balance']
            hhi = self._calculate_hhi(balances)
            metrics['hhi_index'] = hhi
            
            # Top 10 borrower concentration
            top_10_concentration = balances.nlargest(10).sum() / balances.sum() * 100
            metrics['top_10_concentration'] = top_10_concentration
        
        # Geographic concentration
        if 'geographic_region' in data.columns:
            geo_concentration = self._calculate_geographic_concentration(data)
            metrics['geographic_concentration'] = geo_concentration
        
        # Product type concentration
        if 'loan_type' in data.columns:
            product_concentration = self._calculate_product_concentration(data)
            metrics['product_concentration'] = product_concentration
        
        # Large loan concentration (loans > 1% of capital)
        if 'current_balance' in data.columns:
            total_balance = data['current_balance'].sum()
            large_loans = data[data['current_balance'] > total_balance * 0.01]
            metrics['large_loan_concentration'] = len(large_loans) / len(data) * 100
        
        return metrics
    
    def _calculate_hhi(self, balances: pd.Series) -> float:
        """
        Calculate Herfindahl-Hirschman Index for concentration.
        
        Args:
            balances: Series of loan balances
            
        Returns:
            HHI index (0-10000)
        """
        if balances.empty or balances.sum() == 0:
            return 0.0
        
        total = balances.sum()
        market_shares = balances / total
        hhi = (market_shares ** 2).sum() * 10000
        
        return float(hhi)
    
    def _calculate_geographic_concentration(self, data: pd.DataFrame) -> float:
        """Calculate geographic concentration using HHI."""
        if 'geographic_region' not in data.columns or 'current_balance' not in data.columns:
            return 0.0
        
        regional_balances = data.groupby('geographic_region')['current_balance'].sum()
        return self._calculate_hhi(regional_balances)
    
    def _calculate_product_concentration(self, data: pd.DataFrame) -> float:
        """Calculate product type concentration using HHI."""
        if 'loan_type' not in data.columns or 'current_balance' not in data.columns:
            return 0.0
        
        product_balances = data.groupby('loan_type')['current_balance'].sum()
        return self._calculate_hhi(product_balances)
    
    def _calculate_delinquency_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate delinquency-related metrics."""
        if data.empty:
            return {}
        
        metrics = {}
        
        # Current delinquency rates
        if len(data) > 0:
            latest_row = data.iloc[0]
            
            total_balance = float(latest_row.get('total_loan_balance', 1))
            
            metrics['current_delinquency_rate'] = (
                float(latest_row.get('total_delinquent_amount', 0)) / total_balance * 100
            )
            
            metrics['delinquency_30_rate'] = (
                float(latest_row.get('delinquency_30_days', 0)) / total_balance * 100
            )
            
            metrics['delinquency_60_rate'] = (
                float(latest_row.get('delinquency_60_days', 0)) / total_balance * 100
            )
            
            metrics['delinquency_90_rate'] = (
                float(latest_row.get('delinquency_90_days', 0)) / total_balance * 100
            )
        
        # Trend analysis
        if len(data) >= 6:
            recent_trend = self._calculate_delinquency_trend(data.head(6))
            metrics['delinquency_trend'] = recent_trend
        
        return metrics
    
    def _calculate_delinquency_trend(self, data: pd.DataFrame) -> float:
        """Calculate delinquency trend (6-month slope)."""
        if len(data) < 3:
            return 0.0
        
        # Calculate delinquency rates
        data = data.copy()
        data['delinq_rate'] = (
            data['total_delinquent_amount'] / data['total_loan_balance'] * 100
        )
        
        # Use linear regression to find trend
        x = np.arange(len(data))
        y = data['delinq_rate'].values
        
        if len(y) < 2:
            return 0.0
        
        slope, _, _, _, _ = stats.linregress(x, y)
        return float(slope)
    
    def _calculate_credit_quality_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate credit quality metrics."""
        if data.empty:
            return {}
        
        metrics = {}
        
        # Weighted average FICO score
        if 'fico_score' in data.columns and 'current_balance' in data.columns:
            valid_fico = data[data['fico_score'] > 0]
            if not valid_fico.empty:
                weighted_fico = np.average(
                    valid_fico['fico_score'], 
                    weights=valid_fico['current_balance']
                )
                metrics['weighted_avg_fico'] = float(weighted_fico)
        
        # High-risk loans percentage (FICO < 650)
        if 'fico_score' in data.columns:
            high_risk_loans = data[data['fico_score'] < 650]
            metrics['high_risk_loans_pct'] = len(high_risk_loans) / len(data) * 100
        
        # Average LTV ratio
        if 'loan_to_value_ratio' in data.columns:
            valid_ltv = data[data['loan_to_value_ratio'] > 0]
            if not valid_ltv.empty:
                avg_ltv = valid_ltv['loan_to_value_ratio'].mean()
                metrics['ltv_average'] = float(avg_ltv)
        
        # Average DTI ratio
        if 'debt_to_income_ratio' in data.columns:
            valid_dti = data[data['debt_to_income_ratio'] > 0]
            if not valid_dti.empty:
                avg_dti = valid_dti['debt_to_income_ratio'].mean()
                metrics['dti_average'] = float(avg_dti)
        
        return metrics
    
    def _calculate_loss_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate loss and recovery metrics."""
        if data.empty:
            return {}
        
        metrics = {}
        
        # Net charge-off rate
        if 'charge_off_amount' in data.columns and 'current_balance' in data.columns:
            total_charge_offs = data['charge_off_amount'].sum()
            total_balance = data['current_balance'].sum()
            metrics['net_charge_off_rate'] = self.safe_divide(total_charge_offs, total_balance) * 100
        
        # Recovery rate
        if 'recovery_amount' in data.columns and 'charge_off_amount' in data.columns:
            total_recoveries = data['recovery_amount'].sum()
            total_charge_offs = data['charge_off_amount'].sum()
            metrics['recovery_rate'] = self.safe_divide(total_recoveries, total_charge_offs) * 100
        
        # Provision coverage ratio
        if 'provision_amount' in data.columns and 'current_balance' in data.columns:
            total_provisions = data['provision_amount'].sum()
            total_balance = data['current_balance'].sum()
            metrics['provision_coverage'] = self.safe_divide(total_provisions, total_balance) * 100
        
        return metrics
    
    def _forecast_delinquency(self, data: pd.DataFrame, periods: int = 6) -> Dict[str, Any]:
        """
        Forecast delinquency rates using time series analysis.
        
        Args:
            data: Historical delinquency data
            periods: Number of periods to forecast
            
        Returns:
            Dictionary with forecast results
        """
        if data.empty or len(data) < 6:
            return {'forecast': [], 'confidence_intervals': []}
        
        # Prepare data
        data_sorted = data.sort_values('report_date')
        delinq_rates = (
            data_sorted['total_delinquent_amount'] / 
            data_sorted['total_loan_balance'] * 100
        ).values
        
        # Simple exponential smoothing forecast
        alpha = 0.3  # Smoothing parameter
        forecast = []
        
        # Initialize with last observed value
        last_value = delinq_rates[-1]
        
        for i in range(periods):
            # Simple exponential smoothing
            if i == 0:
                forecast_value = last_value
            else:
                forecast_value = alpha * forecast[-1] + (1 - alpha) * forecast_value
            
            forecast.append(float(forecast_value))
        
        # Calculate confidence intervals (simplified)
        std_error = np.std(delinq_rates) if len(delinq_rates) > 1 else 0.1
        confidence_intervals = [
            {
                'lower_bound': max(0, f - 1.96 * std_error),
                'upper_bound': f + 1.96 * std_error
            }
            for f in forecast
        ]
        
        return {
            'forecast': forecast,
            'confidence_intervals': confidence_intervals,
            'forecast_periods': periods
        }
    
    def _run_stress_scenarios(self, data: pd.DataFrame, unemployment_shock: float, 
                            house_price_shock: float, interest_rate_shock: float) -> Dict[str, Any]:
        """
        Run stress test scenarios using Monte Carlo simulation.
        
        Args:
            data: Portfolio data
            unemployment_shock: Unemployment rate increase (%)
            house_price_shock: House price decline (%)
            interest_rate_shock: Interest rate increase (%)
            
        Returns:
            Dictionary with stress test results
        """
        n_simulations = 1000
        
        # Base loss rate (current charge-off rate)
        base_loss_rate = self.safe_divide(
            data['charge_off_amount'].sum(),
            data['current_balance'].sum()
        )
        
        # Stress multipliers based on economic shocks
        unemployment_multiplier = 1 + (unemployment_shock * 0.15)  # 15% increase per 1% unemployment
        house_price_multiplier = 1 + (abs(house_price_shock) * 0.02)  # 2% increase per 1% price decline
        interest_rate_multiplier = 1 + (interest_rate_shock * 0.08)  # 8% increase per 1% rate increase
        
        stress_multiplier = unemployment_multiplier * house_price_multiplier * interest_rate_multiplier
        
        # Monte Carlo simulation
        stressed_loss_rates = []
        total_portfolio_value = data['current_balance'].sum()
        
        for _ in range(n_simulations):
            # Add random variation
            random_factor = np.random.normal(1.0, 0.2)  # 20% volatility
            simulated_loss_rate = base_loss_rate * stress_multiplier * random_factor
            
            # Calculate losses
            total_losses = simulated_loss_rate * total_portfolio_value
            stressed_loss_rates.append(simulated_loss_rate)
        
        # Calculate statistics
        stress_results = {
            'scenario_parameters': {
                'unemployment_shock': unemployment_shock,
                'house_price_shock': house_price_shock,
                'interest_rate_shock': interest_rate_shock
            },
            'base_loss_rate': float(base_loss_rate * 100),
            'stressed_loss_rate_mean': float(np.mean(stressed_loss_rates) * 100),
            'stressed_loss_rate_95th_percentile': float(np.percentile(stressed_loss_rates, 95) * 100),
            'stressed_loss_rate_99th_percentile': float(np.percentile(stressed_loss_rates, 99) * 100),
            'expected_losses': float(np.mean(stressed_loss_rates) * total_portfolio_value),
            'var_95': float(np.percentile(stressed_loss_rates, 95) * total_portfolio_value),
            'var_99': float(np.percentile(stressed_loss_rates, 99) * total_portfolio_value),
            'summary_metrics': {
                'stress_multiplier': float(stress_multiplier),
                'portfolio_value': float(total_portfolio_value),
                'base_losses': float(base_loss_rate * total_portfolio_value)
            }
        }
        
        return stress_results
    
    def _calculate_overall_risk_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate overall portfolio risk score (0-100).
        
        Args:
            metrics: Dictionary of risk metrics
            
        Returns:
            Risk score where higher values indicate higher risk
        """
        score = 0.0
        weights = {
            'concentration_weight': 0.25,
            'delinquency_weight': 0.30,
            'credit_quality_weight': 0.25,
            'loss_weight': 0.20
        }
        
        # Concentration risk component (0-25 points)
        hhi = metrics.get('hhi_index', 0)
        if hhi > 2500:  # Highly concentrated
            concentration_score = 25
        elif hhi > 1500:  # Moderately concentrated
            concentration_score = 15
        elif hhi > 1000:  # Low concentration
            concentration_score = 10
        else:  # Well diversified
            concentration_score = 5
        
        # Delinquency risk component (0-30 points)
        delinq_rate = metrics.get('current_delinquency_rate', 0)
        if delinq_rate > 3.0:  # High delinquency
            delinquency_score = 30
        elif delinq_rate > 2.0:  # Moderate delinquency
            delinquency_score = 20
        elif delinq_rate > 1.0:  # Low delinquency
            delinquency_score = 10
        else:  # Very low delinquency
            delinquency_score = 5
        
        # Credit quality component (0-25 points)
        avg_fico = metrics.get('weighted_avg_fico', 750)
        if avg_fico < 650:  # Poor credit quality
            credit_quality_score = 25
        elif avg_fico < 700:  # Fair credit quality
            credit_quality_score = 15
        elif avg_fico < 750:  # Good credit quality
            credit_quality_score = 10
        else:  # Excellent credit quality
            credit_quality_score = 5
        
        # Loss component (0-20 points)
        charge_off_rate = metrics.get('net_charge_off_rate', 0)
        if charge_off_rate > 1.0:  # High losses
            loss_score = 20
        elif charge_off_rate > 0.5:  # Moderate losses
            loss_score = 15
        elif charge_off_rate > 0.25:  # Low losses
            loss_score = 10
        else:  # Very low losses
            loss_score = 5
        
        # Calculate weighted score
        total_score = (
            concentration_score * weights['concentration_weight'] +
            delinquency_score * weights['delinquency_weight'] +
            credit_quality_score * weights['credit_quality_weight'] +
            loss_score * weights['loss_weight']
        )
        
        return min(100.0, max(0.0, total_score))
    
    def _get_risk_rating(self, risk_score: float) -> str:
        """Convert risk score to rating."""
        if risk_score >= 80:
            return "High Risk"
        elif risk_score >= 60:
            return "Moderate-High Risk"
        elif risk_score >= 40:
            return "Moderate Risk"
        elif risk_score >= 20:
            return "Low-Moderate Risk"
        else:
            return "Low Risk"
    
    def _identify_key_concerns(self, metrics: Dict[str, float]) -> List[str]:
        """Identify key risk concerns based on metrics."""
        concerns = []
        
        # Check concentration
        hhi = metrics.get('hhi_index', 0)
        if hhi > 2500:
            concerns.append("High portfolio concentration risk")
        
        # Check delinquency
        delinq_rate = metrics.get('current_delinquency_rate', 0)
        if delinq_rate > 2.0:
            concerns.append("Elevated delinquency rates")
        
        # Check credit quality
        avg_fico = metrics.get('weighted_avg_fico', 750)
        if avg_fico < 680:
            concerns.append("Below-average credit quality")
        
        # Check losses
        charge_off_rate = metrics.get('net_charge_off_rate', 0)
        if charge_off_rate > 0.75:
            concerns.append("High charge-off rates")
        
        if not concerns:
            concerns.append("No significant risk concerns identified")
        
        return concerns
    
    def _generate_risk_forecast(self, database: str, as_of_date: str) -> Dict[str, Any]:
        """Generate risk forecasts."""
        delinquency_data = self._get_delinquency_data(database, as_of_date)
        
        if not delinquency_data.empty:
            delinquency_forecast = self._forecast_delinquency(delinquency_data)
        else:
            delinquency_forecast = {'forecast': [], 'confidence_intervals': []}
        
        return {
            'delinquency_forecast': delinquency_forecast
        }
    
    def _calculate_concentration_breakdowns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate detailed concentration breakdowns."""
        breakdowns = {}
        
        if 'loan_type' in data.columns and 'current_balance' in data.columns:
            product_breakdown = data.groupby('loan_type')['current_balance'].sum()
            total_balance = data['current_balance'].sum()
            product_pct = (product_breakdown / total_balance * 100).to_dict()
            breakdowns['product_breakdown'] = product_pct
        
        if 'geographic_region' in data.columns and 'current_balance' in data.columns:
            geo_breakdown = data.groupby('geographic_region')['current_balance'].sum()
            total_balance = data['current_balance'].sum()
            geo_pct = (geo_breakdown / total_balance * 100).to_dict()
            breakdowns['geographic_breakdown'] = geo_pct
        
        return breakdowns
    
    def _generate_concentration_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Generate concentration risk recommendations."""
        recommendations = []
        
        hhi = metrics.get('hhi_index', 0)
        top_10_conc = metrics.get('top_10_concentration', 0)
        
        if hhi > 2500:
            recommendations.append("Portfolio is highly concentrated - consider diversification strategies")
        
        if top_10_conc > 15:
            recommendations.append("Top 10 borrowers represent significant concentration - monitor closely")
        
        if hhi < 1000:
            recommendations.append("Portfolio shows good diversification characteristics")
        
        return recommendations
    
    def _analyze_delinquency_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze delinquency trends in detail."""
        if data.empty:
            return {}
        
        # Calculate monthly changes
        data_sorted = data.sort_values('report_date')
        data_sorted['delinq_rate'] = (
            data_sorted['total_delinquent_amount'] / 
            data_sorted['total_loan_balance'] * 100
        )
        
        trends = {
            'monthly_rates': data_sorted['delinq_rate'].tolist(),
            'trend_direction': 'increasing' if len(data_sorted) > 1 and 
                             data_sorted['delinq_rate'].iloc[-1] > data_sorted['delinq_rate'].iloc[0]
                             else 'decreasing',
            'volatility': float(data_sorted['delinq_rate'].std()) if len(data_sorted) > 1 else 0.0
        }
        
        return trends
