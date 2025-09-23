"""
Operations Efficiency Agent for Credit Union Analytics

Provides comprehensive operational analysis including:
- Branch performance analysis and ranking
- Channel efficiency and attribution analysis
- Cost per member calculations and trending
- Process efficiency metrics and optimization
- Staff productivity analysis
- Service delivery optimization
- Digital transformation insights
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import math
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from .base_agent import BaseAgent, AnalysisContext, AnalysisResult


class OperationsAgent(BaseAgent):
    """
    Specialized agent for operational efficiency analysis.
    
    Provides advanced operational analytics including branch performance,
    channel efficiency, cost analysis, and productivity optimization.
    """
    
    def get_capabilities(self) -> List[str]:
        """Return list of analysis capabilities."""
        return [
            "branch_performance_analysis",
            "channel_efficiency_analysis", 
            "cost_per_member_calculation",
            "staff_productivity_analysis",
            "process_efficiency_metrics",
            "service_delivery_optimization",
            "digital_adoption_analysis",
            "capacity_utilization",
            "workflow_optimization",
            "resource_allocation",
            "benchmarking_analysis",
            "operational_forecasting"
        ]
    
    def analyze(self, context: AnalysisContext) -> AnalysisResult:
        """
        Main analysis method for operations efficiency.
        
        Args:
            context: Analysis context with parameters
            
        Returns:
            AnalysisResult with operational analysis results
        """
        self.log_analysis_start("operations", context)
        
        try:
            focus_area = context.parameters.get('focus_area', 'all')
            as_of_date = context.as_of_date or datetime.now().strftime('%Y-%m-%d')
            
            # Route to specific analysis based on type
            if focus_area == 'branch':
                result = self._analyze_branch_performance(context.database, as_of_date)
            elif focus_area == 'channel':
                result = self._analyze_channel_efficiency(context.database, as_of_date)
            elif focus_area == 'staff':
                result = self._analyze_staff_productivity(context.database, as_of_date)
            elif focus_area == 'all':
                result = self._comprehensive_operations_analysis(context.database, as_of_date)
            else:
                raise ValueError(f"Unknown focus area: {focus_area}")
            
            self.log_analysis_complete("operations", result)
            return result
            
        except Exception as e:
            return self.handle_analysis_error("operations", e)
    
    def _comprehensive_operations_analysis(self, database: str, as_of_date: str) -> AnalysisResult:
        """
        Perform comprehensive operational efficiency analysis.
        
        Args:
            database: Target database
            as_of_date: Analysis date
            
        Returns:
            AnalysisResult with comprehensive operational metrics
        """
        # Get operational data
        branch_data = self._get_branch_data(database, as_of_date)
        transaction_data = self._get_transaction_data(database, as_of_date)
        staff_data = self._get_staff_data(database, as_of_date)
        cost_data = self._get_cost_data(database, as_of_date)
        
        if branch_data.empty and staff_data.empty:
            return self.create_result(
                analysis_type="comprehensive",
                success=False,
                errors=["No operational data available for the specified date"]
            )
        
        # Perform all operational analyses
        branch_analysis = self._analyze_branch_performance_data(branch_data, transaction_data)
        channel_analysis = self._analyze_channel_performance(transaction_data)
        staff_analysis = self._analyze_staff_performance(staff_data, transaction_data)
        cost_analysis = self._analyze_cost_efficiency(cost_data, branch_data)
        
        # Calculate efficiency scores
        efficiency_scores = self._calculate_efficiency_scores({
            'branch': branch_analysis,
            'channel': channel_analysis,
            'staff': staff_analysis,
            'cost': cost_analysis
        })
        
        # Generate optimization recommendations
        recommendations = self._generate_optimization_recommendations(
            branch_analysis, channel_analysis, staff_analysis, cost_analysis
        )
        
        analysis_data = {
            'overall_performance': {
                'efficiency_score': efficiency_scores['overall'],
                'efficiency_rating': self._get_efficiency_rating(efficiency_scores['overall']),
                'key_strengths': self._identify_strengths(efficiency_scores),
                'improvement_areas': self._identify_improvement_areas(efficiency_scores)
            },
            'branch_performance': branch_analysis,
            'channel_performance': channel_analysis, 
            'staff_productivity': staff_analysis,
            'cost_efficiency': cost_analysis,
            'benchmarks': self._get_industry_benchmarks(),
            'recommendations': recommendations,
            'as_of_date': as_of_date
        }
        
        # Combine metrics from all areas
        all_metrics = {
            'overall_efficiency_score': efficiency_scores['overall'],
            **efficiency_scores
        }
        
        # Add area-specific metrics
        for area, area_data in [
            ('branch', branch_analysis),
            ('channel', channel_analysis),
            ('staff', staff_analysis),
            ('cost', cost_analysis)
        ]:
            if isinstance(area_data, dict) and 'metrics' in area_data:
                for key, value in area_data['metrics'].items():
                    all_metrics[f"{area}_{key}"] = value
        
        return self.create_result(
            analysis_type="comprehensive",
            data=analysis_data,
            metrics=all_metrics,
            metadata={'calculation_date': as_of_date}
        )
    
    def _analyze_branch_performance(self, database: str, as_of_date: str) -> AnalysisResult:
        """Analyze branch performance metrics."""
        branch_data = self._get_branch_data(database, as_of_date)
        transaction_data = self._get_transaction_data(database, as_of_date)
        
        if branch_data.empty:
            return self.create_result(
                analysis_type="branch",
                success=False,
                errors=["No branch data available"]
            )
        
        branch_analysis = self._analyze_branch_performance_data(branch_data, transaction_data)
        
        return self.create_result(
            analysis_type="branch",
            data=branch_analysis,
            metrics=branch_analysis.get('metrics', {})
        )
    
    def _analyze_channel_efficiency(self, database: str, as_of_date: str) -> AnalysisResult:
        """Analyze channel efficiency metrics."""
        transaction_data = self._get_transaction_data(database, as_of_date)
        
        if transaction_data.empty:
            return self.create_result(
                analysis_type="channel",
                success=False,
                errors=["No transaction data available for channel analysis"]
            )
        
        channel_analysis = self._analyze_channel_performance(transaction_data)
        
        return self.create_result(
            analysis_type="channel",
            data=channel_analysis,
            metrics=channel_analysis.get('metrics', {})
        )
    
    def _analyze_staff_productivity(self, database: str, as_of_date: str) -> AnalysisResult:
        """Analyze staff productivity metrics."""
        staff_data = self._get_staff_data(database, as_of_date)
        transaction_data = self._get_transaction_data(database, as_of_date)
        
        if staff_data.empty:
            return self.create_result(
                analysis_type="staff",
                success=False,
                errors=["No staff data available"]
            )
        
        staff_analysis = self._analyze_staff_performance(staff_data, transaction_data)
        
        return self.create_result(
            analysis_type="staff",
            data=staff_analysis,
            metrics=staff_analysis.get('metrics', {})
        )
    
    def _get_branch_data(self, database: str, as_of_date: str) -> pd.DataFrame:
        """
        Retrieve branch operational data.
        
        Args:
            database: Target database
            as_of_date: Analysis date
            
        Returns:
            DataFrame with branch data
        """
        query = """
        SELECT 
            b.branch_id,
            b.branch_name,
            b.branch_type,
            b.location,
            b.square_footage,
            b.opened_date,
            
            -- Member metrics
            COUNT(DISTINCT m.member_id) as total_members,
            COUNT(DISTINCT CASE WHEN m.join_date >= DATEADD(month, -12, ?) THEN m.member_id END) as new_members_12m,
            
            -- Account metrics
            COUNT(DISTINCT a.account_id) as total_accounts,
            SUM(a.current_balance) as total_deposits,
            
            -- Loan metrics
            COUNT(DISTINCT l.loan_id) as total_loans,
            SUM(l.current_balance) as total_loan_balance,
            
            -- Staffing
            COUNT(DISTINCT s.staff_id) as staff_count,
            SUM(s.annual_salary) as total_staff_cost,
            
            -- Operating expenses
            b.monthly_rent,
            b.monthly_utilities,
            b.monthly_other_expenses,
            
            -- Performance metrics
            b.monthly_fee_income,
            b.monthly_transactions,
            b.customer_satisfaction_score
            
        FROM branches b
        LEFT JOIN members m ON b.branch_id = m.primary_branch
        LEFT JOIN accounts a ON m.member_id = a.member_id AND a.status = 'Active'
        LEFT JOIN loans l ON m.member_id = l.member_id AND l.status IN ('Active', 'Current')
        LEFT JOIN staff s ON b.branch_id = s.branch_id AND s.status = 'Active'
        WHERE b.status = 'Active'
        GROUP BY b.branch_id, b.branch_name, b.branch_type, b.location, b.square_footage,
                 b.opened_date, b.monthly_rent, b.monthly_utilities, b.monthly_other_expenses,
                 b.monthly_fee_income, b.monthly_transactions, b.customer_satisfaction_score
        """
        
        try:
            result = self.execute_query(query, database, params={'as_of_date': as_of_date})
            return result
        except Exception as e:
            self.logger.warning(f"Could not retrieve branch data: {e}")
            return pd.DataFrame()
    
    def _get_transaction_data(self, database: str, as_of_date: str) -> pd.DataFrame:
        """Retrieve transaction data for channel analysis."""
        # Get last 3 months for channel analysis
        start_date = (datetime.strptime(as_of_date, '%Y-%m-%d') - timedelta(days=90)).strftime('%Y-%m-%d')
        
        query = """
        SELECT 
            t.transaction_id,
            t.member_id,
            t.transaction_date,
            t.transaction_type,
            t.amount,
            t.channel,
            t.branch_id,
            t.processing_time_seconds,
            t.staff_id,
            m.primary_branch
        FROM transactions t
        JOIN members m ON t.member_id = m.member_id
        WHERE t.transaction_date BETWEEN ? AND ?
        AND t.transaction_type NOT IN ('Transfer', 'Internal')
        """
        
        try:
            result = self.execute_query(
                query, 
                database, 
                params={'start_date': start_date, 'end_date': as_of_date}
            )
            return result
        except Exception as e:
            self.logger.warning(f"Could not retrieve transaction data: {e}")
            return pd.DataFrame()
    
    def _get_staff_data(self, database: str, as_of_date: str) -> pd.DataFrame:
        """Retrieve staff performance data."""
        query = """
        SELECT 
            s.staff_id,
            s.employee_id,
            s.first_name,
            s.last_name,
            s.position,
            s.department,
            s.branch_id,
            s.hire_date,
            s.annual_salary,
            s.performance_rating,
            
            -- Performance metrics
            s.monthly_transactions_processed,
            s.monthly_accounts_opened,
            s.monthly_loans_originated,
            s.customer_satisfaction_score,
            s.error_rate_pct,
            s.training_hours_completed,
            
            -- Attendance
            s.days_worked_last_month,
            s.days_available_last_month
            
        FROM staff s
        WHERE s.status = 'Active'
        AND s.as_of_date <= ?
        """
        
        try:
            result = self.execute_query(query, database, params={'as_of_date': as_of_date})
            return result
        except Exception as e:
            self.logger.warning(f"Could not retrieve staff data: {e}")
            return pd.DataFrame()
    
    def _get_cost_data(self, database: str, as_of_date: str) -> pd.DataFrame:
        """Retrieve cost and expense data."""
        query = """
        SELECT 
            cost_center,
            cost_category,
            monthly_amount,
            annual_budget,
            cost_per_member,
            cost_per_transaction,
            variance_from_budget_pct,
            report_date
        FROM operational_costs
        WHERE report_date <= ?
        ORDER BY report_date DESC
        """
        
        try:
            result = self.execute_query(query, database, params={'as_of_date': as_of_date})
            return result
        except Exception as e:
            self.logger.warning(f"Could not retrieve cost data: {e}")
            return pd.DataFrame()
    
    def _analyze_branch_performance_data(self, branch_data: pd.DataFrame, transaction_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze branch performance metrics."""
        if branch_data.empty:
            return {}
        
        # Calculate branch efficiency metrics
        branch_metrics = []
        
        for _, branch in branch_data.iterrows():
            branch_id = branch['branch_id']
            
            # Calculate key performance indicators
            members_per_staff = self.safe_divide(branch['total_members'], branch['staff_count'])
            deposits_per_staff = self.safe_divide(branch['total_deposits'], branch['staff_count'])
            transactions_per_staff = self.safe_divide(branch['monthly_transactions'], branch['staff_count'])
            
            # Calculate cost efficiency
            total_monthly_costs = (
                branch.get('monthly_rent', 0) + 
                branch.get('monthly_utilities', 0) + 
                branch.get('monthly_other_expenses', 0) +
                self.safe_divide(branch.get('total_staff_cost', 0), 12)  # Monthly staff cost
            )
            
            cost_per_member = self.safe_divide(total_monthly_costs, branch['total_members'])
            revenue_per_member = self.safe_divide(branch.get('monthly_fee_income', 0), branch['total_members'])
            
            # Calculate profitability
            net_income = branch.get('monthly_fee_income', 0) - total_monthly_costs
            roi = self.safe_divide(net_income, total_monthly_costs) * 100
            
            # Get branch-specific transaction data
            branch_transactions = transaction_data[transaction_data['branch_id'] == branch_id] if not transaction_data.empty else pd.DataFrame()
            
            # Calculate efficiency score
            efficiency_score = self._calculate_branch_efficiency_score({
                'members_per_staff': members_per_staff,
                'cost_per_member': cost_per_member,
                'revenue_per_member': revenue_per_member,
                'customer_satisfaction': branch.get('customer_satisfaction_score', 0),
                'roi': roi
            })
            
            branch_metrics.append({
                'branch_id': branch_id,
                'branch_name': branch['branch_name'],
                'branch_type': branch['branch_type'],
                'total_members': branch['total_members'],
                'staff_count': branch['staff_count'],
                'members_per_staff': members_per_staff,
                'deposits_per_staff': deposits_per_staff,
                'transactions_per_staff': transactions_per_staff,
                'cost_per_member': cost_per_member,
                'revenue_per_member': revenue_per_member,
                'monthly_net_income': net_income,
                'roi_percent': roi,
                'efficiency_score': efficiency_score,
                'customer_satisfaction': branch.get('customer_satisfaction_score', 0),
                'new_members_12m': branch.get('new_members_12m', 0)
            })
        
        # Create DataFrame for analysis
        branch_metrics_df = pd.DataFrame(branch_metrics)
        
        # Rank branches
        branch_rankings = self._rank_branches(branch_metrics_df)
        
        # Calculate summary statistics
        summary_stats = {
            'total_branches': len(branch_metrics_df),
            'avg_members_per_branch': branch_metrics_df['total_members'].mean(),
            'avg_staff_per_branch': branch_metrics_df['staff_count'].mean(),
            'avg_cost_per_member': branch_metrics_df['cost_per_member'].mean(),
            'avg_roi': branch_metrics_df['roi_percent'].mean(),
            'avg_efficiency_score': branch_metrics_df['efficiency_score'].mean()
        }
        
        return {
            'branch_performance': branch_metrics,
            'branch_rankings': branch_rankings,
            'summary_statistics': summary_stats,
            'top_performers': branch_metrics_df.nlargest(5, 'efficiency_score').to_dict('records'),
            'improvement_opportunities': branch_metrics_df.nsmallest(5, 'efficiency_score').to_dict('records'),
            'metrics': summary_stats
        }
    
    def _analyze_channel_performance(self, transaction_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze channel efficiency and performance."""
        if transaction_data.empty:
            return {}
        
        # Analyze by channel
        channel_analysis = transaction_data.groupby('channel').agg({
            'transaction_id': 'count',
            'amount': ['sum', 'mean', 'median'],
            'processing_time_seconds': ['mean', 'median', 'std']
        }).round(2)
        
        channel_analysis.columns = [
            'transaction_count', 'total_volume', 'avg_amount', 'median_amount',
            'avg_processing_time', 'median_processing_time', 'processing_time_std'
        ]
        
        channel_analysis = channel_analysis.reset_index()
        
        # Calculate channel efficiency metrics
        total_transactions = transaction_data['transaction_id'].count()
        total_volume = transaction_data['amount'].sum()
        
        channel_metrics = []
        for _, channel in channel_analysis.iterrows():
            channel_name = channel['channel']
            
            # Market share
            transaction_share = (channel['transaction_count'] / total_transactions) * 100
            volume_share = (channel['total_volume'] / total_volume) * 100
            
            # Efficiency metrics
            cost_per_transaction = self._estimate_channel_cost(channel_name)
            efficiency_score = self._calculate_channel_efficiency_score(channel, cost_per_transaction)
            
            channel_metrics.append({
                'channel': channel_name,
                'transaction_count': channel['transaction_count'],
                'transaction_share_pct': transaction_share,
                'total_volume': channel['total_volume'],
                'volume_share_pct': volume_share,
                'avg_transaction_amount': channel['avg_amount'],
                'avg_processing_time': channel['avg_processing_time'],
                'cost_per_transaction': cost_per_transaction,
                'efficiency_score': efficiency_score
            })
        
        # Sort by efficiency score
        channel_metrics.sort(key=lambda x: x['efficiency_score'], reverse=True)
        
        # Calculate digital adoption metrics
        digital_channels = ['Online Banking', 'Mobile Banking', 'ATM']
        digital_transactions = transaction_data[transaction_data['channel'].isin(digital_channels)]
        digital_adoption_rate = (len(digital_transactions) / len(transaction_data)) * 100
        
        return {
            'channel_performance': channel_metrics,
            'digital_adoption_rate': digital_adoption_rate,
            'most_efficient_channel': channel_metrics[0]['channel'] if channel_metrics else None,
            'least_efficient_channel': channel_metrics[-1]['channel'] if channel_metrics else None,
            'total_transactions_analyzed': total_transactions,
            'recommendations': self._generate_channel_recommendations(channel_metrics),
            'metrics': {
                'digital_adoption_rate': digital_adoption_rate,
                'avg_processing_time': transaction_data['processing_time_seconds'].mean(),
                'total_channels': len(channel_metrics)
            }
        }
    
    def _analyze_staff_performance(self, staff_data: pd.DataFrame, transaction_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze staff productivity and performance."""
        if staff_data.empty:
            return {}
        
        # Calculate staff productivity metrics
        staff_metrics = []
        
        for _, staff in staff_data.iterrows():
            staff_id = staff['staff_id']
            
            # Get staff transaction data
            staff_transactions = transaction_data[transaction_data['staff_id'] == staff_id] if not transaction_data.empty else pd.DataFrame()
            
            # Calculate productivity metrics
            transactions_per_day = self.safe_divide(
                staff.get('monthly_transactions_processed', 0),
                staff.get('days_worked_last_month', 1)
            )
            
            accounts_per_day = self.safe_divide(
                staff.get('monthly_accounts_opened', 0),
                staff.get('days_worked_last_month', 1)
            )
            
            loans_per_day = self.safe_divide(
                staff.get('monthly_loans_originated', 0),
                staff.get('days_worked_last_month', 1)
            )
            
            # Calculate efficiency metrics
            salary_per_transaction = self.safe_divide(
                self.safe_divide(staff.get('annual_salary', 0), 12),  # Monthly salary
                staff.get('monthly_transactions_processed', 1)
            )
            
            # Calculate overall productivity score
            productivity_score = self._calculate_staff_productivity_score({
                'transactions_per_day': transactions_per_day,
                'accounts_per_day': accounts_per_day,
                'loans_per_day': loans_per_day,
                'customer_satisfaction': staff.get('customer_satisfaction_score', 0),
                'error_rate': staff.get('error_rate_pct', 0),
                'attendance_rate': self.safe_divide(
                    staff.get('days_worked_last_month', 0),
                    staff.get('days_available_last_month', 1)
                ) * 100
            })
            
            staff_metrics.append({
                'staff_id': staff_id,
                'employee_id': staff['employee_id'],
                'name': f"{staff['first_name']} {staff['last_name']}",
                'position': staff['position'],
                'department': staff['department'],
                'branch_id': staff['branch_id'],
                'transactions_per_day': transactions_per_day,
                'accounts_per_day': accounts_per_day,
                'loans_per_day': loans_per_day,
                'salary_per_transaction': salary_per_transaction,
                'customer_satisfaction': staff.get('customer_satisfaction_score', 0),
                'error_rate_pct': staff.get('error_rate_pct', 0),
                'productivity_score': productivity_score,
                'annual_salary': staff.get('annual_salary', 0),
                'performance_rating': staff.get('performance_rating', 0)
            })
        
        # Create DataFrame for analysis
        staff_metrics_df = pd.DataFrame(staff_metrics)
        
        # Calculate department-level metrics
        dept_metrics = staff_metrics_df.groupby('department').agg({
            'productivity_score': 'mean',
            'customer_satisfaction': 'mean',
            'error_rate_pct': 'mean',
            'annual_salary': 'mean',
            'staff_id': 'count'
        }).round(2)
        
        dept_metrics.columns = ['avg_productivity', 'avg_satisfaction', 'avg_error_rate', 'avg_salary', 'staff_count']
        dept_metrics = dept_metrics.reset_index()
        
        # Identify top and bottom performers
        top_performers = staff_metrics_df.nlargest(10, 'productivity_score').to_dict('records')
        improvement_candidates = staff_metrics_df.nsmallest(10, 'productivity_score').to_dict('records')
        
        return {
            'staff_performance': staff_metrics,
            'department_metrics': dept_metrics.to_dict('records'),
            'top_performers': top_performers,
            'improvement_candidates': improvement_candidates,
            'total_staff': len(staff_metrics_df),
            'avg_productivity_score': staff_metrics_df['productivity_score'].mean(),
            'recommendations': self._generate_staff_recommendations(staff_metrics_df),
            'metrics': {
                'avg_productivity_score': staff_metrics_df['productivity_score'].mean(),
                'avg_customer_satisfaction': staff_metrics_df['customer_satisfaction'].mean(),
                'avg_error_rate': staff_metrics_df['error_rate_pct'].mean(),
                'total_staff': len(staff_metrics_df)
            }
        }
    
    def _analyze_cost_efficiency(self, cost_data: pd.DataFrame, branch_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze cost efficiency and budget performance."""
        if cost_data.empty:
            return {}
        
        # Get latest cost data
        latest_costs = cost_data.head(1) if not cost_data.empty else pd.DataFrame()
        
        if latest_costs.empty:
            return {}
        
        # Analyze cost categories
        cost_analysis = cost_data.groupby('cost_category').agg({
            'monthly_amount': 'sum',
            'annual_budget': 'sum',
            'variance_from_budget_pct': 'mean'
        }).round(2)
        
        cost_analysis['budget_utilization_pct'] = (
            cost_analysis['monthly_amount'] * 12 / cost_analysis['annual_budget'] * 100
        ).round(2)
        
        cost_analysis = cost_analysis.reset_index()
        
        # Calculate total member count for cost per member calculations
        total_members = branch_data['total_members'].sum() if not branch_data.empty else 1
        
        # Calculate cost efficiency metrics
        total_monthly_costs = cost_data['monthly_amount'].sum()
        total_annual_budget = cost_data['annual_budget'].sum()
        cost_per_member_monthly = total_monthly_costs / total_members
        cost_per_member_annual = cost_per_member_monthly * 12
        
        # Identify cost optimization opportunities
        over_budget_categories = cost_analysis[cost_analysis['variance_from_budget_pct'] > 10]
        under_budget_categories = cost_analysis[cost_analysis['variance_from_budget_pct'] < -10]
        
        return {
            'cost_by_category': cost_analysis.to_dict('records'),
            'total_monthly_costs': total_monthly_costs,
            'total_annual_budget': total_annual_budget,
            'cost_per_member_monthly': cost_per_member_monthly,
            'cost_per_member_annual': cost_per_member_annual,
            'budget_variance_pct': cost_data['variance_from_budget_pct'].mean(),
            'over_budget_categories': over_budget_categories.to_dict('records'),
            'under_budget_categories': under_budget_categories.to_dict('records'),
            'cost_optimization_opportunities': self._identify_cost_optimization_opportunities(cost_analysis),
            'metrics': {
                'total_monthly_costs': total_monthly_costs,
                'cost_per_member_monthly': cost_per_member_monthly,
                'budget_variance_pct': cost_data['variance_from_budget_pct'].mean(),
                'over_budget_categories_count': len(over_budget_categories)
            }
        }
    
    def _calculate_branch_efficiency_score(self, metrics: Dict[str, float]) -> float:
        """Calculate branch efficiency score (0-100)."""
        score = 0
        weights = {
            'productivity': 0.25,    # Members per staff
            'cost_efficiency': 0.30, # Cost per member
            'profitability': 0.25,   # ROI
            'satisfaction': 0.20     # Customer satisfaction
        }
        
        # Productivity score (members per staff)
        members_per_staff = metrics.get('members_per_staff', 0)
        if members_per_staff >= 500:
            productivity_score = 100
        elif members_per_staff >= 400:
            productivity_score = 80
        elif members_per_staff >= 300:
            productivity_score = 60
        elif members_per_staff >= 200:
            productivity_score = 40
        else:
            productivity_score = 20
        
        # Cost efficiency score (lower cost per member = higher score)
        cost_per_member = metrics.get('cost_per_member', 1000)
        if cost_per_member <= 300:
            cost_score = 100
        elif cost_per_member <= 400:
            cost_score = 80
        elif cost_per_member <= 500:
            cost_score = 60
        elif cost_per_member <= 600:
            cost_score = 40
        else:
            cost_score = 20
        
        # Profitability score
        roi = metrics.get('roi', 0)
        if roi >= 20:
            profit_score = 100
        elif roi >= 15:
            profit_score = 80
        elif roi >= 10:
            profit_score = 60
        elif roi >= 5:
            profit_score = 40
        else:
            profit_score = 20
        
        # Customer satisfaction score
        satisfaction = metrics.get('customer_satisfaction', 0)
        satisfaction_score = min(100, satisfaction * 20)  # Assuming 5-point scale
        
        # Calculate weighted score
        total_score = (
            productivity_score * weights['productivity'] +
            cost_score * weights['cost_efficiency'] +
            profit_score * weights['profitability'] +
            satisfaction_score * weights['satisfaction']
        )
        
        return min(100, max(0, total_score))
    
    def _calculate_channel_efficiency_score(self, channel_data: pd.Series, cost_per_transaction: float) -> float:
        """Calculate channel efficiency score."""
        # Lower processing time and cost = higher efficiency
        processing_time = channel_data.get('avg_processing_time', 300)  # seconds
        
        # Time efficiency (0-50 points)
        if processing_time <= 30:
            time_score = 50
        elif processing_time <= 60:
            time_score = 40
        elif processing_time <= 120:
            time_score = 30
        elif processing_time <= 300:
            time_score = 20
        else:
            time_score = 10
        
        # Cost efficiency (0-50 points)
        if cost_per_transaction <= 1.0:
            cost_score = 50
        elif cost_per_transaction <= 2.0:
            cost_score = 40
        elif cost_per_transaction <= 5.0:
            cost_score = 30
        elif cost_per_transaction <= 10.0:
            cost_score = 20
        else:
            cost_score = 10
        
        return time_score + cost_score
    
    def _calculate_staff_productivity_score(self, metrics: Dict[str, float]) -> float:
        """Calculate staff productivity score (0-100)."""
        score = 0
        weights = {
            'transaction_volume': 0.25,
            'account_opening': 0.20,
            'loan_origination': 0.20,
            'customer_satisfaction': 0.20,
            'quality': 0.10,
            'attendance': 0.05
        }
        
        # Transaction volume score
        transactions_per_day = metrics.get('transactions_per_day', 0)
        if transactions_per_day >= 50:
            trans_score = 100
        elif transactions_per_day >= 40:
            trans_score = 80
        elif transactions_per_day >= 30:
            trans_score = 60
        elif transactions_per_day >= 20:
            trans_score = 40
        else:
            trans_score = 20
        
        # Account opening score
        accounts_per_day = metrics.get('accounts_per_day', 0)
        if accounts_per_day >= 3:
            account_score = 100
        elif accounts_per_day >= 2:
            account_score = 80
        elif accounts_per_day >= 1:
            account_score = 60
        else:
            account_score = 40
        
        # Loan origination score
        loans_per_day = metrics.get('loans_per_day', 0)
        if loans_per_day >= 2:
            loan_score = 100
        elif loans_per_day >= 1:
            loan_score = 80
        elif loans_per_day >= 0.5:
            loan_score = 60
        else:
            loan_score = 40
        
        # Customer satisfaction score
        satisfaction = metrics.get('customer_satisfaction', 0)
        satisfaction_score = min(100, satisfaction * 20)  # Assuming 5-point scale
        
        # Quality score (lower error rate = higher score)
        error_rate = metrics.get('error_rate', 5)
        if error_rate <= 1:
            quality_score = 100
        elif error_rate <= 2:
            quality_score = 80
        elif error_rate <= 3:
            quality_score = 60
        elif error_rate <= 5:
            quality_score = 40
        else:
            quality_score = 20
        
        # Attendance score
        attendance_rate = metrics.get('attendance_rate', 80)
        if attendance_rate >= 95:
            attendance_score = 100
        elif attendance_rate >= 90:
            attendance_score = 80
        elif attendance_rate >= 85:
            attendance_score = 60
        else:
            attendance_score = 40
        
        # Calculate weighted score
        total_score = (
            trans_score * weights['transaction_volume'] +
            account_score * weights['account_opening'] +
            loan_score * weights['loan_origination'] +
            satisfaction_score * weights['customer_satisfaction'] +
            quality_score * weights['quality'] +
            attendance_score * weights['attendance']
        )
        
        return min(100, max(0, total_score))
    
    def _estimate_channel_cost(self, channel: str) -> float:
        """Estimate cost per transaction by channel."""
        cost_estimates = {
            'ATM': 0.50,
            'Online Banking': 0.25,
            'Mobile Banking': 0.20,
            'Phone Banking': 3.00,
            'Branch Teller': 5.00,
            'Drive-Through': 4.00,
            'Mail': 2.00
        }
        
        return cost_estimates.get(channel, 3.00)  # Default cost
    
    def _rank_branches(self, branch_metrics_df: pd.DataFrame) -> Dict[str, Any]:
        """Rank branches by various performance metrics."""
        if branch_metrics_df.empty:
            return {}
        
        rankings = {
            'by_efficiency_score': branch_metrics_df.nlargest(10, 'efficiency_score')[
                ['branch_name', 'efficiency_score']
            ].to_dict('records'),
            'by_roi': branch_metrics_df.nlargest(10, 'roi_percent')[
                ['branch_name', 'roi_percent']
            ].to_dict('records'),
            'by_member_growth': branch_metrics_df.nlargest(10, 'new_members_12m')[
                ['branch_name', 'new_members_12m']
            ].to_dict('records'),
            'by_cost_efficiency': branch_metrics_df.nsmallest(10, 'cost_per_member')[
                ['branch_name', 'cost_per_member']
            ].to_dict('records')
        }
        
        return rankings
    
    def _calculate_efficiency_scores(self, analysis_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate efficiency scores for each operational area."""
        scores = {}
        
        # Branch efficiency
        branch_data = analysis_data.get('branch', {})
        if 'summary_statistics' in branch_data:
            scores['branch'] = branch_data['summary_statistics'].get('avg_efficiency_score', 0)
        else:
            scores['branch'] = 0
        
        # Channel efficiency
        channel_data = analysis_data.get('channel', {})
        if 'channel_performance' in channel_data:
            channel_scores = [ch.get('efficiency_score', 0) for ch in channel_data['channel_performance']]
            scores['channel'] = np.mean(channel_scores) if channel_scores else 0
        else:
            scores['channel'] = 0
        
        # Staff efficiency
        staff_data = analysis_data.get('staff', {})
        scores['staff'] = staff_data.get('avg_productivity_score', 0)
        
        # Cost efficiency (inverse of cost growth)
        cost_data = analysis_data.get('cost', {})
        budget_variance = cost_data.get('budget_variance_pct', 0)
        scores['cost'] = max(0, 100 - abs(budget_variance))  # Penalize budget overruns
        
        # Overall efficiency (weighted average)
        weights = {'branch': 0.30, 'channel': 0.25, 'staff': 0.25, 'cost': 0.20}
        scores['overall'] = sum(scores[area] * weights[area] for area in weights.keys())
        
        return scores
    
    def _get_efficiency_rating(self, efficiency_score: float) -> str:
        """Convert efficiency score to rating."""
        if efficiency_score >= 90:
            return "Excellent"
        elif efficiency_score >= 80:
            return "Good"
        elif efficiency_score >= 70:
            return "Satisfactory"
        elif efficiency_score >= 60:
            return "Needs Improvement"
        else:
            return "Poor"
    
    def _identify_strengths(self, efficiency_scores: Dict[str, float]) -> List[str]:
        """Identify operational strengths based on efficiency scores."""
        strengths = []
        
        for area, score in efficiency_scores.items():
            if area != 'overall' and score >= 80:
                area_name = area.replace('_', ' ').title()
                strengths.append(f"Strong {area_name} Performance (Score: {score:.1f})")
        
        return strengths if strengths else ["Opportunities for improvement across all areas"]
    
    def _identify_improvement_areas(self, efficiency_scores: Dict[str, float]) -> List[str]:
        """Identify areas needing improvement based on efficiency scores."""
        improvements = []
        
        for area, score in efficiency_scores.items():
            if area != 'overall' and score < 70:
                area_name = area.replace('_', ' ').title()
                improvements.append(f"{area_name} Performance Below Target (Score: {score:.1f})")
        
        return improvements if improvements else ["All areas performing at acceptable levels"]
    
    def _get_industry_benchmarks(self) -> Dict[str, Any]:
        """Get industry benchmarks for operational metrics."""
        return {
            'branch_metrics': {
                'members_per_staff': 450,
                'cost_per_member_monthly': 375,
                'roi_percent': 15,
                'customer_satisfaction': 4.2
            },
            'channel_metrics': {
                'digital_adoption_rate': 75,
                'avg_processing_time_seconds': 120,
                'cost_per_transaction': 2.50
            },
            'staff_metrics': {
                'productivity_score': 75,
                'customer_satisfaction': 4.0,
                'error_rate_pct': 2.5
            },
            'cost_metrics': {
                'cost_per_member_annual': 450,
                'budget_variance_tolerance': 5
            }
        }
    
    def _generate_optimization_recommendations(self, branch_analysis: Dict, channel_analysis: Dict, 
                                            staff_analysis: Dict, cost_analysis: Dict) -> List[str]:
        """Generate optimization recommendations based on analysis results."""
        recommendations = []
        
        # Branch recommendations
        if branch_analysis and 'summary_statistics' in branch_analysis:
            avg_efficiency = branch_analysis['summary_statistics'].get('avg_efficiency_score', 0)
            if avg_efficiency < 70:
                recommendations.append("Consider branch consolidation or efficiency improvements for underperforming locations")
        
        # Channel recommendations  
        if channel_analysis and 'digital_adoption_rate' in channel_analysis:
            digital_rate = channel_analysis['digital_adoption_rate']
            if digital_rate < 60:
                recommendations.append("Increase digital channel adoption through member education and incentives")
        
        # Staff recommendations
        if staff_analysis and 'avg_productivity_score' in staff_analysis:
            avg_productivity = staff_analysis['avg_productivity_score']
            if avg_productivity < 70:
                recommendations.append("Implement staff training programs to improve productivity and performance")
        
        # Cost recommendations
        if cost_analysis and 'budget_variance_pct' in cost_analysis:
            variance = cost_analysis['budget_variance_pct']
            if abs(variance) > 10:
                recommendations.append("Review and adjust budget allocation to better align with actual operational needs")
        
        if not recommendations:
            recommendations.append("Continue monitoring operational metrics and maintain current performance levels")
        
        return recommendations
    
    def _generate_channel_recommendations(self, channel_metrics: List[Dict]) -> List[str]:
        """Generate channel-specific recommendations."""
        recommendations = []
        
        if not channel_metrics:
            return recommendations
        
        # Find least efficient channel
        least_efficient = min(channel_metrics, key=lambda x: x['efficiency_score'])
        most_efficient = max(channel_metrics, key=lambda x: x['efficiency_score'])
        
        if least_efficient['efficiency_score'] < 50:
            recommendations.append(f"Consider improvements to {least_efficient['channel']} channel efficiency")
        
        recommendations.append(f"Promote {most_efficient['channel']} as the most efficient channel")
        
        # Check for low digital adoption
        digital_channels = ['Online Banking', 'Mobile Banking']
        digital_share = sum(ch['transaction_share_pct'] for ch in channel_metrics 
                          if ch['channel'] in digital_channels)
        
        if digital_share < 60:
            recommendations.append("Focus on increasing digital channel adoption")
        
        return recommendations
    
    def _generate_staff_recommendations(self, staff_metrics_df: pd.DataFrame) -> List[str]:
        """Generate staff-specific recommendations."""
        recommendations = []
        
        if staff_metrics_df.empty:
            return recommendations
        
        # Identify training needs
        low_performers = staff_metrics_df[staff_metrics_df['productivity_score'] < 60]
        if len(low_performers) > 0:
            recommendations.append(f"Provide additional training for {len(low_performers)} staff members with low productivity scores")
        
        # Check error rates
        high_error_staff = staff_metrics_df[staff_metrics_df['error_rate_pct'] > 5]
        if len(high_error_staff) > 0:
            recommendations.append("Implement quality improvement programs for staff with high error rates")
        
        # Customer satisfaction
        low_satisfaction_staff = staff_metrics_df[staff_metrics_df['customer_satisfaction'] < 3.5]
        if len(low_satisfaction_staff) > 0:
            recommendations.append("Focus on customer service training for staff with low satisfaction scores")
        
        return recommendations
    
    def _identify_cost_optimization_opportunities(self, cost_analysis: pd.DataFrame) -> List[str]:
        """Identify cost optimization opportunities."""
        opportunities = []
        
        if cost_analysis.empty:
            return opportunities
        
        # Find categories over budget
        over_budget = cost_analysis[cost_analysis['variance_from_budget_pct'] > 10]
        for _, category in over_budget.iterrows():
            opportunities.append(f"Reduce spending in {category['cost_category']} (over budget by {category['variance_from_budget_pct']:.1f}%)")
        
        # Find high-cost categories
        high_cost = cost_analysis.nlargest(3, 'monthly_amount')
        for _, category in high_cost.iterrows():
            if category['variance_from_budget_pct'] > 5:
                opportunities.append(f"Review {category['cost_category']} expenses for potential savings")
        
        return opportunities
