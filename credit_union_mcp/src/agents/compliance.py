"""
Compliance Agent for Credit Union Analytics

Provides comprehensive regulatory compliance monitoring including:
- BSA/AML anomaly detection and reporting
- Regulatory ratio monitoring (NCUA requirements)
- CECL (Current Expected Credit Loss) calculations
- Capital adequacy monitoring
- Member Business Loan limit enforcement
- Regulatory reporting assistance
- Suspicious activity pattern detection
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


class ComplianceAgent(BaseAgent):
    """
    Specialized agent for regulatory compliance monitoring.
    
    Monitors compliance with NCUA regulations, BSA/AML requirements,
    and other regulatory obligations for credit unions.
    """
    
    def get_capabilities(self) -> List[str]:
        """Return list of analysis capabilities."""
        return [
            "capital_adequacy_monitoring",
            "bsa_aml_monitoring",
            "cecl_calculation",
            "mbl_limit_monitoring",
            "regulatory_ratio_analysis",
            "suspicious_activity_detection",
            "large_transaction_monitoring",
            "cash_transaction_reporting",
            "regulatory_reporting",
            "compliance_dashboard",
            "risk_assessment",
            "audit_trail_analysis"
        ]
    
    def analyze(self, context: AnalysisContext) -> AnalysisResult:
        """
        Main analysis method for compliance monitoring.
        
        Args:
            context: Analysis context with parameters
            
        Returns:
            AnalysisResult with compliance analysis results
        """
        self.log_analysis_start("compliance", context)
        
        try:
            check_type = context.parameters.get('check_type', 'all')
            as_of_date = context.as_of_date or datetime.now().strftime('%Y-%m-%d')
            
            # Route to specific analysis based on type
            if check_type == 'capital':
                result = self._check_capital_adequacy(context.database, as_of_date)
            elif check_type == 'bsa_aml':
                result = self._check_bsa_aml_compliance(context.database, as_of_date)
            elif check_type == 'lending_limits':
                result = self._check_lending_limits(context.database, as_of_date)
            elif check_type == 'cecl':
                result = self._calculate_cecl(context.database, as_of_date)
            elif check_type == 'all':
                result = self._comprehensive_compliance_check(context.database, as_of_date)
            else:
                raise ValueError(f"Unknown check type: {check_type}")
            
            self.log_analysis_complete("compliance", result)
            return result
            
        except Exception as e:
            return self.handle_analysis_error("compliance", e)
    
    def _comprehensive_compliance_check(self, database: str, as_of_date: str) -> AnalysisResult:
        """
        Perform comprehensive compliance analysis.
        
        Args:
            database: Target database
            as_of_date: Analysis date
            
        Returns:
            AnalysisResult with comprehensive compliance status
        """
        # Get required data
        financial_data = self._get_financial_data(database, as_of_date)
        transaction_data = self._get_transaction_data(database, as_of_date)
        loan_data = self._get_loan_data(database, as_of_date)
        
        if financial_data.empty:
            return self.create_result(
                analysis_type="comprehensive",
                success=False,
                errors=["No financial data available for compliance analysis"]
            )
        
        # Perform all compliance checks
        capital_status = self._analyze_capital_adequacy(financial_data)
        bsa_aml_status = self._analyze_bsa_aml_compliance(transaction_data)
        lending_limits_status = self._analyze_lending_limits(financial_data, loan_data)
        cecl_status = self._analyze_cecl_requirements(loan_data)
        
        # Calculate overall compliance score
        compliance_score = self._calculate_compliance_score({
            'capital': capital_status,
            'bsa_aml': bsa_aml_status,
            'lending_limits': lending_limits_status,
            'cecl': cecl_status
        })
        
        # Identify critical issues
        critical_issues = self._identify_critical_issues({
            'capital': capital_status,
            'bsa_aml': bsa_aml_status, 
            'lending_limits': lending_limits_status,
            'cecl': cecl_status
        })
        
        analysis_data = {
            'overall_status': {
                'compliance_score': compliance_score,
                'overall_rating': self._get_compliance_rating(compliance_score),
                'critical_issues_count': len(critical_issues),
                'requires_immediate_attention': len(critical_issues) > 0
            },
            'capital_adequacy': capital_status,
            'bsa_aml': bsa_aml_status,
            'lending_limits': lending_limits_status,
            'cecl': cecl_status,
            'critical_issues': critical_issues,
            'recommendations': self._generate_compliance_recommendations(critical_issues),
            'as_of_date': as_of_date
        }
        
        # Combine metrics from all areas
        all_metrics = {
            'compliance_score': compliance_score,
            'critical_issues_count': len(critical_issues)
        }
        
        # Add area-specific metrics
        for area, area_data in [
            ('capital', capital_status),
            ('bsa_aml', bsa_aml_status),
            ('lending_limits', lending_limits_status),
            ('cecl', cecl_status)
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
    
    def _check_capital_adequacy(self, database: str, as_of_date: str) -> AnalysisResult:
        """Check capital adequacy requirements."""
        financial_data = self._get_financial_data(database, as_of_date)
        
        if financial_data.empty:
            return self.create_result(
                analysis_type="capital",
                success=False,
                errors=["No financial data available for capital analysis"]
            )
        
        capital_status = self._analyze_capital_adequacy(financial_data)
        
        return self.create_result(
            analysis_type="capital",
            data=capital_status,
            metrics=capital_status.get('metrics', {})
        )
    
    def _check_bsa_aml_compliance(self, database: str, as_of_date: str) -> AnalysisResult:
        """Check BSA/AML compliance."""
        transaction_data = self._get_transaction_data(database, as_of_date)
        
        bsa_aml_status = self._analyze_bsa_aml_compliance(transaction_data)
        
        return self.create_result(
            analysis_type="bsa_aml",
            data=bsa_aml_status,
            metrics=bsa_aml_status.get('metrics', {})
        )
    
    def _check_lending_limits(self, database: str, as_of_date: str) -> AnalysisResult:
        """Check lending limit compliance."""
        financial_data = self._get_financial_data(database, as_of_date)
        loan_data = self._get_loan_data(database, as_of_date)
        
        if financial_data.empty:
            return self.create_result(
                analysis_type="lending_limits",
                success=False,
                errors=["No financial data available for lending limits analysis"]
            )
        
        lending_status = self._analyze_lending_limits(financial_data, loan_data)
        
        return self.create_result(
            analysis_type="lending_limits",
            data=lending_status,
            metrics=lending_status.get('metrics', {})
        )
    
    def _calculate_cecl(self, database: str, as_of_date: str) -> AnalysisResult:
        """Calculate CECL provisions."""
        loan_data = self._get_loan_data(database, as_of_date)
        
        if loan_data.empty:
            return self.create_result(
                analysis_type="cecl",
                success=False,
                errors=["No loan data available for CECL calculation"]
            )
        
        cecl_status = self._analyze_cecl_requirements(loan_data)
        
        return self.create_result(
            analysis_type="cecl",
            data=cecl_status,
            metrics=cecl_status.get('metrics', {})
        )
    
    def _get_financial_data(self, database: str, as_of_date: str) -> pd.DataFrame:
        """Retrieve financial data for compliance analysis."""
        query = """
        SELECT 
            total_assets,
            total_equity,
            total_liabilities,
            net_worth,
            retained_earnings,
            undivided_earnings,
            member_deposits,
            borrowed_funds,
            investments,
            cash_equivalents,
            
            -- Income statement items
            net_income,
            interest_income,
            interest_expense,
            fee_income,
            operating_expenses,
            provision_for_losses,
            
            -- Member business loans
            mbl_outstanding,
            mbl_limit,
            
            -- Risk-weighted assets (if available)
            risk_weighted_assets,
            
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
    
    def _get_transaction_data(self, database: str, as_of_date: str) -> pd.DataFrame:
        """Retrieve transaction data for BSA/AML analysis."""
        # Get last 30 days for monitoring
        start_date = (datetime.strptime(as_of_date, '%Y-%m-%d') - timedelta(days=30)).strftime('%Y-%m-%d')
        
        query = """
        SELECT 
            t.transaction_id,
            t.member_id,
            t.transaction_date,
            t.transaction_type,
            t.amount,
            t.description,
            t.channel,
            t.location,
            t.foreign_transaction,
            m.member_risk_rating,
            m.pep_status,
            m.country_of_citizenship
        FROM transactions t
        JOIN members m ON t.member_id = m.member_id
        WHERE t.transaction_date BETWEEN ? AND ?
        AND t.amount > 1000  -- Focus on larger transactions
        ORDER BY t.amount DESC, t.transaction_date DESC
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
    
    def _get_loan_data(self, database: str, as_of_date: str) -> pd.DataFrame:
        """Retrieve loan data for CECL and lending limits analysis."""
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
            charge_off_amount,
            recovery_amount,
            provision_amount,
            probability_of_default,
            loss_given_default,
            expected_life,
            loan_purpose,
            secured_unsecured
        FROM loan_portfolio
        WHERE as_of_date <= ?
        AND loan_status IN ('Active', 'Current', 'Delinquent')
        """
        
        try:
            result = self.execute_query(query, database, params={'as_of_date': as_of_date})
            return result
        except Exception as e:
            self.logger.warning(f"Could not retrieve loan data: {e}")
            return pd.DataFrame()
    
    def _analyze_capital_adequacy(self, financial_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze capital adequacy requirements."""
        if financial_data.empty:
            return {'status': 'No Data', 'compliant': False}
        
        row = financial_data.iloc[0]
        
        total_assets = float(row.get('total_assets', 0))
        total_equity = float(row.get('total_equity', 0))
        net_worth = float(row.get('net_worth', total_equity))
        risk_weighted_assets = float(row.get('risk_weighted_assets', total_assets))
        
        # Calculate capital ratios
        net_worth_ratio = self.safe_divide(net_worth, total_assets) * 100
        leverage_ratio = self.safe_divide(total_equity, total_assets) * 100
        risk_based_capital_ratio = self.safe_divide(total_equity, risk_weighted_assets) * 100
        
        # NCUA capital adequacy classifications
        capital_classification = self._classify_capital_adequacy(net_worth_ratio)
        
        # Check compliance
        is_compliant = net_worth_ratio >= 7.0  # NCUA minimum
        
        return {
            'status': capital_classification['classification'],
            'compliant': is_compliant,
            'ratios': {
                'net_worth_ratio': net_worth_ratio,
                'leverage_ratio': leverage_ratio,
                'risk_based_capital_ratio': risk_based_capital_ratio
            },
            'thresholds': {
                'minimum_net_worth': 7.0,
                'well_capitalized': 10.0,
                'adequately_capitalized': 8.0,
                'undercapitalized': 6.0
            },
            'classification': capital_classification,
            'metrics': {
                'net_worth_ratio': net_worth_ratio,
                'leverage_ratio': leverage_ratio,
                'risk_based_capital_ratio': risk_based_capital_ratio,
                'compliance_status': 1 if is_compliant else 0
            }
        }
    
    def _classify_capital_adequacy(self, net_worth_ratio: float) -> Dict[str, Any]:
        """Classify capital adequacy based on NCUA standards."""
        if net_worth_ratio >= 10.0:
            classification = "Well Capitalized"
            action_required = "None"
            risk_level = "Low"
        elif net_worth_ratio >= 8.0:
            classification = "Adequately Capitalized"
            action_required = "Monitor"
            risk_level = "Low"
        elif net_worth_ratio >= 6.0:
            classification = "Undercapitalized"
            action_required = "Capital Plan Required"
            risk_level = "Medium"
        elif net_worth_ratio >= 2.0:
            classification = "Significantly Undercapitalized"
            action_required = "Immediate Action Required"
            risk_level = "High"
        else:
            classification = "Critically Undercapitalized"
            action_required = "Regulatory Intervention"
            risk_level = "Critical"
        
        return {
            'classification': classification,
            'action_required': action_required,
            'risk_level': risk_level,
            'net_worth_ratio': net_worth_ratio
        }
    
    def _analyze_bsa_aml_compliance(self, transaction_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze BSA/AML compliance indicators."""
        if transaction_data.empty:
            return {
                'status': 'No Recent Transactions', 
                'compliant': True,
                'alerts': [],
                'metrics': {}
            }
        
        alerts = []
        
        # Check for large cash transactions (>$10,000)
        large_cash = transaction_data[
            (transaction_data['amount'] > 10000) & 
            (transaction_data['transaction_type'].str.contains('Cash', case=False, na=False))
        ]
        
        if not large_cash.empty:
            alerts.extend(self._generate_ctr_alerts(large_cash))
        
        # Check for suspicious patterns
        suspicious_patterns = self._detect_suspicious_patterns(transaction_data)
        alerts.extend(suspicious_patterns)
        
        # Check for rapid succession transactions (structuring)
        structuring_alerts = self._detect_structuring(transaction_data)
        alerts.extend(structuring_alerts)
        
        # Check foreign transactions from high-risk countries
        foreign_alerts = self._check_foreign_transactions(transaction_data)
        alerts.extend(foreign_alerts)
        
        # Calculate risk metrics
        total_alerts = len(alerts)
        high_risk_alerts = len([a for a in alerts if a.get('risk_level') == 'High'])
        
        return {
            'status': 'Compliant' if total_alerts == 0 else 'Alerts Present',
            'compliant': high_risk_alerts == 0,
            'total_alerts': total_alerts,
            'high_risk_alerts': high_risk_alerts,
            'alerts': alerts,
            'summary': {
                'large_cash_transactions': len(large_cash),
                'suspicious_patterns': len(suspicious_patterns),
                'structuring_alerts': len(structuring_alerts),
                'foreign_transaction_alerts': len(foreign_alerts)
            },
            'metrics': {
                'total_alerts': total_alerts,
                'high_risk_alerts': high_risk_alerts,
                'alert_rate': self.safe_divide(total_alerts, len(transaction_data)) * 100
            }
        }
    
    def _generate_ctr_alerts(self, large_cash: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate Currency Transaction Report alerts."""
        alerts = []
        
        for _, transaction in large_cash.iterrows():
            alerts.append({
                'type': 'CTR Required',
                'transaction_id': transaction['transaction_id'],
                'member_id': transaction['member_id'],
                'amount': transaction['amount'],
                'date': transaction['transaction_date'],
                'description': f"Cash transaction over $10,000: ${transaction['amount']:,.2f}",
                'risk_level': 'Medium',
                'action_required': 'File CTR within 15 days'
            })
        
        return alerts
    
    def _detect_suspicious_patterns(self, transaction_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect suspicious transaction patterns."""
        alerts = []
        
        # Group by member to analyze patterns
        member_groups = transaction_data.groupby('member_id')
        
        for member_id, member_transactions in member_groups:
            # Check for unusual frequency
            daily_counts = member_transactions.groupby('transaction_date').size()
            if daily_counts.max() > 10:  # More than 10 transactions in one day
                alerts.append({
                    'type': 'High Frequency',
                    'member_id': member_id,
                    'description': f"Member had {daily_counts.max()} transactions in one day",
                    'risk_level': 'Medium',
                    'transactions_count': daily_counts.max(),
                    'date': daily_counts.idxmax()
                })
            
            # Check for round dollar amounts (potential structuring)
            round_amounts = member_transactions[member_transactions['amount'] % 1000 == 0]
            if len(round_amounts) > 3:  # Multiple round amounts
                alerts.append({
                    'type': 'Round Amount Pattern',
                    'member_id': member_id,
                    'description': f"Member has {len(round_amounts)} transactions with round amounts",
                    'risk_level': 'Low',
                    'pattern_count': len(round_amounts)
                })
        
        return alerts
    
    def _detect_structuring(self, transaction_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect potential structuring (breaking large amounts into smaller ones)."""
        alerts = []
        
        # Look for multiple transactions just under $10,000 by same member on same day
        member_daily = transaction_data.groupby(['member_id', 'transaction_date'])
        
        for (member_id, date), group in member_daily:
            # Check for multiple transactions between $9,000 and $10,000
            near_threshold = group[
                (group['amount'] >= 9000) & 
                (group['amount'] < 10000)
            ]
            
            if len(near_threshold) >= 2:
                total_amount = near_threshold['amount'].sum()
                alerts.append({
                    'type': 'Potential Structuring',
                    'member_id': member_id,
                    'date': date,
                    'description': f"Multiple transactions just under $10,000 totaling ${total_amount:,.2f}",
                    'risk_level': 'High',
                    'transaction_count': len(near_threshold),
                    'total_amount': total_amount,
                    'action_required': 'Review for SAR filing'
                })
        
        return alerts
    
    def _check_foreign_transactions(self, transaction_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Check for foreign transactions requiring attention."""
        alerts = []
        
        # High-risk countries (simplified list)
        high_risk_countries = ['Iran', 'North Korea', 'Syria', 'Cuba']
        
        # Check for foreign transactions
        foreign_transactions = transaction_data[
            transaction_data['foreign_transaction'] == True
        ]
        
        for _, transaction in foreign_transactions.iterrows():
            country = transaction.get('country_of_citizenship', 'Unknown')
            
            if country in high_risk_countries:
                alerts.append({
                    'type': 'High-Risk Foreign Transaction',
                    'transaction_id': transaction['transaction_id'],
                    'member_id': transaction['member_id'],
                    'amount': transaction['amount'],
                    'country': country,
                    'description': f"Transaction from high-risk country: {country}",
                    'risk_level': 'High',
                    'action_required': 'Enhanced due diligence'
                })
            elif transaction['amount'] > 5000:
                alerts.append({
                    'type': 'Large Foreign Transaction',
                    'transaction_id': transaction['transaction_id'],
                    'member_id': transaction['member_id'],
                    'amount': transaction['amount'],
                    'country': country,
                    'description': f"Large foreign transaction: ${transaction['amount']:,.2f}",
                    'risk_level': 'Medium'
                })
        
        return alerts
    
    def _analyze_lending_limits(self, financial_data: pd.DataFrame, loan_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze lending limit compliance."""
        if financial_data.empty:
            return {'status': 'No Data', 'compliant': False}
        
        row = financial_data.iloc[0]
        
        total_assets = float(row.get('total_assets', 0))
        net_worth = float(row.get('net_worth', 0))
        mbl_outstanding = float(row.get('mbl_outstanding', 0))
        
        # Member Business Loan limits
        mbl_analysis = self._check_mbl_limits(mbl_outstanding, total_assets, net_worth)
        
        # Individual loan limits (if loan data available)
        individual_limits = {}
        if not loan_data.empty:
            individual_limits = self._check_individual_loan_limits(loan_data, net_worth)
        
        # Aggregate lending limits
        aggregate_limits = self._check_aggregate_limits(financial_data, loan_data)
        
        # Overall compliance
        is_compliant = (
            mbl_analysis['compliant'] and
            individual_limits.get('compliant', True) and
            aggregate_limits.get('compliant', True)
        )
        
        return {
            'status': 'Compliant' if is_compliant else 'Non-Compliant',
            'compliant': is_compliant,
            'mbl_limits': mbl_analysis,
            'individual_limits': individual_limits,
            'aggregate_limits': aggregate_limits,
            'metrics': {
                'mbl_utilization': mbl_analysis.get('utilization_pct', 0),
                'mbl_compliance': 1 if mbl_analysis['compliant'] else 0,
                'overall_compliance': 1 if is_compliant else 0
            }
        }
    
    def _check_mbl_limits(self, mbl_outstanding: float, total_assets: float, net_worth: float) -> Dict[str, Any]:
        """Check Member Business Loan regulatory limits."""
        # MBL limits: 1.75x net worth OR 12.25% of assets, whichever is less
        limit_by_net_worth = net_worth * 1.75
        limit_by_assets = total_assets * 0.1225
        applicable_limit = min(limit_by_net_worth, limit_by_assets)
        
        utilization_pct = self.safe_divide(mbl_outstanding, applicable_limit) * 100
        is_compliant = mbl_outstanding <= applicable_limit
        
        return {
            'mbl_balance': mbl_outstanding,
            'applicable_limit': applicable_limit,
            'limit_by_net_worth': limit_by_net_worth,
            'limit_by_assets': limit_by_assets,
            'utilization_pct': utilization_pct,
            'compliant': is_compliant,
            'available_capacity': max(0, applicable_limit - mbl_outstanding),
            'status': 'Within Limits' if is_compliant else 'Exceeds Limits'
        }
    
    def _check_individual_loan_limits(self, loan_data: pd.DataFrame, net_worth: float) -> Dict[str, Any]:
        """Check individual loan limits."""
        if loan_data.empty or net_worth == 0:
            return {'compliant': True, 'violations': []}
        
        # Individual loan limit is typically 15% of net worth for unsecured loans
        unsecured_limit = net_worth * 0.15
        secured_limit = net_worth * 0.25  # Higher limit for secured loans
        
        violations = []
        
        for _, loan in loan_data.iterrows():
            loan_balance = float(loan.get('current_balance', 0))
            is_secured = loan.get('secured_unsecured', 'Unknown') == 'Secured'
            
            applicable_limit = secured_limit if is_secured else unsecured_limit
            
            if loan_balance > applicable_limit:
                violations.append({
                    'loan_id': loan['loan_id'],
                    'member_id': loan['member_id'],
                    'loan_balance': loan_balance,
                    'applicable_limit': applicable_limit,
                    'excess_amount': loan_balance - applicable_limit,
                    'loan_type': 'Secured' if is_secured else 'Unsecured'
                })
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'violations_count': len(violations),
            'unsecured_limit': unsecured_limit,
            'secured_limit': secured_limit
        }
    
    def _check_aggregate_limits(self, financial_data: pd.DataFrame, loan_data: pd.DataFrame) -> Dict[str, Any]:
        """Check aggregate lending limits."""
        if financial_data.empty:
            return {'compliant': True}
        
        row = financial_data.iloc[0]
        total_assets = float(row.get('total_assets', 0))
        
        if loan_data.empty:
            total_loans = 0
        else:
            total_loans = loan_data['current_balance'].sum()
        
        # Typical aggregate limit is 80% of assets
        aggregate_limit = total_assets * 0.80
        loan_to_asset_ratio = self.safe_divide(total_loans, total_assets) * 100
        
        is_compliant = total_loans <= aggregate_limit
        
        return {
            'total_loans': total_loans,
            'total_assets': total_assets,
            'aggregate_limit': aggregate_limit,
            'loan_to_asset_ratio': loan_to_asset_ratio,
            'compliant': is_compliant,
            'utilization_pct': self.safe_divide(total_loans, aggregate_limit) * 100
        }
    
    def _analyze_cecl_requirements(self, loan_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze CECL (Current Expected Credit Loss) requirements."""
        if loan_data.empty:
            return {'status': 'No Loan Data', 'cecl_provision': 0}
        
        # Calculate CECL provision using simplified approach
        cecl_provision = 0
        loan_segments = []
        
        # Group loans by type for CECL calculation
        loan_types = loan_data['loan_type'].unique()
        
        for loan_type in loan_types:
            segment_loans = loan_data[loan_data['loan_type'] == loan_type]
            
            # Calculate expected credit losses for this segment
            segment_ecl = self._calculate_segment_ecl(segment_loans)
            cecl_provision += segment_ecl['total_ecl']
            
            loan_segments.append({
                'loan_type': loan_type,
                'loan_count': len(segment_loans),
                'total_balance': segment_loans['current_balance'].sum(),
                'expected_credit_loss': segment_ecl['total_ecl'],
                'ecl_rate': segment_ecl['ecl_rate']
            })
        
        total_loan_balance = loan_data['current_balance'].sum()
        current_provision = loan_data['provision_amount'].sum()
        
        provision_adequacy = self.safe_divide(current_provision, cecl_provision)
        provision_shortfall = max(0, cecl_provision - current_provision)
        
        return {
            'status': 'Adequate' if provision_adequacy >= 1.0 else 'Inadequate',
            'cecl_provision_required': cecl_provision,
            'current_provision': current_provision,
            'provision_adequacy_ratio': provision_adequacy,
            'provision_shortfall': provision_shortfall,
            'loan_segments': loan_segments,
            'total_loan_balance': total_loan_balance,
            'overall_ecl_rate': self.safe_divide(cecl_provision, total_loan_balance) * 100,
            'metrics': {
                'cecl_provision': cecl_provision,
                'provision_adequacy': provision_adequacy,
                'ecl_rate': self.safe_divide(cecl_provision, total_loan_balance) * 100
            }
        }
    
    def _calculate_segment_ecl(self, segment_loans: pd.DataFrame) -> Dict[str, float]:
        """Calculate Expected Credit Loss for a loan segment."""
        if segment_loans.empty:
            return {'total_ecl': 0, 'ecl_rate': 0}
        
        total_ecl = 0
        total_balance = segment_loans['current_balance'].sum()
        
        for _, loan in segment_loans.iterrows():
            loan_balance = float(loan.get('current_balance', 0))
            
            # Use provided PD and LGD if available, otherwise estimate
            pd_rate = float(loan.get('probability_of_default', 0))
            lgd_rate = float(loan.get('loss_given_default', 0))
            expected_life = float(loan.get('expected_life', 1))
            
            # If not provided, estimate based on credit quality
            if pd_rate == 0:
                pd_rate = self._estimate_pd_from_fico(loan.get('fico_score', 750))
            
            if lgd_rate == 0:
                lgd_rate = self._estimate_lgd_by_loan_type(loan.get('loan_type', 'Other'))
            
            if expected_life == 0:
                expected_life = self._estimate_expected_life(loan)
            
            # Calculate lifetime ECL
            lifetime_pd = 1 - ((1 - pd_rate) ** expected_life)
            loan_ecl = loan_balance * lifetime_pd * lgd_rate
            
            total_ecl += loan_ecl
        
        ecl_rate = self.safe_divide(total_ecl, total_balance)
        
        return {
            'total_ecl': total_ecl,
            'ecl_rate': ecl_rate
        }
    
    def _estimate_pd_from_fico(self, fico_score: float) -> float:
        """Estimate probability of default from FICO score."""
        if fico_score >= 760:
            return 0.005  # 0.5%
        elif fico_score >= 700:
            return 0.015  # 1.5%
        elif fico_score >= 650:
            return 0.035  # 3.5%
        elif fico_score >= 600:
            return 0.075  # 7.5%
        else:
            return 0.150  # 15%
    
    def _estimate_lgd_by_loan_type(self, loan_type: str) -> float:
        """Estimate Loss Given Default by loan type."""
        lgd_rates = {
            'Auto Loan': 0.25,
            'Real Estate': 0.20,
            'Personal Loan': 0.60,
            'Credit Card': 0.75,
            'Business Loan': 0.45,
            'Home Equity': 0.30
        }
        
        return lgd_rates.get(loan_type, 0.50)  # Default 50%
    
    def _estimate_expected_life(self, loan: pd.Series) -> float:
        """Estimate expected life of loan in years."""
        if pd.notna(loan.get('maturity_date')) and pd.notna(loan.get('origination_date')):
            maturity = pd.to_datetime(loan['maturity_date'])
            origination = pd.to_datetime(loan['origination_date'])
            total_life = (maturity - origination).days / 365.25
            
            # Assume average remaining life is half of total life
            return total_life / 2
        else:
            # Default estimates by loan type
            loan_type = loan.get('loan_type', 'Other')
            if 'Auto' in loan_type:
                return 3.0  # 3 years average remaining
            elif 'Real Estate' in loan_type or 'Mortgage' in loan_type:
                return 15.0  # 15 years average remaining
            elif 'Personal' in loan_type:
                return 2.0  # 2 years average remaining
            else:
                return 2.5  # Default 2.5 years
    
    def _calculate_compliance_score(self, compliance_results: Dict[str, Any]) -> float:
        """Calculate overall compliance score (0-100)."""
        weights = {
            'capital': 0.40,      # 40% weight - most critical
            'bsa_aml': 0.25,      # 25% weight
            'lending_limits': 0.20, # 20% weight
            'cecl': 0.15          # 15% weight
        }
        
        scores = {}
        
        # Capital score
        capital_data = compliance_results.get('capital', {})
        if capital_data.get('compliant', False):
            net_worth_ratio = capital_data.get('ratios', {}).get('net_worth_ratio', 0)
            if net_worth_ratio >= 10:
                scores['capital'] = 100
            elif net_worth_ratio >= 8:
                scores['capital'] = 85
            elif net_worth_ratio >= 7:
                scores['capital'] = 70
            else:
                scores['capital'] = 50
        else:
            scores['capital'] = 30
        
        # BSA/AML score
        bsa_data = compliance_results.get('bsa_aml', {})
        high_risk_alerts = bsa_data.get('high_risk_alerts', 0)
        total_alerts = bsa_data.get('total_alerts', 0)
        
        if high_risk_alerts == 0:
            scores['bsa_aml'] = 100 - min(total_alerts * 5, 30)  # Reduce score for total alerts
        else:
            scores['bsa_aml'] = 60 - min(high_risk_alerts * 10, 40)
        
        # Lending limits score
        lending_data = compliance_results.get('lending_limits', {})
        if lending_data.get('compliant', True):
            scores['lending_limits'] = 100
        else:
            # Reduce score based on number of violations
            mbl_compliant = lending_data.get('mbl_limits', {}).get('compliant', True)
            individual_violations = len(lending_data.get('individual_limits', {}).get('violations', []))
            
            if mbl_compliant and individual_violations == 0:
                scores['lending_limits'] = 100
            elif mbl_compliant:
                scores['lending_limits'] = 80 - min(individual_violations * 5, 30)
            else:
                scores['lending_limits'] = 60
        
        # CECL score
        cecl_data = compliance_results.get('cecl', {})
        adequacy_ratio = cecl_data.get('provision_adequacy_ratio', 1.0)
        
        if adequacy_ratio >= 1.0:
            scores['cecl'] = 100
        elif adequacy_ratio >= 0.9:
            scores['cecl'] = 85
        elif adequacy_ratio >= 0.8:
            scores['cecl'] = 70
        else:
            scores['cecl'] = 50
        
        # Calculate weighted average
        total_score = sum(scores[area] * weights[area] for area in weights.keys())
        
        return min(100, max(0, total_score))
    
    def _get_compliance_rating(self, compliance_score: float) -> str:
        """Convert compliance score to rating."""
        if compliance_score >= 90:
            return "Excellent"
        elif compliance_score >= 80:
            return "Good"
        elif compliance_score >= 70:
            return "Satisfactory"
        elif compliance_score >= 60:
            return "Needs Improvement"
        else:
            return "Poor"
    
    def _identify_critical_issues(self, compliance_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify critical compliance issues requiring immediate attention."""
        critical_issues = []
        
        # Capital adequacy issues
        capital_data = compliance_results.get('capital', {})
        if not capital_data.get('compliant', True):
            classification = capital_data.get('classification', {})
            critical_issues.append({
                'category': 'Capital Adequacy',
                'severity': 'Critical',
                'issue': f"Capital classification: {classification.get('classification', 'Unknown')}",
                'action_required': classification.get('action_required', 'Review required'),
                'regulatory_impact': 'High'
            })
        
        # High-risk BSA/AML alerts
        bsa_data = compliance_results.get('bsa_aml', {})
        high_risk_alerts = bsa_data.get('high_risk_alerts', 0)
        if high_risk_alerts > 0:
            critical_issues.append({
                'category': 'BSA/AML',
                'severity': 'High',
                'issue': f"{high_risk_alerts} high-risk BSA/AML alerts",
                'action_required': 'Review and file SARs if necessary',
                'regulatory_impact': 'High'
            })
        
        # Lending limit violations
        lending_data = compliance_results.get('lending_limits', {})
        if not lending_data.get('compliant', True):
            mbl_compliant = lending_data.get('mbl_limits', {}).get('compliant', True)
            if not mbl_compliant:
                critical_issues.append({
                    'category': 'Lending Limits',
                    'severity': 'Critical',
                    'issue': 'Member Business Loan limits exceeded',
                    'action_required': 'Reduce MBL exposure immediately',
                    'regulatory_impact': 'High'
                })
        
        # CECL provision inadequacy
        cecl_data = compliance_results.get('cecl', {})
        adequacy_ratio = cecl_data.get('provision_adequacy_ratio', 1.0)
        if adequacy_ratio < 0.8:
            shortfall = cecl_data.get('provision_shortfall', 0)
            critical_issues.append({
                'category': 'CECL',
                'severity': 'Medium',
                'issue': f"CECL provision shortfall: ${shortfall:,.2f}",
                'action_required': 'Increase loan loss provisions',
                'regulatory_impact': 'Medium'
            })
        
        return critical_issues
    
    def _generate_compliance_recommendations(self, critical_issues: List[Dict[str, Any]]) -> List[str]:
        """Generate compliance recommendations based on identified issues."""
        recommendations = []
        
        if not critical_issues:
            recommendations.append("Maintain current compliance monitoring procedures")
            return recommendations
        
        # Group issues by category
        issues_by_category = {}
        for issue in critical_issues:
            category = issue['category']
            if category not in issues_by_category:
                issues_by_category[category] = []
            issues_by_category[category].append(issue)
        
        # Generate category-specific recommendations
        for category, issues in issues_by_category.items():
            if category == 'Capital Adequacy':
                recommendations.append("Develop capital restoration plan and consider reducing risk-weighted assets")
            elif category == 'BSA/AML':
                recommendations.append("Enhance transaction monitoring systems and staff training")
            elif category == 'Lending Limits':
                recommendations.append("Review lending policies and implement stronger limit monitoring")
            elif category == 'CECL':
                recommendations.append("Update CECL models and consider increasing loan loss provisions")
        
        # General recommendations
        if len(critical_issues) > 2:
            recommendations.append("Consider engaging external compliance consultant for comprehensive review")
        
        return recommendations
