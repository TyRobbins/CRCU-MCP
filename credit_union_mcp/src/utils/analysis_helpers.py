"""
Analysis Helpers for Credit Union MCP Server

Provides common analysis functions, calculations, and statistical methods
to reduce code duplication across agents and ensure consistent results.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta, date
import math
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

from loguru import logger


class FinancialCalculations:
    """Common financial calculations and ratios."""
    
    @staticmethod
    def calculate_roa(net_income: float, total_assets: float) -> float:
        """Calculate Return on Assets."""
        if total_assets == 0:
            return 0.0
        return (net_income / total_assets) * 100
    
    @staticmethod
    def calculate_roe(net_income: float, total_equity: float) -> float:
        """Calculate Return on Equity."""
        if total_equity == 0:
            return 0.0
        return (net_income / total_equity) * 100
    
    @staticmethod
    def calculate_nim(net_interest_income: float, average_earning_assets: float) -> float:
        """Calculate Net Interest Margin."""
        if average_earning_assets == 0:
            return 0.0
        return (net_interest_income / average_earning_assets) * 100
    
    @staticmethod
    def calculate_efficiency_ratio(non_interest_expense: float, 
                                 net_interest_income: float, 
                                 non_interest_income: float) -> float:
        """Calculate Efficiency Ratio."""
        total_revenue = net_interest_income + non_interest_income
        if total_revenue == 0:
            return 100.0  # Worst case
        return (non_interest_expense / total_revenue) * 100
    
    @staticmethod
    def calculate_capital_ratio(tier1_capital: float, risk_weighted_assets: float) -> float:
        """Calculate Tier 1 Capital Ratio."""
        if risk_weighted_assets == 0:
            return 0.0
        return (tier1_capital / risk_weighted_assets) * 100
    
    @staticmethod
    def calculate_leverage_ratio(tier1_capital: float, total_assets: float) -> float:
        """Calculate Leverage Ratio."""
        if total_assets == 0:
            return 0.0
        return (tier1_capital / total_assets) * 100
    
    @staticmethod
    def calculate_loan_to_deposit_ratio(total_loans: float, total_deposits: float) -> float:
        """Calculate Loan-to-Deposit Ratio."""
        if total_deposits == 0:
            return 0.0
        return (total_loans / total_deposits) * 100
    
    @staticmethod
    def calculate_net_worth_ratio(total_equity: float, total_assets: float) -> float:
        """Calculate Net Worth Ratio."""
        if total_assets == 0:
            return 0.0
        return (total_equity / total_assets) * 100
    
    @staticmethod
    def calculate_delinquency_rate(delinquent_loans: float, total_loans: float) -> float:
        """Calculate Delinquency Rate."""
        if total_loans == 0:
            return 0.0
        return (delinquent_loans / total_loans) * 100
    
    @staticmethod
    def calculate_charge_off_rate(charge_offs: float, total_loans: float) -> float:
        """Calculate Charge-off Rate."""
        if total_loans == 0:
            return 0.0
        return (charge_offs / total_loans) * 100


class StatisticalAnalysis:
    """Statistical analysis and scoring methods."""
    
    @staticmethod
    def calculate_z_score(value: float, mean: float, std_dev: float) -> float:
        """Calculate Z-score for a value."""
        if std_dev == 0:
            return 0.0
        return (value - mean) / std_dev
    
    @staticmethod
    def calculate_percentile_rank(value: float, data_series: pd.Series) -> float:
        """Calculate percentile rank of a value in a series."""
        if data_series.empty:
            return 50.0
        return stats.percentileofscore(data_series.dropna(), value)
    
    @staticmethod
    def detect_outliers(data: pd.Series, method: str = 'iqr', 
                       threshold: float = 1.5) -> pd.Series:
        """Detect outliers using IQR or Z-score method."""
        if data.empty:
            return pd.Series(dtype=bool)
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return (data < lower_bound) | (data > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data.dropna()))
            return pd.Series(z_scores > threshold, index=data.index)
        
        else:
            raise ValueError("Method must be 'iqr' or 'zscore'")
    
    @staticmethod
    def calculate_correlation_matrix(data: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix with handling for non-numeric data."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return pd.DataFrame()
        return data[numeric_cols].corr()
    
    @staticmethod
    def perform_trend_analysis(data: pd.Series, periods: int = 12) -> Dict[str, float]:
        """Perform trend analysis on time series data."""
        if len(data) < 2:
            return {'trend': 0.0, 'slope': 0.0, 'r_squared': 0.0}
        
        # Create time index
        x = np.arange(len(data))
        y = data.values
        
        # Calculate linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Determine trend direction
        if abs(slope) < 0.01:
            trend = 0.0  # Flat
        elif slope > 0:
            trend = 1.0  # Upward
        else:
            trend = -1.0  # Downward
        
        return {
            'trend': trend,
            'slope': slope,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'forecast_next': intercept + slope * len(data)
        }


class RiskAnalysis:
    """Risk assessment and concentration analysis methods."""
    
    @staticmethod
    def calculate_herfindahl_index(balances: pd.Series) -> float:
        """Calculate Herfindahl-Hirschman Index for concentration."""
        if balances.empty or balances.sum() == 0:
            return 0.0
        
        # Convert to market shares (proportions)
        shares = balances / balances.sum()
        # Square each share and sum
        hhi = (shares ** 2).sum()
        return hhi
    
    @staticmethod
    def calculate_concentration_ratio(balances: pd.Series, top_n: int = 5) -> float:
        """Calculate concentration ratio for top N entities."""
        if balances.empty or balances.sum() == 0:
            return 0.0
        
        # Sort in descending order and take top N
        top_balances = balances.nlargest(top_n)
        return (top_balances.sum() / balances.sum()) * 100
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk (VaR)."""
        if returns.empty:
            return 0.0
        
        return np.percentile(returns.dropna(), confidence_level * 100)
    
    @staticmethod
    def calculate_expected_shortfall(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        if returns.empty:
            return 0.0
        
        var = StatisticalAnalysis.calculate_var(returns, confidence_level)
        tail_losses = returns[returns <= var]
        return tail_losses.mean() if not tail_losses.empty else 0.0
    
    @staticmethod
    def stress_test_scenario(base_value: float, shock_percentage: float) -> Dict[str, float]:
        """Apply stress test scenario to a base value."""
        stressed_value = base_value * (1 + shock_percentage / 100)
        impact = stressed_value - base_value
        
        return {
            'base_value': base_value,
            'stressed_value': stressed_value,
            'absolute_impact': impact,
            'percentage_impact': (impact / base_value * 100) if base_value != 0 else 0.0
        }


class MemberAnalysis:
    """Member behavior and segmentation analysis methods."""
    
    @staticmethod
    def calculate_rfm_scores(transaction_data: pd.DataFrame,
                           customer_id_col: str = 'member_id',
                           date_col: str = 'transaction_date',
                           amount_col: str = 'transaction_amount') -> pd.DataFrame:
        """Calculate RFM (Recency, Frequency, Monetary) scores."""
        
        if transaction_data.empty:
            return pd.DataFrame()
        
        # Ensure date column is datetime
        transaction_data[date_col] = pd.to_datetime(transaction_data[date_col])
        reference_date = transaction_data[date_col].max()
        
        # Calculate RFM metrics
        rfm = transaction_data.groupby(customer_id_col).agg({
            date_col: lambda x: (reference_date - x.max()).days,  # Recency
            amount_col: ['count', 'sum']  # Frequency and Monetary
        }).round(2)
        
        # Flatten column names
        rfm.columns = ['recency', 'frequency', 'monetary']
        rfm.reset_index(inplace=True)
        
        # Calculate quintiles for scoring
        rfm['r_score'] = pd.qcut(rfm['recency'].rank(method='first'), 5, labels=[5,4,3,2,1])
        rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
        rfm['m_score'] = pd.qcut(rfm['monetary'].rank(method='first'), 5, labels=[1,2,3,4,5])
        
        # Combine scores
        rfm['rfm_score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)
        
        return rfm
    
    @staticmethod
    def segment_rfm_customers(rfm_data: pd.DataFrame) -> pd.DataFrame:
        """Segment customers based on RFM scores."""
        
        def assign_segment(row):
            r, f, m = int(row['r_score']), int(row['f_score']), int(row['m_score'])
            
            if r >= 4 and f >= 4 and m >= 4:
                return 'Champions'
            elif r >= 3 and f >= 3 and m >= 3:
                return 'Loyal Customers'
            elif r >= 4 and f <= 2:
                return 'New Customers'
            elif r >= 3 and f >= 3 and m <= 2:
                return 'Potential Loyalists'
            elif r >= 3 and f <= 2:
                return 'Promising'
            elif r <= 2 and f >= 3 and m >= 3:
                return 'Need Attention'
            elif r <= 2 and f >= 3:
                return 'About to Sleep'
            elif r >= 3 and f <= 2 and m <= 2:
                return 'At Risk'
            elif r <= 2 and f <= 2:
                return 'Cannot Lose Them'
            else:
                return 'Others'
        
        if 'r_score' in rfm_data.columns:
            rfm_data['segment'] = rfm_data.apply(assign_segment, axis=1)
        
        return rfm_data
    
    @staticmethod
    def calculate_customer_lifetime_value(member_data: pd.DataFrame,
                                        transaction_data: pd.DataFrame,
                                        prediction_months: int = 24) -> pd.DataFrame:
        """Calculate Customer Lifetime Value."""
        
        if transaction_data.empty:
            return pd.DataFrame()
        
        # Calculate average transaction value and frequency per member
        member_stats = transaction_data.groupby('member_id').agg({
            'transaction_amount': ['mean', 'count'],
            'transaction_date': ['min', 'max']
        }).round(2)
        
        member_stats.columns = ['avg_transaction', 'total_transactions', 'first_transaction', 'last_transaction']
        member_stats.reset_index(inplace=True)
        
        # Calculate transaction frequency (transactions per month)
        member_stats['days_active'] = (member_stats['last_transaction'] - member_stats['first_transaction']).dt.days
        member_stats['months_active'] = member_stats['days_active'] / 30.44  # Average days per month
        member_stats['transactions_per_month'] = member_stats['total_transactions'] / member_stats['months_active'].replace(0, 1)
        
        # Calculate CLV components
        member_stats['monthly_value'] = member_stats['avg_transaction'] * member_stats['transactions_per_month']
        member_stats['predicted_ltv'] = member_stats['monthly_value'] * prediction_months
        
        return member_stats
    
    @staticmethod
    def perform_clustering(features: pd.DataFrame, max_clusters: int = 8) -> Dict[str, Any]:
        """Perform K-means clustering on member features."""
        
        if features.empty or len(features) < max_clusters:
            return {'error': 'Insufficient data for clustering'}
        
        # Prepare features - only numeric columns
        numeric_features = features.select_dtypes(include=[np.number])
        if numeric_features.empty:
            return {'error': 'No numeric features for clustering'}
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(numeric_features.fillna(0))
        
        # Find optimal number of clusters using silhouette score
        silhouette_scores = []
        k_range = range(2, min(max_clusters + 1, len(features)))
        
        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(scaled_features)
                silhouette_avg = silhouette_score(scaled_features, cluster_labels)
                silhouette_scores.append(silhouette_avg)
            except:
                silhouette_scores.append(0)
        
        if not silhouette_scores:
            return {'error': 'Clustering failed'}
        
        # Select optimal number of clusters
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        # Perform final clustering
        final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = final_kmeans.fit_predict(scaled_features)
        
        # Analyze clusters
        cluster_analysis = {}
        for cluster in range(optimal_k):
            cluster_mask = cluster_labels == cluster
            cluster_data = numeric_features[cluster_mask]
            
            cluster_analysis[f'cluster_{cluster}'] = {
                'size': int(cluster_mask.sum()),
                'percentage': float(cluster_mask.sum() / len(cluster_labels) * 100),
                'characteristics': cluster_data.mean().to_dict()
            }
        
        return {
            'optimal_clusters': optimal_k,
            'silhouette_score': max(silhouette_scores),
            'cluster_labels': cluster_labels,
            'cluster_analysis': cluster_analysis,
            'feature_names': list(numeric_features.columns)
        }


class DateTimeHelpers:
    """Date and time utility functions."""
    
    @staticmethod
    def convert_arcusym_date(date_value: Union[int, str, float]) -> Optional[datetime]:
        """Convert ARCUSYM date format to datetime."""
        try:
            if pd.isna(date_value) or date_value == 0:
                return None
            
            if isinstance(date_value, str):
                # Try different string formats
                for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%Y%m%d']:
                    try:
                        return datetime.strptime(date_value, fmt)
                    except ValueError:
                        continue
                return None
            
            elif isinstance(date_value, (int, float)):
                date_int = int(date_value)
                if date_int == 0:
                    return None
                
                # ARCUSYM format: CYYMMDD where C is century (0=1900s, 1=2000s)
                if date_int > 9999999:  # 8 digits
                    date_str = str(date_int)
                    century = int(date_str[0])
                    year = int(date_str[1:3]) + (1900 if century == 0 else 2000)
                    month = int(date_str[3:5])
                    day = int(date_str[5:7])
                else:  # 7 digits or less - assume YYMMDD format
                    date_str = str(date_int).zfill(6)
                    year = int(date_str[0:2])
                    year = year + (2000 if year < 50 else 1900)  # Pivot at 50
                    month = int(date_str[2:4])
                    day = int(date_str[4:6])
                
                return datetime(year, month, day)
                
        except (ValueError, IndexError):
            return None
    
    @staticmethod
    def calculate_age(birth_date: datetime, as_of_date: Optional[datetime] = None) -> int:
        """Calculate age from birth date."""
        if birth_date is None:
            return 0
        
        if as_of_date is None:
            as_of_date = datetime.now()
        
        age = as_of_date.year - birth_date.year
        
        # Adjust for birthday not yet occurred this year
        if as_of_date.month < birth_date.month or \
           (as_of_date.month == birth_date.month and as_of_date.day < birth_date.day):
            age -= 1
        
        return max(0, age)
    
    @staticmethod
    def get_business_days_between(start_date: datetime, end_date: datetime) -> int:
        """Calculate business days between two dates."""
        return pd.bdate_range(start_date, end_date).size
    
    @staticmethod
    def get_month_end(date_input: datetime) -> datetime:
        """Get last day of month for given date."""
        next_month = date_input.replace(day=28) + timedelta(days=4)
        return next_month - timedelta(days=next_month.day)
    
    @staticmethod
    def get_quarter_end(date_input: datetime) -> datetime:
        """Get last day of quarter for given date."""
        quarter = (date_input.month - 1) // 3 + 1
        quarter_end_month = quarter * 3
        quarter_end = date_input.replace(month=quarter_end_month, day=1)
        return DateTimeHelpers.get_month_end(quarter_end)


class ValidationHelpers:
    """Data validation and quality check utilities."""
    
    @staticmethod
    def validate_financial_data(data: pd.DataFrame, required_cols: List[str]) -> Dict[str, Any]:
        """Validate financial data quality."""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check required columns
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_cols}")
        
        # Check for empty dataset
        if data.empty:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Dataset is empty")
            return validation_results
        
        # Check data quality
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = data[col]
            
            # Check for negative values in amount columns
            if 'amount' in col.lower() or 'balance' in col.lower():
                if (col_data < 0).any():
                    validation_results['warnings'].append(f"Negative values found in {col}")
            
            # Check for extreme outliers
            if not col_data.empty:
                q1, q3 = col_data.quantile([0.25, 0.75])
                iqr = q3 - q1
                outliers = col_data[(col_data < q1 - 3*iqr) | (col_data > q3 + 3*iqr)]
                
                if len(outliers) > len(col_data) * 0.05:  # More than 5% outliers
                    validation_results['warnings'].append(f"High number of outliers in {col}: {len(outliers)}")
        
        # Calculate data statistics
        validation_results['statistics'] = {
            'row_count': len(data),
            'column_count': len(data.columns),
            'null_percentage': (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100,
            'numeric_columns': len(numeric_cols),
            'duplicate_rows': data.duplicated().sum()
        }
        
        return validation_results
    
    @staticmethod
    def clean_numeric_data(series: pd.Series, 
                          remove_outliers: bool = False,
                          outlier_method: str = 'iqr',
                          fill_na: bool = True) -> pd.Series:
        """Clean and standardize numeric data."""
        cleaned = series.copy()
        
        # Convert to numeric, coercing errors to NaN
        cleaned = pd.to_numeric(cleaned, errors='coerce')
        
        # Remove outliers if requested
        if remove_outliers and not cleaned.empty:
            outlier_mask = StatisticalAnalysis.detect_outliers(cleaned, method=outlier_method)
            cleaned.loc[outlier_mask] = np.nan
        
        # Fill NaN values
        if fill_na:
            if cleaned.dtype in ['int64', 'float64']:
                cleaned = cleaned.fillna(cleaned.median())
            else:
                cleaned = cleaned.fillna(0)
        
        return cleaned
    
    @staticmethod
    def standardize_member_data(member_data: pd.DataFrame) -> pd.DataFrame:
        """Standardize member data formats and types."""
        standardized = member_data.copy()
        
        # Standardize common column names
        column_mapping = {
            'ACCOUNT': 'member_id',
            'FIRST': 'first_name', 
            'LAST': 'last_name',
            'BIRTHDATE': 'birth_date',
            'OPENDATE': 'join_date',
            'CLOSEDATE': 'close_date',
            'STATUS': 'status',
            'ADDRESS': 'address',
            'CITY': 'city',
            'STATE': 'state',
            'ZIPCODE': 'zip_code',
            'EMAIL': 'email',
            'PHONE': 'phone'
        }
        
        # Rename columns if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in standardized.columns:
                standardized = standardized.rename(columns={old_name: new_name})
        
        # Convert date columns
        date_columns = ['birth_date', 'join_date', 'close_date']
        for col in date_columns:
            if col in standardized.columns:
                standardized[col] = standardized[col].apply(DateTimeHelpers.convert_arcusym_date)
        
        # Clean and standardize string columns
        string_columns = ['first_name', 'last_name', 'address', 'city', 'state']
        for col in string_columns:
            if col in standardized.columns:
                standardized[col] = standardized[col].astype(str).str.strip().str.title()
        
        # Standardize phone numbers (basic cleaning)
        if 'phone' in standardized.columns:
            standardized['phone'] = standardized['phone'].astype(str).str.replace(r'[^\d]', '', regex=True)
        
        # Standardize zip codes
        if 'zip_code' in standardized.columns:
            standardized['zip_code'] = standardized['zip_code'].astype(str).str.strip().str[:5]
        
        return standardized


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if division by zero."""
    try:
        if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
            return default
        return float(numerator / denominator)
    except (TypeError, ZeroDivisionError):
        return default


def safe_percentage(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Calculate percentage safely with zero-division protection."""
    return safe_divide(numerator, denominator, default) * 100


def format_currency(amount: float, currency_symbol: str = '$') -> str:
    """Format number as currency string."""
    try:
        if pd.isna(amount):
            return f"{currency_symbol}0.00"
        return f"{currency_symbol}{amount:,.2f}"
    except (TypeError, ValueError):
        return f"{currency_symbol}0.00"


def format_percentage(value: float, decimal_places: int = 2) -> str:
    """Format number as percentage string."""
    try:
        if pd.isna(value):
            return "0.00%"
        return f"{value:.{decimal_places}f}%"
    except (TypeError, ValueError):
        return "0.00%"


def calculate_percentage_change(current: float, previous: float) -> float:
    """Calculate percentage change between two values."""
    if previous == 0 or pd.isna(previous) or pd.isna(current):
        return 0.0
    return ((current - previous) / previous) * 100


def create_age_groups(ages: pd.Series) -> pd.Series:
    """Create age group categories from age data."""
    return pd.cut(ages, 
                 bins=[0, 25, 35, 45, 55, 65, 100], 
                 labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'],
                 include_lowest=True)


def create_balance_tiers(balances: pd.Series) -> pd.Series:
    """Create balance tier categories."""
    return pd.cut(balances,
                 bins=[0, 1000, 5000, 25000, 100000, float('inf')],
                 labels=['< $1K', '$1K-$5K', '$5K-$25K', '$25K-$100K', '$100K+'],
                 include_lowest=True)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame column names to lowercase with underscores."""
    df_copy = df.copy()
    df_copy.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df_copy.columns]
    return df_copy


def get_business_rules() -> Dict[str, Any]:
    """Get common business rules and thresholds for credit unions."""
    return {
        'financial_thresholds': {
            'roa_excellent': 1.0,
            'roa_good': 0.75,
            'roa_satisfactory': 0.50,
            'roe_excellent': 10.0,
            'roe_good': 8.0,
            'nim_excellent': 3.5,
            'nim_good': 3.0,
            'efficiency_ratio_excellent': 65.0,
            'efficiency_ratio_good': 75.0,
            'net_worth_ratio_well_capitalized': 7.0,
            'net_worth_ratio_adequately_capitalized': 6.0
        },
        'risk_thresholds': {
            'delinquency_rate_low': 1.0,
            'delinquency_rate_moderate': 3.0,
            'delinquency_rate_high': 5.0,
            'concentration_warning': 15.0,
            'concentration_limit': 25.0,
            'hhi_low_concentration': 0.15,
            'hhi_moderate_concentration': 0.25,
            'loan_to_asset_ratio_high': 75.0
        },
        'member_thresholds': {
            'high_value_member': 50000.0,
            'new_member_days': 90,
            'inactive_member_days': 365,
            'frequent_transaction_monthly': 10,
            'digital_adoption_good': 70.0
        }
    }
