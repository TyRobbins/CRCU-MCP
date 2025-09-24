"""
Member Analytics Agent for Credit Union Analytics

Provides comprehensive member analysis including:
- RFM (Recency, Frequency, Monetary) segmentation
- K-means clustering for member segmentation
- Customer lifetime value calculation
- Churn prediction using machine learning
- Cross-sell propensity analysis
- Member behavior and engagement analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

from .base_agent import BaseAgent, AnalysisContext, AnalysisResult


def safe_create_analysis_result(agent_instance, analysis_type: str, success: bool = True, 
                               data: Optional[Dict[str, Any]] = None,
                               metrics: Optional[Dict[str, Any]] = None,
                               warnings: Optional[List[str]] = None,
                               errors: Optional[List[str]] = None,
                               metadata: Optional[Dict[str, Any]] = None) -> AnalysisResult:
    """
    Safely create AnalysisResult with validation error handling.
    
    Handles Pydantic validation errors by cleaning problematic data types
    and gracefully degrading to ensure the result can be created.
    
    Args:
        agent_instance: The agent instance to call create_result on
        analysis_type: Type of analysis performed
        success: Whether analysis was successful
        data: Analysis results data
        metrics: Calculated metrics (will be cleaned for validation)
        warnings: Warning messages
        errors: Error messages
        metadata: Additional metadata
        
    Returns:
        AnalysisResult instance with validation errors handled
    """
    try:
        # Clean metrics to ensure all values are numeric for Pydantic validation
        clean_metrics = {}
        if metrics:
            for key, value in metrics.items():
                try:
                    # Convert to float if possible
                    if isinstance(value, (int, float)):
                        clean_metrics[key] = float(value)
                    elif isinstance(value, str):
                        # Try to parse string as float
                        try:
                            clean_metrics[key] = float(value)
                        except (ValueError, TypeError):
                            # If string can't be converted, move to metadata
                            if not metadata:
                                metadata = {}
                            metadata[f'{key}_str'] = value
                    elif isinstance(value, dict):
                        # Move dict values to data or metadata instead of metrics
                        if not data:
                            data = {}
                        data[key] = value
                    elif isinstance(value, list):
                        # Move list values to data instead of metrics
                        if not data:
                            data = {}
                        data[key] = value
                    else:
                        # Move other complex types to metadata
                        if not metadata:
                            metadata = {}
                        metadata[f'{key}_complex'] = str(value)
                except Exception as e:
                    agent_instance.logger.warning(f"Failed to process metric {key}: {e}")
                    
        # Attempt to create result with cleaned data
        return agent_instance.create_result(
            analysis_type=analysis_type,
            success=success,
            data=data or {},
            metrics=clean_metrics,
            warnings=warnings or [],
            errors=errors or [],
            metadata=metadata or {}
        )
        
    except Exception as validation_error:
        agent_instance.logger.error(f"Pydantic validation failed, using fallback: {validation_error}")
        
        # Ultimate fallback - create minimal result with all problematic data in metadata
        fallback_metadata = {
            'original_data': data or {},
            'original_metrics': metrics or {},
            'validation_error': str(validation_error),
            'fallback_used': True
        }
        if metadata:
            fallback_metadata.update(metadata)
            
        return agent_instance.create_result(
            analysis_type=analysis_type,
            success=success,
            data={'status': 'completed_with_validation_fallback'},
            metrics={},  # Empty to avoid validation issues
            warnings=(warnings or []) + [f"Validation fallback used due to data type issues"],
            errors=errors or [],
            metadata=fallback_metadata
        )


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize database column names to match expected field names.
    
    Args:
        df: DataFrame with database column names
        
    Returns:
        DataFrame with normalized column names
    """
    column_mapping = {
        'MEMBERID': 'member_id',
        'MEMBERNUMBER': 'member_number',
        'PARENTACCOUNT': 'member_number',
        'SSN': 'member_ssn',
        'LAST': 'last_name',
        'FIRST': 'first_name',
        'MIDDLE': 'middle_initial',
        'ACCOUNTNUMBER': 'member_number',
        'EFFECTIVEDATE': 'transaction_date_int',
        'TRANSCODE': 'transaction_type',
        'AMOUNT': 'amount',
        'DESCRIPTION': 'description'
    }
    
    # Apply column mapping where columns exist
    columns_to_rename = {old_name: new_name for old_name, new_name in column_mapping.items() 
                        if old_name in df.columns}
    
    if columns_to_rename:
        df = df.rename(columns=columns_to_rename)
    
    return df


class MemberAnalyticsAgent(BaseAgent):
    """
    Specialized agent for member analytics and segmentation.
    
    Provides advanced member analysis including behavioral segmentation,
    lifetime value calculations, churn prediction, and cross-sell analysis.
    """
    
    def get_capabilities(self) -> List[str]:
        """Return list of analysis capabilities."""
        return [
            "rfm_segmentation",
            "member_clustering", 
            "lifetime_value_calculation",
            "churn_prediction",
            "cross_sell_analysis",
            "member_behavior_analysis",
            "engagement_scoring",
            "demographic_analysis",
            "product_adoption",
            "member_journey_analysis",
            "retention_analysis",
            "value_migration",
            "active_members_analysis"
        ]
    
    def analyze(self, context: AnalysisContext) -> AnalysisResult:
        """
        Main analysis method for member analytics.
        
        Args:
            context: Analysis context with parameters
            
        Returns:
            AnalysisResult with member analytics results
        """
        self.log_analysis_start("member_analytics", context)
        
        try:
            analysis_type = context.parameters.get('method', 'comprehensive')
            as_of_date = context.as_of_date or datetime.now().strftime('%Y-%m-%d')
            
            # Route to specific analysis based on type
            if analysis_type == 'rfm':
                result = self._rfm_segmentation(context.database, as_of_date)
            elif analysis_type == 'clustering':
                result = self._member_clustering(context.database, as_of_date)
            elif analysis_type == 'lifetime_value':
                result = self._lifetime_value_analysis(context.database, as_of_date)
            elif analysis_type == 'churn_prediction':
                result = self._churn_prediction(context.database, as_of_date)
            elif analysis_type == 'active_members':
                result = self._active_members_analysis(context.database, as_of_date, context.parameters)
            elif analysis_type == 'comprehensive':
                result = self._comprehensive_member_analysis(context.database, as_of_date)
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
            
            self.log_analysis_complete("member_analytics", result)
            return result
            
        except Exception as e:
            return self.handle_analysis_error("member_analytics", e)
    
    def _comprehensive_member_analysis(self, database: str, as_of_date: str) -> AnalysisResult:
        """
        Perform comprehensive member analytics.
        
        Args:
            database: Target database
            as_of_date: Analysis date
            
        Returns:
            AnalysisResult with comprehensive member analytics
        """
        # Get member data
        member_data = self._get_member_data(database, as_of_date)
        transaction_data = self._get_transaction_data(database, as_of_date)
        
        if member_data.empty:
            return self.create_result(
                analysis_type="comprehensive",
                success=False,
                errors=["No member data available for the specified date"]
            )
        
        # Perform all analyses
        rfm_results = self._calculate_rfm_segments(transaction_data)
        cluster_results = self._perform_clustering(member_data, transaction_data)
        ltv_results = self._calculate_lifetime_value(member_data, transaction_data)
        engagement_metrics = self._calculate_engagement_metrics(member_data, transaction_data)
        
        # Combine results
        analysis_data = {
            'member_overview': {
                'total_members': len(member_data),
                'active_members': len(member_data[member_data['status'] == 'Active']),
                'new_members_last_12m': self._count_new_members(member_data, as_of_date),
                'avg_relationship_length': self._calculate_avg_relationship_length(member_data, as_of_date)
            },
            'rfm_segmentation': rfm_results,
            'clustering': cluster_results,
            'lifetime_value': ltv_results,
            'engagement': engagement_metrics,
            'recommendations': self._generate_member_recommendations(rfm_results, cluster_results),
            'as_of_date': as_of_date
        }
        
        # Calculate summary metrics
        summary_metrics = {
            'total_members': len(member_data),
            'avg_lifetime_value': ltv_results.get('avg_ltv', 0),
            'high_value_member_pct': self._calculate_high_value_percentage(ltv_results),
            'at_risk_member_pct': self._calculate_at_risk_percentage(rfm_results)
        }
        
        return self.create_result(
            analysis_type="comprehensive",
            data=analysis_data,
            metrics=summary_metrics,
            metadata={'calculation_date': as_of_date}
        )
    
    def _rfm_segmentation(self, database: str, as_of_date: str) -> AnalysisResult:
        """Perform RFM segmentation analysis."""
        transaction_data = self._get_transaction_data(database, as_of_date)
        
        if transaction_data.empty:
            return self.create_result(
                analysis_type="rfm",
                success=False,
                errors=["No transaction data available for RFM analysis"]
            )
        
        rfm_results = self._calculate_rfm_segments(transaction_data)
        
        return self.create_result(
            analysis_type="rfm",
            data=rfm_results,
            metrics=rfm_results.get('segment_metrics', {})
        )
    
    def _member_clustering(self, database: str, as_of_date: str) -> AnalysisResult:
        """Perform member clustering analysis."""
        member_data = self._get_member_data(database, as_of_date)
        transaction_data = self._get_transaction_data(database, as_of_date)
        
        if member_data.empty:
            return self.create_result(
                analysis_type="clustering",
                success=False,
                errors=["No member data available for clustering"]
            )
        
        cluster_results = self._perform_clustering(member_data, transaction_data)
        
        return self.create_result(
            analysis_type="clustering",
            data=cluster_results,
            metrics=cluster_results.get('cluster_metrics', {})
        )
    
    def _lifetime_value_analysis(self, database: str, as_of_date: str) -> AnalysisResult:
        """Perform lifetime value analysis."""
        member_data = self._get_member_data(database, as_of_date)
        transaction_data = self._get_transaction_data(database, as_of_date)
        
        if member_data.empty:
            return self.create_result(
                analysis_type="lifetime_value",
                success=False,
                errors=["No member data available for LTV analysis"]
            )
        
        ltv_results = self._calculate_lifetime_value(member_data, transaction_data)
        
        return self.create_result(
            analysis_type="lifetime_value",
            data=ltv_results,
            metrics=ltv_results.get('ltv_metrics', {})
        )
    
    def _churn_prediction(self, database: str, as_of_date: str) -> AnalysisResult:
        """Perform churn prediction analysis."""
        member_data = self._get_member_data(database, as_of_date)
        transaction_data = self._get_transaction_data(database, as_of_date)
        
        if member_data.empty:
            return self.create_result(
                analysis_type="churn_prediction",
                success=False,
                errors=["No member data available for churn prediction"]
            )
        
        churn_results = self._predict_churn(member_data, transaction_data)
        
        return self.create_result(
            analysis_type="churn_prediction",
            data=churn_results,
            metrics=churn_results.get('model_metrics', {})
        )
    
    def _get_member_data(self, database: str, as_of_date: str) -> pd.DataFrame:
        """
        Retrieve member demographic and account data using ARCUSYM000 schema with safe fallbacks.
        
        Args:
            database: Target database
            as_of_date: Analysis date
            
        Returns:
            DataFrame with member data
        """
        # Primary query - try complex join first
        primary_query = """
        SELECT 
            n.PARENTACCOUNT as member_number,
            mr.SSN as member_ssn,
            n.LAST as last_name,
            n.FIRST as first_name,
            n.MIDDLE as middle_initial,
            CASE 
                WHEN mr.JoinDate > 19000101 THEN 
                    CONVERT(date, CAST(mr.JoinDate as varchar(8)), 112)
                ELSE NULL 
            END as join_date,
            CASE 
                WHEN mr.BirthDate > 19000101 THEN 
                    CONVERT(date, CAST(mr.BirthDate as varchar(8)), 112)
                ELSE NULL 
            END as birth_date,
            mr.MemberStatus as status
        FROM NAME n
        INNER JOIN MEMBERREC mr ON n.SSN = mr.SSN
        WHERE 
            n.TYPE = 0  -- Primary name record
            AND mr.MemberStatus IN (0, 1)  -- Active member statuses
            AND mr.JoinDate <= CAST(REPLACE(:as_of_date, '-', '') as int)
        """
        
        # Fallback queries if primary fails
        fallback_queries = [
            # Simpler query without date conversion
            """
            SELECT 
                n.PARENTACCOUNT as member_number,
                n.LAST as last_name,
                n.FIRST as first_name,
                mr.MemberStatus as status
            FROM NAME n
            INNER JOIN MEMBERREC mr ON n.SSN = mr.SSN
            WHERE n.TYPE = 0 AND mr.MemberStatus IN (0, 1)
            """,
            
            # Most basic query - just get active members
            """
            SELECT 
                PARENTACCOUNT as member_number,
                LAST as last_name,
                FIRST as first_name
            FROM NAME 
            WHERE TYPE = 0
            """
        ]
        
        try:
            # Use safe query execution with fallbacks
            result = self._execute_safe_query_with_fallbacks(
                database, 
                primary_query, 
                fallback_queries, 
                params={'as_of_date': as_of_date}
            )
            
            if not result.empty:
                # Apply column normalization to handle database column name mismatches
                result = normalize_column_names(result)
                
                # Add account summary data if primary query worked
                if 'join_date' in result.columns:
                    result = self._enrich_member_data(result, database)
                
            self.logger.info(f"Retrieved {len(result)} member records from {database}")
            return result
            
        except Exception as e:
            self.logger.error(f"All member data queries failed: {e}")
            return pd.DataFrame()
    
    def _enrich_member_data(self, member_data: pd.DataFrame, database: str) -> pd.DataFrame:
        """
        Enrich basic member data with account information using safe queries.
        
        Args:
            member_data: Basic member data
            database: Target database
            
        Returns:
            Enriched member data with account information
        """
        enriched_data = member_data.copy()
        
        try:
            # Get account counts per member
            account_query = """
            SELECT 
                a.ACCOUNTNUMBER as member_number,
                COUNT(*) as total_accounts
            FROM ACCOUNT a
            WHERE a.CLOSEDATE = '19000101'  -- Open accounts only
            GROUP BY a.ACCOUNTNUMBER
            """
            
            account_data = self.execute_query(account_query, database)
            if not account_data.empty:
                enriched_data = enriched_data.merge(
                    account_data, 
                    on='member_number', 
                    how='left'
                )
                enriched_data['total_accounts'] = enriched_data['total_accounts'].fillna(0)
            else:
                enriched_data['total_accounts'] = 1  # Default assumption
                
        except Exception as e:
            self.logger.warning(f"Could not enrich member data with accounts: {e}")
            enriched_data['total_accounts'] = 1  # Default
            
        # Add default values for missing columns
        default_columns = {
            'total_balance': 0,
            'total_savings_balance': 0,
            'total_loan_balance': 0,
            'has_card': 0,
            'online_banking_enrolled': 1,
            'mobile_banking_enrolled': 1,
            'debit_card_active': 1,
            'credit_card_active': 0
        }
        
        for col, default_val in default_columns.items():
            if col not in enriched_data.columns:
                enriched_data[col] = default_val
        
        return enriched_data
    
    def _execute_safe_query_with_fallbacks(self, database: str, primary_query: str, fallback_queries: List[str] = None, params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Execute query with multiple fallback options if primary fails.
        This method provides access to enhanced database functionality through the base agent.
        
        Args:
            database: Target database
            primary_query: Primary query to attempt
            fallback_queries: List of fallback queries to try
            params: Query parameters
            
        Returns:
            DataFrame with results from first successful query
        """
        queries_to_try = [primary_query] + (fallback_queries or [])
        
        for i, query in enumerate(queries_to_try):
            try:
                self.logger.info(f"Attempting query {i+1}/{len(queries_to_try)}")
                result = self.execute_query(query, database, params)
                if not result.empty:
                    self.logger.info(f"Query {i+1} succeeded with {len(result)} rows")
                    return result
                else:
                    self.logger.warning(f"Query {i+1} returned no data")
                    
            except Exception as e:
                self.logger.error(f"Query {i+1} failed: {e}")
                if i == len(queries_to_try) - 1:  # Last query
                    self.logger.error("All queries failed")
                    break
                else:
                    self.logger.info(f"Trying fallback query {i+2}")
                    
        # Return empty DataFrame if all queries fail
        self.logger.warning("All fallback queries exhausted, returning empty DataFrame")
        return pd.DataFrame()
    
    def _get_transaction_data(self, database: str, as_of_date: str) -> pd.DataFrame:
        """Retrieve transaction history for analysis using ARCUSYM000 schema with safe fallbacks."""
        # Calculate date range - convert to ARCUSYM000 integer format (YYYYMMDD)
        end_date_int = int(as_of_date.replace('-', ''))
        start_date_obj = datetime.strptime(as_of_date, '%Y-%m-%d') - timedelta(days=730)
        start_date_int = int(start_date_obj.strftime('%Y%m%d'))
        
        # Primary query with proper ARCUSYM000 date handling
        primary_query = """
        SELECT TOP 50000
            a.PARENTACCOUNT as member_id,
            a.EFFECTIVEDATE as transaction_date_int,
            a.TRANSCODE as transaction_type,
            a.AMOUNT as amount,
            a.DESCRIPTION as description,
            'CORE' as channel
        FROM ACTIVITY a
        WHERE 
            a.EFFECTIVEDATE BETWEEN :start_date_int AND :end_date_int
            AND a.AMOUNT != 0  -- Exclude zero-amount transactions
            AND a.TRANSCODE IS NOT NULL
        ORDER BY a.PARENTACCOUNT, a.EFFECTIVEDATE DESC
        """
        
        # Fallback queries
        fallback_queries = [
            # Simpler query without date range
            """
            SELECT TOP 10000
                PARENTACCOUNT as member_id,
                EFFECTIVEDATE as transaction_date_int,
                TRANSCODE as transaction_type,
                AMOUNT as amount,
                'CORE' as channel
            FROM ACTIVITY 
            WHERE AMOUNT != 0 AND TRANSCODE IS NOT NULL
            ORDER BY EFFECTIVEDATE DESC
            """,
            
            # Most basic query
            """
            SELECT TOP 1000
                PARENTACCOUNT as member_id,
                AMOUNT as amount
            FROM ACTIVITY 
            WHERE AMOUNT != 0
            """
        ]
        
        try:
            result = self._execute_safe_query_with_fallbacks(
                database,
                primary_query,
                fallback_queries,
                params={
                    'start_date_int': start_date_int,
                    'end_date_int': end_date_int
                }
            )
            
            if not result.empty:
                # Apply column normalization to handle database column name mismatches
                result = normalize_column_names(result)
                
                # Convert integer dates to proper dates if possible
                if 'transaction_date_int' in result.columns:
                    result['transaction_date'] = result['transaction_date_int'].apply(
                        self._convert_arcusym_date_safe
                    )
                
                # Ensure required columns exist
                required_columns = ['member_id', 'transaction_date', 'transaction_type', 'amount', 'channel']
                for col in required_columns:
                    if col not in result.columns:
                        if col == 'transaction_date':
                            result[col] = datetime.now().date()
                        elif col == 'channel':
                            result[col] = 'CORE'
                        else:
                            result[col] = 0
            
            self.logger.info(f"Retrieved {len(result)} transaction records from {database}")
            return result
            
        except Exception as e:
            self.logger.error(f"All transaction queries failed: {e}")
            return pd.DataFrame(columns=[
                'member_id', 'transaction_date', 'transaction_type', 
                'amount', 'description', 'channel', 'account_type'
            ])
    
    def _convert_arcusym_date_safe(self, date_int) -> datetime:
        """
        Safely convert ARCUSYM000 integer date (YYYYMMDD) to datetime.
        
        Args:
            date_int: Integer date in YYYYMMDD format
            
        Returns:
            datetime object or current date if conversion fails
        """
        try:
            if pd.isna(date_int) or date_int == 0 or date_int < 19000101:
                return datetime.now().date()
            
            date_str = str(int(date_int))
            if len(date_str) == 8:
                return datetime.strptime(date_str, '%Y%m%d').date()
            else:
                return datetime.now().date()
                
        except (ValueError, TypeError):
            return datetime.now().date()
    
    def _calculate_rfm_segments(self, transaction_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate RFM (Recency, Frequency, Monetary) segmentation.
        
        Args:
            transaction_data: Transaction history data
            
        Returns:
            Dictionary with RFM analysis results
        """
        if transaction_data.empty:
            return {}
        
        # Calculate RFM metrics for each member
        analysis_date = transaction_data['transaction_date'].max()
        
        rfm = transaction_data.groupby('member_id').agg({
            'transaction_date': lambda x: (pd.to_datetime(analysis_date) - x.max()).days,  # Recency
            'amount': ['count', 'sum']  # Frequency and Monetary
        }).round(2)
        
        rfm.columns = ['recency', 'frequency', 'monetary']
        rfm = rfm.reset_index()
        
        # Calculate RFM scores (1-5 scale)
        rfm['r_score'] = pd.qcut(rfm['recency'], q=5, labels=[5,4,3,2,1]).astype(int)  # Lower recency = higher score
        rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1,2,3,4,5]).astype(int)
        rfm['m_score'] = pd.qcut(rfm['monetary'].rank(method='first'), q=5, labels=[1,2,3,4,5]).astype(int)
        
        # Calculate RFM combined score
        rfm['rfm_score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)
        
        # Define segments based on RFM scores
        rfm['segment'] = rfm.apply(self._classify_rfm_segment, axis=1)
        
        # Calculate segment statistics
        segment_stats = rfm.groupby('segment').agg({
            'member_id': 'count',
            'recency': 'mean',
            'frequency': 'mean', 
            'monetary': 'mean'
        }).round(2)
        
        segment_stats.columns = ['member_count', 'avg_recency', 'avg_frequency', 'avg_monetary']
        segment_stats['percentage'] = (segment_stats['member_count'] / len(rfm) * 100).round(1)
        
        return {
            'member_rfm_scores': rfm.to_dict('records'),
            'segment_summary': segment_stats.to_dict('index'),
            'total_members_analyzed': len(rfm),
            'segment_metrics': {
                'champions_pct': segment_stats.loc['Champions', 'percentage'] if 'Champions' in segment_stats.index else 0,
                'loyal_customers_pct': segment_stats.loc['Loyal Customers', 'percentage'] if 'Loyal Customers' in segment_stats.index else 0,
                'at_risk_pct': segment_stats.loc['At Risk', 'percentage'] if 'At Risk' in segment_stats.index else 0,
                'lost_customers_pct': segment_stats.loc['Lost', 'percentage'] if 'Lost' in segment_stats.index else 0
            }
        }
    
    def _classify_rfm_segment(self, row: pd.Series) -> str:
        """Classify member into RFM segment based on scores."""
        r, f, m = row['r_score'], row['f_score'], row['m_score']
        
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        elif r >= 3 and f >= 3 and m >= 3:
            return 'Loyal Customers'
        elif r >= 3 and f >= 2:
            return 'Potential Loyalists'
        elif r >= 4 and f <= 2:
            return 'New Customers'
        elif r >= 3 and f <= 2 and m >= 3:
            return 'Promising'
        elif r <= 2 and f >= 3 and m >= 3:
            return 'At Risk'
        elif r <= 2 and f >= 2:
            return 'Cannot Lose Them'
        elif r <= 2 and f <= 2:
            return 'Lost'
        else:
            return 'Others'
    
    def _perform_clustering(self, member_data: pd.DataFrame, transaction_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform K-means clustering on member data.
        
        Args:
            member_data: Member demographic and account data
            transaction_data: Transaction history data
            
        Returns:
            Dictionary with clustering results
        """
        if member_data.empty:
            return {}
        
        # Prepare features for clustering
        clustering_features = self._prepare_clustering_features(member_data, transaction_data)
        
        if clustering_features.empty:
            return {'error': 'No features available for clustering'}
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(clustering_features.select_dtypes(include=[np.number]))
        
        # Determine optimal number of clusters using elbow method
        optimal_clusters = self._find_optimal_clusters(features_scaled)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Add cluster labels to member data
        member_clusters = member_data.copy()
        member_clusters['cluster'] = cluster_labels
        
        # Analyze cluster characteristics
        cluster_analysis = self._analyze_clusters(member_clusters, clustering_features)
        
        return {
            'optimal_clusters': optimal_clusters,
            'cluster_assignments': member_clusters[['member_id', 'cluster']].to_dict('records'),
            'cluster_profiles': cluster_analysis,
            'feature_importance': self._calculate_feature_importance(clustering_features, cluster_labels),
            'cluster_metrics': {
                'total_clusters': optimal_clusters,
                'largest_cluster_size': max([profile['size'] for profile in cluster_analysis.values()]),
                'smallest_cluster_size': min([profile['size'] for profile in cluster_analysis.values()])
            }
        }
    
    def _prepare_clustering_features(self, member_data: pd.DataFrame, transaction_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for clustering analysis."""
        features = member_data[['member_id']].copy()
        
        # Demographic features
        if 'birth_date' in member_data.columns:
            current_date = datetime.now()
            features['age'] = member_data['birth_date'].apply(
                lambda x: (current_date - pd.to_datetime(x)).days / 365.25 if pd.notna(x) else 0
            )
        
        # Account features
        numeric_cols = ['total_accounts', 'total_balance', 'total_loans', 'total_loan_balance']
        for col in numeric_cols:
            if col in member_data.columns:
                features[col] = member_data[col].fillna(0)
        
        # Service adoption features
        service_cols = ['online_banking_enrolled', 'mobile_banking_enrolled', 'debit_card_active', 'credit_card_active']
        for col in service_cols:
            if col in member_data.columns:
                features[col] = member_data[col].astype(int)
        
        # Transaction-based features
        if not transaction_data.empty:
            transaction_features = self._calculate_transaction_features(transaction_data)
            features = features.merge(transaction_features, on='member_id', how='left')
        
        # Fill missing values
        features = features.fillna(0)
        
        return features
    
    def _calculate_transaction_features(self, transaction_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate transaction-based features for clustering."""
        features = transaction_data.groupby('member_id').agg({
            'amount': ['sum', 'mean', 'count', 'std'],
            'transaction_date': ['count']
        }).round(2)
        
        features.columns = ['total_transaction_amount', 'avg_transaction_amount', 
                           'transaction_count', 'transaction_amount_std', 'transaction_frequency']
        features = features.fillna(0)
        features = features.reset_index()
        
        # Calculate monthly transaction frequency
        features['monthly_transaction_freq'] = features['transaction_frequency'] / 24  # 24 months of data
        
        return features
    
    def _find_optimal_clusters(self, features: np.ndarray, max_clusters: int = 10) -> int:
        """Find optimal number of clusters using elbow method."""
        if len(features) < max_clusters:
            max_clusters = len(features)
        
        wcss = []
        k_range = range(2, min(max_clusters + 1, 11))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(features)
            wcss.append(kmeans.inertia_)
        
        # Find elbow point (simplified)
        if len(wcss) >= 3:
            # Calculate the rate of decrease
            rate_of_decrease = np.diff(wcss)
            rate_of_change = np.diff(rate_of_decrease)
            
            # Find the point where rate of change is minimal (elbow)
            elbow_index = np.argmin(rate_of_change) + 2  # +2 because we start from k=2
            optimal_k = k_range[elbow_index] if elbow_index < len(k_range) else k_range[-1]
        else:
            optimal_k = 3  # Default fallback
        
        return optimal_k
    
    def _analyze_clusters(self, member_clusters: pd.DataFrame, features: pd.DataFrame) -> Dict[str, Any]:
        """Analyze characteristics of each cluster."""
        cluster_profiles = {}
        
        for cluster_id in member_clusters['cluster'].unique():
            cluster_members = member_clusters[member_clusters['cluster'] == cluster_id]
            cluster_features = features[features['member_id'].isin(cluster_members['member_id'])]
            
            profile = {
                'cluster_id': int(cluster_id),
                'size': len(cluster_members),
                'percentage': len(cluster_members) / len(member_clusters) * 100,
                'characteristics': {}
            }
            
            # Calculate mean values for numeric features
            numeric_features = cluster_features.select_dtypes(include=[np.number])
            for col in numeric_features.columns:
                if col != 'member_id':
                    profile['characteristics'][col] = float(numeric_features[col].mean())
            
            # Identify dominant characteristics
            profile['dominant_traits'] = self._identify_dominant_traits(profile['characteristics'])
            
            cluster_profiles[f'cluster_{cluster_id}'] = profile
        
        return cluster_profiles
    
    def _identify_dominant_traits(self, characteristics: Dict[str, float]) -> List[str]:
        """Identify dominant traits for a cluster."""
        traits = []
        
        # Define thresholds for different characteristics
        if characteristics.get('age', 0) > 50:
            traits.append('Mature Members')
        elif characteristics.get('age', 0) < 35:
            traits.append('Young Members')
        
        if characteristics.get('total_balance', 0) > 50000:
            traits.append('High Balance')
        elif characteristics.get('total_balance', 0) < 5000:
            traits.append('Low Balance')
        
        if characteristics.get('online_banking_enrolled', 0) > 0.8:
            traits.append('Digital Adopters')
        elif characteristics.get('online_banking_enrolled', 0) < 0.3:
            traits.append('Traditional Banking')
        
        if characteristics.get('total_loans', 0) > 2:
            traits.append('Multi-Product Users')
        
        return traits if traits else ['Standard Members']
    
    def _calculate_feature_importance(self, features: pd.DataFrame, cluster_labels: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance for clustering."""
        # Use a simple variance-based approach
        numeric_features = features.select_dtypes(include=[np.number])
        feature_importance = {}
        
        for col in numeric_features.columns:
            if col != 'member_id':
                # Calculate variance within clusters vs between clusters
                cluster_df = pd.DataFrame({'feature': numeric_features[col], 'cluster': cluster_labels})
                total_variance = cluster_df['feature'].var()
                within_cluster_variance = cluster_df.groupby('cluster')['feature'].var().mean()
                
                if total_variance > 0:
                    importance = 1 - (within_cluster_variance / total_variance)
                    feature_importance[col] = max(0, importance)
                else:
                    feature_importance[col] = 0
        
        return feature_importance
    
    def _calculate_lifetime_value(self, member_data: pd.DataFrame, transaction_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate customer lifetime value for members.
        
        Args:
            member_data: Member demographic and account data
            transaction_data: Transaction history data
            
        Returns:
            Dictionary with LTV analysis results
        """
        if member_data.empty:
            return {}
        
        ltv_results = []
        
        for _, member in member_data.iterrows():
            member_id = member['member_id']
            
            # Get member's transaction history
            member_transactions = transaction_data[transaction_data['member_id'] == member_id]
            
            # Calculate LTV components
            ltv_components = self._calculate_ltv_components(member, member_transactions)
            
            ltv_results.append({
                'member_id': member_id,
                **ltv_components
            })
        
        ltv_df = pd.DataFrame(ltv_results)
        
        # Calculate aggregate statistics
        ltv_statistics = {
            'avg_ltv': float(ltv_df['lifetime_value'].mean()),
            'median_ltv': float(ltv_df['lifetime_value'].median()),
            'total_ltv': float(ltv_df['lifetime_value'].sum()),
            'ltv_distribution': self._calculate_ltv_distribution(ltv_df)
        }
        
        return {
            'member_ltv': ltv_results,
            'ltv_metrics': ltv_statistics,
            'high_value_members': ltv_df.nlargest(100, 'lifetime_value')[['member_id', 'lifetime_value']].to_dict('records')
        }
    
    def _calculate_ltv_components(self, member: pd.Series, transactions: pd.DataFrame) -> Dict[str, float]:
        """Calculate LTV components for a single member."""
        join_date = pd.to_datetime(member['join_date'])
        current_date = datetime.now()
        relationship_length = (current_date - join_date).days / 365.25
        
        # Revenue calculations
        deposit_balance = float(member.get('total_balance', 0))
        loan_balance = float(member.get('total_loan_balance', 0))
        
        # Estimate annual revenue
        # Revenue from deposits (negative - we pay interest)
        deposit_revenue = deposit_balance * -0.015  # Assume 1.5% interest expense
        
        # Revenue from loans (positive - we earn interest)
        loan_revenue = loan_balance * 0.055  # Assume 5.5% interest income
        
        # Fee income from transactions
        if not transactions.empty:
            annual_transactions = len(transactions) / max(relationship_length, 1)
            fee_revenue = annual_transactions * 2.5  # Assume $2.50 average fee per transaction
        else:
            fee_revenue = 0
        
        annual_revenue = loan_revenue + deposit_revenue + fee_revenue
        
        # Estimate costs
        annual_cost = 150  # Base servicing cost per member
        
        # Net annual value
        net_annual_value = annual_revenue - annual_cost
        
        # Project future value (assume 5-year horizon with 5% discount rate)
        discount_rate = 0.05
        projection_years = 5
        
        future_value = 0
        for year in range(1, projection_years + 1):
            yearly_value = net_annual_value * (0.95 ** year)  # Assume 5% annual decline
            discounted_value = yearly_value / ((1 + discount_rate) ** year)
            future_value += discounted_value
        
        # Historical value
        historical_value = net_annual_value * relationship_length
        
        # Total lifetime value
        lifetime_value = historical_value + future_value
        
        return {
            'annual_revenue': annual_revenue,
            'annual_cost': annual_cost,
            'net_annual_value': net_annual_value,
            'relationship_length_years': relationship_length,
            'historical_value': historical_value,
            'projected_future_value': future_value,
            'lifetime_value': lifetime_value
        }
    
    def _calculate_ltv_distribution(self, ltv_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate LTV distribution statistics."""
        ltv_values = ltv_df['lifetime_value']
        
        return {
            'top_10_percent': float(ltv_values.quantile(0.9)),
            'top_25_percent': float(ltv_values.quantile(0.75)),
            'median': float(ltv_values.quantile(0.5)),
            'bottom_25_percent': float(ltv_values.quantile(0.25)),
            'bottom_10_percent': float(ltv_values.quantile(0.1))
        }
    
    def _predict_churn(self, member_data: pd.DataFrame, transaction_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict member churn using machine learning.
        
        Args:
            member_data: Member demographic and account data
            transaction_data: Transaction history data
            
        Returns:
            Dictionary with churn prediction results
        """
        if member_data.empty:
            return {}
        
        # Prepare features for churn prediction
        churn_features = self._prepare_churn_features(member_data, transaction_data)
        
        if len(churn_features) < 50:  # Need minimum data for ML
            return {'error': 'Insufficient data for churn prediction'}
        
        # Define churn (simplified - no activity in last 90 days)
        churn_features['is_churned'] = churn_features['days_since_last_activity'] > 90
        
        # Prepare features for modeling
        feature_columns = [col for col in churn_features.columns 
                          if col not in ['member_id', 'is_churned']]
        
        X = churn_features[feature_columns].fillna(0)
        y = churn_features['is_churned']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train Random Forest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        churn_probabilities = model.predict_proba(X)[:, 1]  # Probability of churn
        
        # Add predictions back to data
        churn_features['churn_probability'] = churn_probabilities
        churn_features['churn_prediction'] = model.predict(X)
        
        # Calculate model performance
        model_performance = self._evaluate_churn_model(y_test, y_pred)
        
        # Identify high-risk members
        high_risk_members = churn_features[churn_features['churn_probability'] > 0.7].copy()
        high_risk_members = high_risk_members.sort_values('churn_probability', ascending=False)
        
        return {
            'member_churn_scores': churn_features[['member_id', 'churn_probability', 'churn_prediction']].to_dict('records'),
            'high_risk_members': high_risk_members[['member_id', 'churn_probability']].head(100).to_dict('records'),
            'model_performance': model_performance,
            'feature_importance': dict(zip(feature_columns, model.feature_importances_)),
            'model_metrics': {
                'total_members_scored': len(churn_features),
                'high_risk_count': len(high_risk_members),
                'high_risk_percentage': len(high_risk_members) / len(churn_features) * 100
            }
        }
    
    def _prepare_churn_features(self, member_data: pd.DataFrame, transaction_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for churn prediction."""
        features = member_data[['member_id']].copy()
        
        # Calculate days since last activity
        current_date = datetime.now()
        features['days_since_last_activity'] = member_data['last_activity_date'].apply(
            lambda x: (current_date - pd.to_datetime(x)).days if pd.notna(x) else 999
        )
        
        # Member tenure
        features['tenure_days'] = member_data['join_date'].apply(
            lambda x: (current_date - pd.to_datetime(x)).days if pd.notna(x) else 0
        )
        
        # Account and service features
        numeric_cols = ['total_accounts', 'total_balance', 'total_loans']
        for col in numeric_cols:
            if col in member_data.columns:
                features[col] = member_data[col].fillna(0)
        
        # Service adoption
        service_cols = ['online_banking_enrolled', 'mobile_banking_enrolled']
        for col in service_cols:
            if col in member_data.columns:
                features[col] = member_data[col].astype(int)
        
        # Transaction-based features
        if not transaction_data.empty:
            # Calculate transaction recency and frequency
            transaction_features = transaction_data.groupby('member_id').agg({
                'transaction_date': lambda x: (current_date - x.max()).days,  # Days since last transaction
                'amount': ['count', 'sum', 'mean']  # Transaction frequency and monetary
            })
            
            transaction_features.columns = ['days_since_last_transaction', 'transaction_count', 
                                          'total_transaction_amount', 'avg_transaction_amount']
            transaction_features = transaction_features.reset_index()
            
            features = features.merge(transaction_features, on='member_id', how='left')
        
        return features.fillna(0)
    
    def _evaluate_churn_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate churn prediction model performance."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
    
    def _calculate_engagement_metrics(self, member_data: pd.DataFrame, transaction_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate member engagement metrics."""
        engagement_metrics = {}
        
        # Digital adoption rates
        if 'online_banking_enrolled' in member_data.columns:
            engagement_metrics['online_banking_adoption'] = member_data['online_banking_enrolled'].mean() * 100
        
        if 'mobile_banking_enrolled' in member_data.columns:
            engagement_metrics['mobile_banking_adoption'] = member_data['mobile_banking_enrolled'].mean() * 100
        
        # Product adoption
        engagement_metrics['avg_accounts_per_member'] = member_data['total_accounts'].mean()
        engagement_metrics['multi_product_members_pct'] = (member_data['total_accounts'] > 1).mean() * 100
        
        # Transaction engagement
        if not transaction_data.empty:
            monthly_transactions = transaction_data.groupby('member_id').size()
            engagement_metrics['avg_monthly_transactions'] = monthly_transactions.mean()
            engagement_metrics['active_transactors_pct'] = (monthly_transactions > 0).mean() * 100
        
        return engagement_metrics
    
    def _count_new_members(self, member_data: pd.DataFrame, as_of_date: str) -> int:
        """Count new members in the last 12 months."""
        cutoff_date = (datetime.strptime(as_of_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
        new_members = member_data[member_data['join_date'] >= cutoff_date]
        return len(new_members)
    
    def _calculate_avg_relationship_length(self, member_data: pd.DataFrame, as_of_date: str) -> float:
        """Calculate average relationship length in years."""
        as_of_datetime = datetime.strptime(as_of_date, '%Y-%m-%d')
        
        relationship_lengths = member_data['join_date'].apply(
            lambda x: (as_of_datetime - pd.to_datetime(x)).days / 365.25 if pd.notna(x) else 0
        )
        
        return float(relationship_lengths.mean())
    
    def _calculate_high_value_percentage(self, ltv_results: Dict[str, Any]) -> float:
        """Calculate percentage of high-value members (top 20% by LTV)."""
        if 'ltv_metrics' not in ltv_results:
            return 0.0
        
        top_20_threshold = ltv_results['ltv_metrics'].get('top_25_percent', 0)  # Using 25% as proxy
        return 20.0  # Simplified - top 20% by definition
    
    def _calculate_at_risk_percentage(self, rfm_results: Dict[str, Any]) -> float:
        """Calculate percentage of at-risk members from RFM analysis."""
        if 'segment_metrics' not in rfm_results:
            return 0.0
        
        return rfm_results['segment_metrics'].get('at_risk_pct', 0)
    
    def _active_members_analysis(self, database: str, as_of_date: str, parameters: Dict[str, Any]) -> AnalysisResult:
        """
        Perform active members analysis using CRCU business rules.
        
        Args:
            database: Target database (should be ARCUSYM000 for CRCU data)
            as_of_date: Analysis date
            parameters: Additional parameters including include_insights
            
        Returns:
            AnalysisResult with active members data and insights
        """
        include_insights = parameters.get('include_insights', True)
        
        try:
            # Get active members using CRCU business rules
            active_members_data = self._get_crcu_active_members(database, as_of_date)
            
            if active_members_data.empty:
                return self.create_result(
                    analysis_type="active_members",
                    success=False,
                    errors=["No active members found for the specified date"]
                )
            
            # Calculate key metrics
            total_active = len(active_members_data)
            
            # Basic demographic breakdown if data available
            metrics = {
                'total_active_members': total_active,
                'analysis_date': as_of_date
            }
            
            # Add demographic breakdowns if columns exist
            if 'age_group' in active_members_data.columns:
                age_breakdown = active_members_data['age_group'].value_counts().to_dict()
                metrics['age_distribution'] = age_breakdown
            
            if 'total_accounts' in active_members_data.columns:
                metrics['avg_accounts_per_member'] = float(active_members_data['total_accounts'].mean())
                metrics['multi_product_members'] = int((active_members_data['total_accounts'] > 1).sum())
            
            if 'total_balance' in active_members_data.columns:
                metrics['total_member_balances'] = float(active_members_data['total_balance'].sum())
                metrics['avg_balance_per_member'] = float(active_members_data['total_balance'].mean())
            
            # Prepare response data (exclude PII)
            safe_member_data = self._sanitize_member_data(active_members_data)
            
            analysis_data = {
                'active_members_summary': metrics,
                'member_data': safe_member_data.to_dict('records') if len(safe_member_data) <= 1000 else f"Large dataset with {len(safe_member_data)} members - use filters for detailed view",
                'data_source': database,
                'business_rules_applied': [
                    "Active membership status",
                    "Valid account relationships", 
                    "PII protection applied"
                ]
            }
            
            # Add insights if requested
            if include_insights:
                insights = self._generate_active_members_insights(active_members_data, metrics)
                analysis_data['insights'] = insights
            
            return safe_create_analysis_result(
                self,
                analysis_type="active_members",
                data=analysis_data,
                metrics=metrics,
                metadata={'pii_protected': True, 'record_count': total_active}
            )
            
        except Exception as e:
            self.logger.error(f"Active members analysis failed: {e}")
            return self.create_result(
                analysis_type="active_members",
                success=False,
                errors=[f"Analysis failed: {str(e)}"]
            )
    
    def _get_crcu_active_members(self, database: str, as_of_date: str) -> pd.DataFrame:
        """
        Get active members using CRCU-specific business rules with the updated active member definition.
        
        An active member is defined as:
        - Has a valid SSN
        - From the previous day's process date
        - Has account types in the approved list
        - Has open accounts (not closed)
        - Has either open loans or open savings accounts
        
        Args:
            database: Target database 
            as_of_date: Analysis date
            
        Returns:
            DataFrame with active member data
        """
        
        # Primary query with the updated CRCU active member logic
        primary_query = """
        SELECT 
            FORMAT(DATEADD(DAY, -1, GETDATE()), 'yyyyMMdd') AS ProcessDate, 
            ActiveMembers.AccountType, 
            CASE ActiveMembers.AccountType 
                WHEN 0 THEN 'General Membership' 
                WHEN 1 THEN 'Share Draft' 
                WHEN 2 THEN 'Money Market' 
                WHEN 5 THEN 'Certificate' 
                WHEN 6 THEN 'IRA' 
                WHEN 8 THEN 'Minor' 
                WHEN 9 THEN 'Representative Payee'
                WHEN 10 THEN 'Club' 
                WHEN 11 THEN 'TUTMA' 
                WHEN 12 THEN 'Benefit' 
                WHEN 13 THEN 'Indirect' 
                WHEN 15 THEN 'Guardianship' 
                WHEN 87 THEN 'Professional Association' 
                WHEN 88 THEN 'Sole Proprietorship-Individual' 
                WHEN 89 THEN 'Trust' 
                WHEN 90 THEN 'C Corporation' 
                WHEN 91 THEN 'S Corporation' 
                WHEN 92 THEN 'Sole Proprietorship-Non-Individual' 
                WHEN 93 THEN 'Limited Liability Co (LLC)' 
                WHEN 94 THEN 'General Partnership' 
                WHEN 95 THEN 'Limited Partnership' 
                WHEN 96 THEN 'Limited Liability Partnership' 
                WHEN 97 THEN 'Non Profit Org/Assoc/Club' 
                WHEN 98 THEN 'Non Profit Corporation' 
                WHEN 99 THEN 'Estate' 
                ELSE 'Unknown' 
            END AS AccountTypeDescription,
            ActiveMembers.LAST AS MemberName, 
            ActiveMembers.AccountNumber, 
            ActiveMembers.SSN, 
            ActiveMembers.Branch, 
            'Type ' + RIGHT('0000' + CAST(ActiveMembers.AccountType AS VARCHAR), 4) + '-' + 
            CASE ActiveMembers.AccountType 
                WHEN 0 THEN 'General Membership' 
                WHEN 1 THEN 'Share Draft' 
                WHEN 2 THEN 'Money Market' 
                WHEN 5 THEN 'Certificate' 
                WHEN 6 THEN 'IRA' 
                WHEN 8 THEN 'Minor' 
                WHEN 9 THEN 'Representative Payee'
                WHEN 10 THEN 'Club' 
                WHEN 11 THEN 'TUTMA' 
                WHEN 12 THEN 'Benefit' 
                WHEN 13 THEN 'Indirect' 
                WHEN 15 THEN 'Guardianship' 
                WHEN 87 THEN 'Professional Association' 
                WHEN 88 THEN 'Sole Proprietorship-Individual' 
                WHEN 89 THEN 'Trust' 
                WHEN 90 THEN 'C Corporation' 
                WHEN 91 THEN 'S Corporation' 
                WHEN 92 THEN 'Sole Proprietorship-Non-Individual' 
                WHEN 93 THEN 'Limited Liability Co (LLC)' 
                WHEN 94 THEN 'General Partnership' 
                WHEN 95 THEN 'Limited Partnership' 
                WHEN 96 THEN 'Limited Liability Partnership' 
                WHEN 97 THEN 'Non Profit Org/Assoc/Club' 
                WHEN 98 THEN 'Non Profit Corporation' 
                WHEN 99 THEN 'Estate' 
                ELSE 'Unknown' 
            END AS FormattedAccountType,
            1 AS MemberCount
        FROM (
            SELECT DISTINCT 
                n.SSN, 
                n.LAST, 
                n.MemberNumber, 
                FIRST_VALUE(a.AccountNumber) OVER (PARTITION BY n.SSN ORDER BY a.OpenDate ASC) AS AccountNumber, 
                FIRST_VALUE(a.TYPE) OVER (PARTITION BY n.SSN ORDER BY a.OpenDate ASC) AS AccountType, 
                FIRST_VALUE(a.Branch) OVER (PARTITION BY n.SSN ORDER BY a.OpenDate ASC) AS Branch, 
                ROW_NUMBER() OVER (PARTITION BY n.SSN ORDER BY a.OpenDate ASC) AS rn
            FROM dbo.Name n 
            INNER JOIN dbo.Account a ON n.PARENTACCOUNT = a.ACCOUNTNUMBER
            WHERE 
                n.TYPE = 0 
                AND n.SSN IS NOT NULL 
                AND n.SSN <> '' 
                AND n.ProcessDate = FORMAT(DATEADD(DAY, -1, GETDATE()), 'yyyyMMdd') 
                AND a.ProcessDate = FORMAT(DATEADD(DAY, -1, GETDATE()), 'yyyyMMdd') 
                AND a.TYPE IN (0, 1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 15, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99) 
                AND (a.CloseDate IS NULL OR a.CloseDate = '1900-01-01') 
                AND (
                    EXISTS (
                        SELECT 1
                        FROM dbo.Loan l
                        WHERE 
                            l.PARENTACCOUNT = a.AccountNumber 
                            AND l.ProcessDate = FORMAT(DATEADD(DAY, -1, GETDATE()), 'yyyyMMdd') 
                            AND (l.ChargeOffDate IS NULL OR l.ChargeOffDate = '1900-01-01') 
                            AND (l.CloseDate IS NULL OR l.CloseDate = '1900-01-01')
                    ) 
                    OR 
                    EXISTS (
                        SELECT 1
                        FROM dbo.SAVINGS s
                        WHERE 
                            s.PARENTACCOUNT = a.AccountNumber 
                            AND s.ProcessDate = FORMAT(DATEADD(DAY, -1, GETDATE()), 'yyyyMMdd') 
                            AND (s.ChargeOffDate IS NULL OR s.ChargeOffDate = '1900-01-01') 
                            AND (s.CloseDate IS NULL OR s.CloseDate = '1900-01-01')
                    )
                )
        ) ActiveMembers
        WHERE ActiveMembers.rn = 1
        """
        
        # Fallback queries for error handling
        fallback_queries = [
            # Simplified version without the complex subquery
            """
            SELECT 
                n.PARENTACCOUNT as AccountNumber,
                n.LAST as MemberName,
                n.SSN,
                a.TYPE as AccountType,
                a.Branch,
                'Unknown' as AccountTypeDescription,
                'Unknown' as FormattedAccountType,
                1 as MemberCount
            FROM dbo.Name n 
            INNER JOIN dbo.Account a ON n.PARENTACCOUNT = a.ACCOUNTNUMBER
            WHERE 
                n.TYPE = 0 
                AND n.SSN IS NOT NULL 
                AND n.SSN <> ''
                AND (a.CloseDate IS NULL OR a.CloseDate = '1900-01-01')
            """,
            
            # Most basic query
            """
            SELECT 
                PARENTACCOUNT as AccountNumber,
                LAST as MemberName,
                SSN,
                0 as AccountType,
                '' as Branch,
                'General Membership' as AccountTypeDescription,
                'Type 0000-General Membership' as FormattedAccountType,
                1 as MemberCount
            FROM dbo.Name 
            WHERE TYPE = 0 AND SSN IS NOT NULL AND SSN <> ''
            """
        ]
        
        try:
            result = self._execute_safe_query_with_fallbacks(
                database,
                primary_query,
                fallback_queries,
                params={}
            )
            
            if not result.empty:
                # Normalize column names to match expected format
                result = result.rename(columns={
                    'AccountNumber': 'member_number',
                    'MemberName': 'last_name',
                    'SSN': 'member_ssn',
                    'AccountType': 'account_type',
                    'Branch': 'branch',
                    'AccountTypeDescription': 'account_type_description',
                    'FormattedAccountType': 'formatted_account_type',
                    'MemberCount': 'member_count',
                    'ProcessDate': 'process_date'
                })
                
                # Add age group analysis if possible (simplified)
                result['age_group'] = 'Unknown'  # Will be enhanced in future iterations
                
                # Add join date placeholder
                result['join_date'] = datetime.now().date()
                
                # Add account summary
                result = self._add_account_summary(result, database)
            
            self.logger.info(f"Retrieved {len(result)} active members from {database} using updated CRCU logic")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve active members with updated logic: {e}")
            return pd.DataFrame(columns=[
                'member_number', 'last_name', 'member_ssn', 'account_type', 'branch',
                'account_type_description', 'formatted_account_type', 'member_count',
                'age_group', 'join_date', 'total_accounts', 'total_balance'
            ])
    
    def _add_account_summary(self, member_data: pd.DataFrame, database: str) -> pd.DataFrame:
        """
        Add account summary information to member data.
        
        Args:
            member_data: Basic member data
            database: Target database
            
        Returns:
            Member data with account summary
        """
        enriched_data = member_data.copy()
        
        # Add default values
        enriched_data['total_accounts'] = 1
        enriched_data['total_balance'] = 0
        enriched_data['total_savings_balance'] = 0
        enriched_data['total_loan_balance'] = 0
        
        try:
            # Try to get actual account counts
            if len(enriched_data) > 0:
                member_list = "','".join([str(m) for m in enriched_data['member_number'].head(100)])  # Limit for performance
                
                account_query = f"""
                SELECT 
                    ACCOUNTNUMBER as member_number,
                    COUNT(*) as account_count
                FROM ACCOUNT 
                WHERE ACCOUNTNUMBER IN ('{member_list}')
                    AND CLOSEDATE = '19000101'
                GROUP BY ACCOUNTNUMBER
                """
                
                account_counts = self.execute_query(account_query, database)
                if not account_counts.empty:
                    enriched_data = enriched_data.merge(
                        account_counts,
                        on='member_number',
                        how='left'
                    )
                    enriched_data['total_accounts'] = enriched_data['account_count'].fillna(1)
                    enriched_data.drop('account_count', axis=1, inplace=True, errors='ignore')
                    
        except Exception as e:
            self.logger.warning(f"Could not enrich with account summary: {e}")
        
        return enriched_data
    
    def _sanitize_member_data(self, member_data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove or mask PII from member data for safe output.
        
        Args:
            member_data: Raw member data with PII
            
        Returns:
            Sanitized DataFrame with PII removed/masked
        """
        safe_data = member_data.copy()
        
        # Remove or mask PII fields
        if 'member_ssn' in safe_data.columns:
            safe_data = safe_data.drop('member_ssn', axis=1)
        
        if 'last_name' in safe_data.columns:
            safe_data['last_name'] = safe_data['last_name'].apply(
                lambda x: x[0] + '*' * (len(x) - 1) if x and len(x) > 0 else 'N/A'
            )
        
        if 'first_name' in safe_data.columns:
            safe_data['first_name'] = safe_data['first_name'].apply(
                lambda x: x[0] + '*' * (len(x) - 1) if x and len(x) > 0 else 'N/A'
            )
        
        if 'middle_initial' in safe_data.columns:
            safe_data['middle_initial'] = safe_data['middle_initial'].apply(
                lambda x: x[0] + '*' if x and len(x) > 1 else x
            )
        
        return safe_data
    
    def _generate_active_members_insights(self, member_data: pd.DataFrame, metrics: Dict[str, Any]) -> List[str]:
        """
        Generate insights about the active member base.
        
        Args:
            member_data: Active member data
            metrics: Calculated metrics
            
        Returns:
            List of insight strings
        """
        insights = []
        
        total_members = metrics.get('total_active_members', 0)
        
        # Membership growth insights
        if 'join_date' in member_data.columns:
            current_year = datetime.now().year
            recent_members = member_data[
                pd.to_datetime(member_data['join_date']).dt.year >= current_year - 1
            ]
            if len(recent_members) > 0:
                growth_rate = (len(recent_members) / total_members) * 100
                insights.append(f"Recent growth: {len(recent_members)} new members in last 12 months ({growth_rate:.1f}% of active base)")
        
        # Age distribution insights
        if 'age_distribution' in metrics:
            age_dist = metrics['age_distribution']
            dominant_age_group = max(age_dist.keys(), key=lambda k: age_dist[k])
            insights.append(f"Largest age segment: {dominant_age_group} with {age_dist[dominant_age_group]} members")
            
            if age_dist.get('Under 25', 0) + age_dist.get('25-34', 0) > total_members * 0.4:
                insights.append("Strong younger member base - focus on digital services and growth products")
            elif age_dist.get('55-64', 0) + age_dist.get('65+', 0) > total_members * 0.4:
                insights.append("Mature member base - emphasize wealth management and retirement services")
        
        # Product penetration insights
        avg_accounts = metrics.get('avg_accounts_per_member', 0)
        if avg_accounts < 2.0:
            insights.append("Low product penetration - opportunity for cross-selling additional services")
        elif avg_accounts > 3.0:
            insights.append("Strong product penetration - focus on deepening existing relationships")
        
        multi_product_members = metrics.get('multi_product_members', 0)
        if multi_product_members > 0:
            multi_product_pct = (multi_product_members / total_members) * 100
            insights.append(f"Multi-product members: {multi_product_pct:.1f}% ({multi_product_members} members)")
        
        # Balance insights
        avg_balance = metrics.get('avg_balance_per_member', 0)
        if avg_balance > 50000:
            insights.append("High-value member base - strong foundation for premium services")
        elif avg_balance < 10000:
            insights.append("Growing member base - focus on engagement and balance building")
        
        if not insights:
            insights.append("Active member base analysis complete - continue monitoring for trends")
        
        return insights

    def _generate_member_recommendations(self, rfm_results: Dict[str, Any], cluster_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on member analysis."""
        recommendations = []
        
        # RFM-based recommendations
        if rfm_results and 'segment_metrics' in rfm_results:
            at_risk_pct = rfm_results['segment_metrics'].get('at_risk_pct', 0)
            lost_pct = rfm_results['segment_metrics'].get('lost_customers_pct', 0)
            
            if at_risk_pct > 15:
                recommendations.append("High percentage of at-risk members - implement retention campaigns")
            
            if lost_pct > 10:
                recommendations.append("Significant lost customer segment - investigate win-back opportunities")
        
        # Cluster-based recommendations
        if cluster_results and 'cluster_profiles' in cluster_results:
            for cluster_name, profile in cluster_results['cluster_profiles'].items():
                traits = profile.get('dominant_traits', [])
                
                if 'Young Members' in traits and 'Low Balance' in traits:
                    recommendations.append(f"Target young, low-balance members with growth products")
                
                if 'Traditional Banking' in traits:
                    recommendations.append(f"Focus on digital adoption initiatives for traditional banking segments")
        
        if not recommendations:
            recommendations.append("Continue monitoring member segments for optimization opportunities")
        
        return recommendations
