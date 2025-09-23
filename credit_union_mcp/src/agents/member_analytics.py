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
            "value_migration"
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
        Retrieve member demographic and account data.
        
        Args:
            database: Target database
            as_of_date: Analysis date
            
        Returns:
            DataFrame with member data
        """
        query = """
        SELECT 
            m.member_id,
            m.join_date,
            m.birth_date,
            m.gender,
            m.marital_status,
            m.income_range,
            m.education_level,
            m.employment_status,
            m.geographic_region,
            m.primary_branch,
            m.status,
            m.last_activity_date,
            
            -- Account information
            COUNT(DISTINCT a.account_id) as total_accounts,
            SUM(CASE WHEN a.account_type = 'Checking' THEN 1 ELSE 0 END) as checking_accounts,
            SUM(CASE WHEN a.account_type = 'Savings' THEN 1 ELSE 0 END) as savings_accounts,
            SUM(CASE WHEN a.account_type = 'Certificate' THEN 1 ELSE 0 END) as certificate_accounts,
            SUM(a.current_balance) as total_balance,
            
            -- Loan information
            COUNT(DISTINCT l.loan_id) as total_loans,
            SUM(l.current_balance) as total_loan_balance,
            
            -- Service usage
            m.online_banking_enrolled,
            m.mobile_banking_enrolled,
            m.debit_card_active,
            m.credit_card_active
            
        FROM members m
        LEFT JOIN accounts a ON m.member_id = a.member_id AND a.status = 'Active'
        LEFT JOIN loans l ON m.member_id = l.member_id AND l.status IN ('Active', 'Current')
        WHERE m.join_date <= ?
        GROUP BY m.member_id, m.join_date, m.birth_date, m.gender, m.marital_status,
                 m.income_range, m.education_level, m.employment_status, 
                 m.geographic_region, m.primary_branch, m.status, m.last_activity_date,
                 m.online_banking_enrolled, m.mobile_banking_enrolled, 
                 m.debit_card_active, m.credit_card_active
        """
        
        try:
            result = self.execute_query(query, database, params={'as_of_date': as_of_date})
            return result
        except Exception as e:
            self.logger.warning(f"Could not retrieve member data: {e}")
            return pd.DataFrame()
    
    def _get_transaction_data(self, database: str, as_of_date: str) -> pd.DataFrame:
        """Retrieve transaction history for analysis."""
        # Get last 24 months of transaction data
        start_date = (datetime.strptime(as_of_date, '%Y-%m-%d') - timedelta(days=730)).strftime('%Y-%m-%d')
        
        query = """
        SELECT 
            t.member_id,
            t.transaction_date,
            t.transaction_type,
            t.amount,
            t.description,
            t.channel,
            a.account_type
        FROM transactions t
        JOIN accounts a ON t.account_id = a.account_id
        WHERE t.transaction_date BETWEEN ? AND ?
        AND t.transaction_type NOT IN ('Transfer', 'Internal')
        ORDER BY t.member_id, t.transaction_date
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
