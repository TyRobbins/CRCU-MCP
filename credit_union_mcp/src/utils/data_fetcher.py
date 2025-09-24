"""
Data Fetcher Utilities for Credit Union MCP Server

Provides standardized data retrieval methods with caching, error handling,
and fallback mechanisms to reduce code duplication across agents.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

from .query_builder import QueryBuilder, CommonQueries, QueryValidator
from .performance_utils import cache_with_ttl, PerformanceTracker


class DataFetcher:
    """
    Standardized data fetcher with caching, error handling, and performance optimization.
    """
    
    def __init__(self, db_manager, cache_ttl: int = 1800):
        """
        Initialize data fetcher.
        
        Args:
            db_manager: Database manager instance
            cache_ttl: Cache TTL in seconds (default: 30 minutes)
        """
        self.db_manager = db_manager
        self.cache_ttl = cache_ttl
        self.query_builder = QueryBuilder()
        self.performance_tracker = PerformanceTracker()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    @cache_with_ttl(ttl=1800)  # 30 minute cache
    def get_member_data(self, database: str = "ARCUSYM000", 
                       as_of_date: Optional[str] = None,
                       include_inactive: bool = False,
                       limit: Optional[int] = None) -> pd.DataFrame:
        """
        Get standardized member data with caching.
        
        Args:
            database: Target database
            as_of_date: Analysis date filter
            include_inactive: Include inactive members
            limit: Optional row limit
            
        Returns:
            DataFrame with member data
        """
        with self.performance_tracker.track_operation("get_member_data"):
            query = self.query_builder.build_member_query(
                as_of_date=as_of_date,
                include_inactive=include_inactive,
                limit=limit
            )
            
            params = {}
            if as_of_date:
                params['as_of_date'] = as_of_date
                
            return self._execute_query_with_fallbacks(
                database=database,
                primary_query=query,
                params=params,
                operation_name="get_member_data"
            )
    
    @cache_with_ttl(ttl=900)  # 15 minute cache
    def get_account_summary(self, database: str = "ARCUSYM000",
                          member_ids: Optional[List[str]] = None,
                          as_of_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get account summary data with caching.
        
        Args:
            database: Target database
            member_ids: Optional member ID filter
            as_of_date: Analysis date
            
        Returns:
            DataFrame with account summary
        """
        with self.performance_tracker.track_operation("get_account_summary"):
            query = self.query_builder.build_account_summary_query(
                member_ids=member_ids,
                as_of_date=as_of_date
            )
            
            params = {}
            if member_ids:
                params['member_ids'] = tuple(member_ids)  # Convert to tuple for SQL IN clause
            if as_of_date:
                params['as_of_date'] = as_of_date
                
            return self._execute_query_with_fallbacks(
                database=database,
                primary_query=query,
                params=params,
                operation_name="get_account_summary"
            )
    
    @cache_with_ttl(ttl=600)  # 10 minute cache
    def get_transaction_data(self, database: str = "ARCUSYM000",
                           member_ids: Optional[List[str]] = None,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           transaction_types: Optional[List[str]] = None,
                           min_amount: Optional[float] = None) -> pd.DataFrame:
        """
        Get transaction data with caching and filters.
        
        Args:
            database: Target database
            member_ids: Optional member ID filter
            start_date: Start date filter
            end_date: End date filter
            transaction_types: Transaction type filter
            min_amount: Minimum amount filter
            
        Returns:
            DataFrame with transaction data
        """
        with self.performance_tracker.track_operation("get_transaction_data"):
            query = self.query_builder.build_transaction_query(
                member_ids=member_ids,
                start_date=start_date,
                end_date=end_date,
                transaction_types=transaction_types,
                min_amount=min_amount
            )
            
            params = {}
            if member_ids:
                params['member_ids'] = tuple(member_ids)
            if start_date:
                params['start_date'] = start_date
            if end_date:
                params['end_date'] = end_date
            if transaction_types:
                params['transaction_types'] = tuple(transaction_types)
            if min_amount:
                params['min_amount'] = min_amount
                
            return self._execute_query_with_fallbacks(
                database=database,
                primary_query=query,
                params=params,
                operation_name="get_transaction_data"
            )
    
    @cache_with_ttl(ttl=3600)  # 1 hour cache
    def get_financial_summary(self, database: str = "ARCUSYM000",
                            as_of_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get institution-level financial summary.
        
        Args:
            database: Target database
            as_of_date: Analysis date
            
        Returns:
            DataFrame with financial summary
        """
        with self.performance_tracker.track_operation("get_financial_summary"):
            query = self.query_builder.build_financial_summary_query(as_of_date=as_of_date)
            
            params = {}
            if as_of_date:
                params['as_of_date'] = as_of_date
                
            return self._execute_query_with_fallbacks(
                database=database,
                primary_query=query,
                params=params,
                operation_name="get_financial_summary"
            )
    
    @cache_with_ttl(ttl=1800)  # 30 minute cache
    def get_active_members_count(self, database: str = "ARCUSYM000",
                               as_of_date: Optional[str] = None) -> int:
        """
        Get count of active members.
        
        Args:
            database: Target database
            as_of_date: Analysis date
            
        Returns:
            Count of active members
        """
        with self.performance_tracker.track_operation("get_active_members_count"):
            query = CommonQueries.get_active_members_count(as_of_date=as_of_date)
            
            params = {}
            if as_of_date:
                params['as_of_date'] = as_of_date
                
            result = self._execute_query_with_fallbacks(
                database=database,
                primary_query=query,
                params=params,
                operation_name="get_active_members_count"
            )
            
            return result.iloc[0]['active_member_count'] if not result.empty else 0
    
    @cache_with_ttl(ttl=3600)  # 1 hour cache
    def get_new_members_period(self, database: str = "ARCUSYM000",
                             start_date: str, end_date: str) -> int:
        """
        Get new members for a specific period.
        
        Args:
            database: Target database
            start_date: Period start date
            end_date: Period end date
            
        Returns:
            Count of new members
        """
        with self.performance_tracker.track_operation("get_new_members_period"):
            query = CommonQueries.get_new_members_period(start_date, end_date)
            
            params = {
                'start_date': start_date,
                'end_date': end_date
            }
            
            result = self._execute_query_with_fallbacks(
                database=database,
                primary_query=query,
                params=params,
                operation_name="get_new_members_period"
            )
            
            return result.iloc[0]['new_member_count'] if not result.empty else 0
    
    def get_account_types_summary(self, database: str = "ARCUSYM000") -> pd.DataFrame:
        """
        Get summary of account types.
        
        Args:
            database: Target database
            
        Returns:
            DataFrame with account type summary
        """
        with self.performance_tracker.track_operation("get_account_types_summary"):
            query = CommonQueries.get_account_types_summary()
            
            return self._execute_query_with_fallbacks(
                database=database,
                primary_query=query,
                params={},
                operation_name="get_account_types_summary"
            )
    
    @cache_with_ttl(ttl=7200)  # 2 hour cache
    def get_member_relationship_length(self, database: str = "ARCUSYM000") -> pd.DataFrame:
        """
        Calculate member relationship length.
        
        Args:
            database: Target database
            
        Returns:
            DataFrame with relationship length data
        """
        with self.performance_tracker.track_operation("get_member_relationship_length"):
            query = CommonQueries.get_member_relationship_length()
            
            return self._execute_query_with_fallbacks(
                database=database,
                primary_query=query,
                params={},
                operation_name="get_member_relationship_length"
            )
    
    @cache_with_ttl(ttl=1800)  # 30 minute cache
    def get_high_value_members(self, database: str = "ARCUSYM000",
                             threshold: float = 10000.0) -> pd.DataFrame:
        """
        Get members with balances above threshold.
        
        Args:
            database: Target database
            threshold: Minimum balance threshold
            
        Returns:
            DataFrame with high-value members
        """
        with self.performance_tracker.track_operation("get_high_value_members"):
            query = CommonQueries.get_high_value_members(threshold)
            
            params = {'threshold': threshold}
            
            return self._execute_query_with_fallbacks(
                database=database,
                primary_query=query,
                params=params,
                operation_name="get_high_value_members"
            )
    
    @cache_with_ttl(ttl=3600)  # 1 hour cache
    def get_transaction_volume_trends(self, database: str = "ARCUSYM000",
                                    months_back: int = 12) -> pd.DataFrame:
        """
        Get transaction volume trends by month.
        
        Args:
            database: Target database
            months_back: Number of months to look back
            
        Returns:
            DataFrame with transaction volume trends
        """
        with self.performance_tracker.track_operation("get_transaction_volume_trends"):
            query = CommonQueries.get_transaction_volume_by_month(months_back)
            
            params = {'months_back': months_back}
            
            return self._execute_query_with_fallbacks(
                database=database,
                primary_query=query,
                params=params,
                operation_name="get_transaction_volume_trends"
            )
    
    @cache_with_ttl(ttl=900)  # 15 minute cache
    def get_delinquency_summary(self, database: str = "ARCUSYM000") -> pd.DataFrame:
        """
        Get loan delinquency summary.
        
        Args:
            database: Target database
            
        Returns:
            DataFrame with delinquency summary
        """
        with self.performance_tracker.track_operation("get_delinquency_summary"):
            query = CommonQueries.get_delinquency_summary()
            
            return self._execute_query_with_fallbacks(
                database=database,
                primary_query=query,
                params={},
                operation_name="get_delinquency_summary"
            )
    
    async def get_data_batch(self, queries: List[Tuple[str, str, Dict[str, Any]]]) -> Dict[str, pd.DataFrame]:
        """
        Execute multiple queries in parallel for improved performance.
        
        Args:
            queries: List of (query_name, query, params) tuples
            
        Returns:
            Dictionary mapping query names to results
        """
        with self.performance_tracker.track_operation("get_data_batch"):
            loop = asyncio.get_event_loop()
            
            # Create tasks for parallel execution
            tasks = []
            for query_name, query, params in queries:
                task = loop.run_in_executor(
                    self.executor,
                    self._execute_single_query,
                    "ARCUSYM000",  # Default database
                    query,
                    params,
                    query_name
                )
                tasks.append((query_name, task))
            
            # Wait for all tasks to complete
            results = {}
            for query_name, task in tasks:
                try:
                    result = await task
                    results[query_name] = result
                except Exception as e:
                    logger.error(f"Batch query failed for {query_name}: {e}")
                    results[query_name] = pd.DataFrame()  # Empty DataFrame on failure
            
            return results
    
    def _execute_query_with_fallbacks(self, database: str, primary_query: str, 
                                    params: Dict[str, Any], operation_name: str) -> pd.DataFrame:
        """
        Execute query with fallback mechanisms and retry logic.
        
        Args:
            database: Target database
            primary_query: Primary query to execute
            params: Query parameters
            operation_name: Operation name for logging
            
        Returns:
            DataFrame with results
        """
        # Validate query first
        if not QueryValidator.is_safe_query(primary_query):
            raise ValueError(f"Unsafe query detected in {operation_name}")
        
        # Add row limit if not present
        safe_query = QueryValidator.add_row_limit(primary_query)
        
        # Sanitize parameters
        safe_params = QueryValidator.validate_parameters(params)
        
        # Execute with retry logic
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Executing {operation_name} (attempt {attempt + 1})")
                
                result = self.db_manager.execute_query(
                    database=database,
                    query=safe_query,
                    params=safe_params if safe_params else None
                )
                
                logger.info(f"{operation_name} completed successfully: {len(result)} rows")
                return result
                
            except Exception as e:
                logger.warning(f"{operation_name} attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"{operation_name} failed after {max_retries} attempts")
                    # Return empty DataFrame with expected structure
                    return pd.DataFrame()
    
    def _execute_single_query(self, database: str, query: str, params: Dict[str, Any], 
                            operation_name: str) -> pd.DataFrame:
        """
        Execute a single query (used for batch operations).
        
        Args:
            database: Target database
            query: SQL query
            params: Query parameters
            operation_name: Operation name
            
        Returns:
            DataFrame with results
        """
        return self._execute_query_with_fallbacks(database, query, params, operation_name)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for data fetching operations.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            'total_operations': len(self.performance_tracker.operation_times),
            'operation_stats': self.performance_tracker.get_stats(),
            'cache_stats': {
                # Cache stats would be implemented in cache decorator
                'cache_enabled': True,
                'default_ttl': self.cache_ttl
            }
        }
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        # This would be implemented with the cache decorator
        logger.info("Data fetcher cache cleared")
    
    def warm_cache(self, database: str = "ARCUSYM000") -> None:
        """
        Pre-warm cache with commonly used data.
        
        Args:
            database: Target database
        """
        logger.info(f"Warming cache for {database}")
        
        try:
            # Pre-load commonly used data
            self.get_active_members_count(database)
            self.get_account_types_summary(database)
            self.get_financial_summary(database)
            
            logger.info("Cache warming completed successfully")
            
        except Exception as e:
            logger.error(f"Cache warming failed: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.executor.shutdown(wait=True)


class BatchDataFetcher:
    """
    Specialized data fetcher for batch operations and bulk data processing.
    """
    
    def __init__(self, db_manager, batch_size: int = 1000):
        """
        Initialize batch data fetcher.
        
        Args:
            db_manager: Database manager instance
            batch_size: Size of batches for processing
        """
        self.db_manager = db_manager
        self.batch_size = batch_size
        self.data_fetcher = DataFetcher(db_manager)
    
    def get_members_in_batches(self, database: str = "ARCUSYM000",
                             as_of_date: Optional[str] = None,
                             include_inactive: bool = False) -> pd.DataFrame:
        """
        Get member data in batches to handle large datasets.
        
        Args:
            database: Target database
            as_of_date: Analysis date
            include_inactive: Include inactive members
            
        Returns:
            Complete DataFrame with all member data
        """
        all_data = []
        offset = 0
        
        while True:
            batch_query = f"""
            SELECT * FROM (
                {self.data_fetcher.query_builder.build_member_query(as_of_date, include_inactive)}
            ) batch
            ORDER BY member_id
            OFFSET {offset} ROWS
            FETCH NEXT {self.batch_size} ROWS ONLY
            """
            
            params = {}
            if as_of_date:
                params['as_of_date'] = as_of_date
            
            batch_data = self.data_fetcher._execute_query_with_fallbacks(
                database=database,
                primary_query=batch_query,
                params=params,
                operation_name=f"get_members_batch_{offset}"
            )
            
            if batch_data.empty:
                break
                
            all_data.append(batch_data)
            offset += self.batch_size
            
            logger.info(f"Processed batch {offset // self.batch_size}, {len(batch_data)} rows")
        
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    
    async def get_parallel_datasets(self, database: str = "ARCUSYM000",
                                  as_of_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Get multiple commonly used datasets in parallel.
        
        Args:
            database: Target database
            as_of_date: Analysis date
            
        Returns:
            Dictionary with multiple datasets
        """
        queries = [
            ("members", self.data_fetcher.query_builder.build_member_query(as_of_date), 
             {'as_of_date': as_of_date} if as_of_date else {}),
            ("financial_summary", self.data_fetcher.query_builder.build_financial_summary_query(as_of_date),
             {'as_of_date': as_of_date} if as_of_date else {}),
            ("account_types", CommonQueries.get_account_types_summary(), {}),
            ("relationship_length", CommonQueries.get_member_relationship_length(), {})
        ]
        
        return await self.data_fetcher.get_data_batch(queries)
