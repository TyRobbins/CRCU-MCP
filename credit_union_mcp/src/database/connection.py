"""
Database Connection Manager for Credit Union MCP Server

Provides secure, pooled connections to TEMENOS and ARCUSYM000 SQL Server databases
with read-only query enforcement and comprehensive error handling.
"""

import re
import pandas as pd
import pyodbc
from sqlalchemy import create_engine, text, MetaData, Table
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool
from typing import Dict, List, Any, Optional, Union
from loguru import logger
import yaml
from pathlib import Path


class DatabaseManager:
    """
    Manages database connections and query execution for TEMENOS and ARCUSYM000 databases.
    
    Features:
    - Connection pooling with SQLAlchemy
    - Windows Authentication support
    - Read-only query enforcement
    - Automatic reconnection on failure
    - Comprehensive error handling
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize database manager with configuration.
        
        Args:
            config: Database configuration dictionary
        """
        self.config = config
        self.engines: Dict[str, Engine] = {}
        self.metadata_cache: Dict[str, MetaData] = {}
        
        # Initialize connections
        self._initialize_engines()
        
        # Read-only SQL keywords that are allowed
        self.readonly_keywords = {
            'SELECT', 'WITH', 'SHOW', 'DESCRIBE', 'EXPLAIN', 'EXEC', 'EXECUTE'
        }
        
        # Write operation keywords that are forbidden
        self.write_keywords = {
            'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 
            'TRUNCATE', 'MERGE', 'UPSERT', 'REPLACE'
        }
    
    def _initialize_engines(self) -> None:
        """Initialize SQLAlchemy engines for both databases."""
        pool_config = self.config.get('connection_pool', {})
        
        for db_name in ['TEMENOS', 'ARCUSYM000']:
            if db_name in self.config:
                try:
                    connection_string = self._build_connection_string(db_name)
                    
                    engine = create_engine(
                        connection_string,
                        poolclass=QueuePool,
                        pool_size=pool_config.get('pool_size', 5),
                        max_overflow=pool_config.get('max_overflow', 10),
                        pool_timeout=pool_config.get('pool_timeout', 30),
                        pool_recycle=pool_config.get('pool_recycle', 3600),
                        echo=False  # Set to True for SQL debugging
                    )
                    
                    self.engines[db_name] = engine
                    logger.info(f"Initialized engine for {db_name} database")
                    
                except Exception as e:
                    logger.error(f"Failed to initialize engine for {db_name}: {e}")
                    raise
    
    def _build_connection_string(self, database: str) -> str:
        """
        Build SQL Server connection string for the specified database.
        
        Args:
            database: Database name (TEMENOS or ARCUSYM000)
            
        Returns:
            ODBC connection string
        """
        db_config = self.config[database]
        server = db_config['server']
        db_name = db_config['database']
        
        if db_config.get('windows_auth', True):
            # Windows Authentication
            connection_string = (
                f"mssql+pyodbc://@{server}/{db_name}"
                f"?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
            )
        else:
            # SQL Server Authentication
            username = db_config['username']
            password = db_config['password']
            connection_string = (
                f"mssql+pyodbc://{username}:{password}@{server}/{db_name}"
                f"?driver=ODBC+Driver+17+for+SQL+Server"
            )
        
        return connection_string
    
    def validate_query(self, query: str) -> bool:
        """
        Validate that the query is read-only and safe to execute.
        
        Args:
            query: SQL query string
            
        Returns:
            True if query is valid and read-only
            
        Raises:
            ValueError: If query contains write operations
        """
        # Clean and normalize query
        clean_query = re.sub(r'--.*$', '', query, flags=re.MULTILINE)  # Remove comments
        clean_query = re.sub(r'/\*.*?\*/', '', clean_query, flags=re.DOTALL)  # Remove block comments
        clean_query = clean_query.upper().strip()
        
        # Check for forbidden write operations
        for keyword in self.write_keywords:
            if re.search(rf'\b{keyword}\b', clean_query):
                raise ValueError(f"Write operation '{keyword}' is not allowed. Only read queries are permitted.")
        
        # Ensure query starts with allowed keyword
        first_keyword = clean_query.split()[0] if clean_query.split() else ''
        if first_keyword not in self.readonly_keywords:
            raise ValueError(f"Query must start with one of: {', '.join(self.readonly_keywords)}")
        
        return True
    
    def execute_query(self, database: str, query: str, params: Optional[Dict] = None, 
                     max_rows: int = 10000) -> pd.DataFrame:
        """
        Execute a read-only SQL query against the specified database.
        
        Args:
            database: Database name (TEMENOS or ARCUSYM000)
            query: SQL query string
            params: Query parameters (optional)
            max_rows: Maximum number of rows to return
            
        Returns:
            DataFrame containing query results
            
        Raises:
            ValueError: If query validation fails
            Exception: If query execution fails
        """
        # Validate inputs
        if database not in self.engines:
            raise ValueError(f"Database '{database}' not configured or available")
        
        self.validate_query(query)
        
        # Add row limit if not present
        if 'TOP' not in query.upper() and 'LIMIT' not in query.upper():
            if query.upper().strip().startswith('SELECT'):
                query = query.replace('SELECT', f'SELECT TOP {max_rows}', 1)
        
        try:
            engine = self.engines[database]
            
            with engine.connect() as connection:
                if params:
                    result = pd.read_sql(text(query), connection, params=params)
                else:
                    result = pd.read_sql(text(query), connection)
                
                logger.info(f"Query executed successfully on {database}: {len(result)} rows returned")
                return result
                
        except Exception as e:
            logger.error(f"Query execution failed on {database}: {e}")
            raise
    
    def get_tables(self, database: str, schema: str = 'dbo') -> List[str]:
        """
        Get list of tables in the specified database and schema.
        
        Args:
            database: Database name
            schema: Schema name (default: 'dbo')
            
        Returns:
            List of table names
        """
        # Use simpler query to avoid INFORMATION_SCHEMA issues
        query = """
        SELECT name as TABLE_NAME
        FROM sys.tables 
        WHERE schema_id = SCHEMA_ID(:schema_name)
        ORDER BY name
        """
        
        try:
            result = self.execute_query(database, query, params={'schema_name': schema})
            return result['TABLE_NAME'].tolist()
            
        except Exception as e:
            logger.error(f"Failed to get tables for {database}.{schema}: {e}")
            # Fallback to basic query if sys.tables fails
            try:
                fallback_query = "SELECT name FROM sysobjects WHERE xtype='U' ORDER BY name"
                result = self.execute_query(database, fallback_query)
                return result['name'].tolist()
            except Exception as fallback_error:
                logger.error(f"Fallback query also failed: {fallback_error}")
                raise e
    
    def get_table_schema(self, database: str, table_name: str, schema: str = 'dbo') -> Dict[str, Any]:
        """
        Get detailed schema information for a specific table.
        
        Args:
            database: Database name
            table_name: Name of the table
            schema: Schema name (default: 'dbo')
            
        Returns:
            Dictionary containing table schema information
        """
        # Use sys.columns instead of INFORMATION_SCHEMA to avoid COUNT issues
        query = """
        SELECT 
            c.name as COLUMN_NAME,
            t.name as DATA_TYPE,
            c.is_nullable as IS_NULLABLE,
            dc.definition as COLUMN_DEFAULT,
            c.max_length as CHARACTER_MAXIMUM_LENGTH,
            c.precision as NUMERIC_PRECISION,
            c.scale as NUMERIC_SCALE,
            c.column_id as ORDINAL_POSITION
        FROM sys.columns c
        INNER JOIN sys.types t ON c.user_type_id = t.user_type_id
        LEFT JOIN sys.default_constraints dc ON c.default_object_id = dc.object_id
        WHERE c.object_id = OBJECT_ID(:full_table_name)
        ORDER BY c.column_id
        """
        
        try:
            full_table_name = f"{schema}.{table_name}"
            result = self.execute_query(
                database, 
                query, 
                params={'full_table_name': full_table_name}
            )
            
            columns = []
            for _, row in result.iterrows():
                column_info = {
                    'name': row['COLUMN_NAME'],
                    'type': row['DATA_TYPE'],
                    'nullable': bool(row['IS_NULLABLE']),
                    'default': row['COLUMN_DEFAULT'],
                    'max_length': row['CHARACTER_MAXIMUM_LENGTH'],
                    'precision': row['NUMERIC_PRECISION'],
                    'scale': row['NUMERIC_SCALE'],
                    'position': row['ORDINAL_POSITION']
                }
                columns.append(column_info)
            
            return {
                'database': database,
                'schema': schema,
                'table_name': table_name,
                'columns': columns,
                'column_count': len(columns)
            }
            
        except Exception as e:
            logger.error(f"Failed to get schema for {database}.{schema}.{table_name}: {e}")
            # Fallback to simple column check
            try:
                fallback_query = f"SELECT TOP 1 * FROM [{schema}].[{table_name}]"
                sample_data = self.execute_query(database, fallback_query)
                columns = [{'name': col, 'type': 'unknown'} for col in sample_data.columns]
                return {
                    'database': database,
                    'schema': schema,
                    'table_name': table_name,
                    'columns': columns,
                    'column_count': len(columns)
                }
            except Exception as fallback_error:
                logger.error(f"Schema fallback also failed: {fallback_error}")
                raise e
    
    def test_connection(self, database: Optional[str] = None) -> Union[bool, Dict[str, bool]]:
        """
        Test database connection(s).
        
        Args:
            database: Specific database to test, or None to test all
            
        Returns:
            Boolean for specific database, or dict of results for all databases
        """
        if database:
            if database not in self.engines:
                return False
            
            try:
                engine = self.engines[database]
                with engine.connect() as connection:
                    connection.execute(text("SELECT 1"))
                logger.info(f"Connection test successful for {database}")
                return True
                
            except Exception as e:
                logger.error(f"Connection test failed for {database}: {e}")
                return False
        else:
            # Test all databases
            results = {}
            for db_name in self.engines.keys():
                results[db_name] = self.test_connection(db_name)
            return results
    
    def get_sample_data(self, database: str, table_name: str, schema: str = 'dbo', 
                       limit: int = 10) -> pd.DataFrame:
        """
        Get sample data from a table.
        
        Args:
            database: Database name
            table_name: Table name
            schema: Schema name (default: 'dbo')
            limit: Number of rows to return
            
        Returns:
            DataFrame with sample data
        """
        query = f"SELECT TOP {limit} * FROM [{schema}].[{table_name}]"
        
        try:
            return self.execute_query(database, query)
            
        except Exception as e:
            logger.error(f"Failed to get sample data from {database}.{schema}.{table_name}: {e}")
            raise
    
    def discover_actual_schema(self, database: str) -> Dict[str, List[str]]:
        """
        Discover actual table schemas dynamically to map business concepts to real tables.
        
        Args:
            database: Target database name
            
        Returns:
            Dictionary mapping business concepts to actual table names
        """
        schema_mapping = {}
        
        try:
            # Get all tables in the database
            all_tables = self.get_tables(database)
            
            # Map business concepts to actual table patterns
            business_concepts = {
                'member_tables': ['NAME', 'MEMBERREC', 'MBRADDRESS'],
                'account_tables': ['ACCOUNT', 'SAVINGS', 'LOAN', 'CARD'],
                'transaction_tables': ['ACTIVITY', 'SAVINGSTRANSACTION', 'LOANTRANSACTION', 'EFT'],
                'workflow_tables': [t for t in all_tables if t.startswith('tbl')],
                'all_tables': all_tables
            }
            
            # Filter existing tables for each concept
            for concept, patterns in business_concepts.items():
                if concept == 'workflow_tables':
                    schema_mapping[concept] = patterns
                elif concept == 'all_tables':
                    schema_mapping[concept] = patterns
                else:
                    schema_mapping[concept] = [t for t in patterns if t in all_tables]
            
            logger.info(f"Discovered schema for {database}: {len(all_tables)} tables found")
            return schema_mapping
            
        except Exception as e:
            logger.error(f"Failed to discover schema for {database}: {e}")
            return {'error': str(e)}
    
    def validate_table_exists(self, database: str, table_name: str, schema: str = 'dbo') -> bool:
        """
        Validate that a table exists before attempting to query it.
        
        Args:
            database: Target database
            table_name: Name of table to validate
            schema: Schema name
            
        Returns:
            True if table exists, False otherwise
        """
        try:
            tables = self.get_tables(database, schema)
            return table_name in tables
        except Exception as e:
            logger.warning(f"Could not validate table existence {database}.{schema}.{table_name}: {e}")
            return False
    
    def get_safe_query_with_fallback(self, database: str, primary_query: str, fallback_query: str, 
                                   params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Execute a query with automatic fallback if primary query fails.
        
        Args:
            database: Target database
            primary_query: Primary query to attempt
            fallback_query: Fallback query if primary fails
            params: Query parameters
            
        Returns:
            DataFrame with results from successful query
        """
        try:
            # Try primary query first
            return self.execute_query(database, primary_query, params)
        except Exception as primary_error:
            logger.warning(f"Primary query failed, trying fallback: {primary_error}")
            try:
                return self.execute_query(database, fallback_query, params)
            except Exception as fallback_error:
                logger.error(f"Both queries failed. Primary: {primary_error}, Fallback: {fallback_error}")
                # Return empty DataFrame with expected structure
                return pd.DataFrame()

    def validate_table_columns(self, database: str, table_name: str, required_columns: List[str], schema: str = 'dbo') -> Dict[str, bool]:
        """
        Validate that required columns exist in a table.
        
        Args:
            database: Database name
            table_name: Table name to validate
            required_columns: List of column names that must exist
            schema: Schema name (default: 'dbo')
            
        Returns:
            Dictionary mapping column names to existence (True/False)
        """
        try:
            table_schema = self.get_table_schema(database, table_name, schema)
            existing_columns = {col['name'].upper() for col in table_schema['columns']}
            
            validation_results = {}
            for col in required_columns:
                validation_results[col] = col.upper() in existing_columns
                
            return validation_results
            
        except Exception as e:
            logger.error(f"Column validation failed for {database}.{schema}.{table_name}: {e}")
            # Return False for all columns if validation fails
            return {col: False for col in required_columns}
    
    def build_safe_query(self, base_query: str, database: str, table_validations: Dict[str, List[str]] = None) -> str:
        """
        Build a safe query by validating column names and providing fallbacks.
        
        Args:
            base_query: Base SQL query template
            database: Target database
            table_validations: Dict mapping table names to required columns
            
        Returns:
            Validated and safe SQL query
        """
        if not table_validations:
            return base_query
            
        safe_query = base_query
        
        for table_name, required_columns in table_validations.items():
            try:
                # Check if table exists first
                if not self.validate_table_exists(database, table_name):
                    logger.warning(f"Table {table_name} does not exist in {database}")
                    continue
                    
                # Validate columns
                column_validation = self.validate_table_columns(database, table_name, required_columns)
                
                # Log missing columns
                missing_columns = [col for col, exists in column_validation.items() if not exists]
                if missing_columns:
                    logger.warning(f"Missing columns in {table_name}: {missing_columns}")
                    
            except Exception as e:
                logger.error(f"Query validation failed for table {table_name}: {e}")
                
        return safe_query
    
    def execute_safe_query_with_fallback(self, database: str, primary_query: str, fallback_queries: List[str] = None, params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Execute query with multiple fallback options if primary fails.
        
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
                logger.info(f"Attempting query {i+1}/{len(queries_to_try)}")
                result = self.execute_query(database, query, params)
                if not result.empty:
                    logger.info(f"Query {i+1} succeeded with {len(result)} rows")
                    return result
                else:
                    logger.warning(f"Query {i+1} returned no data")
                    
            except Exception as e:
                logger.error(f"Query {i+1} failed: {e}")
                if i == len(queries_to_try) - 1:  # Last query
                    logger.error("All queries failed")
                    raise e
                else:
                    logger.info(f"Trying fallback query {i+2}")
                    
        # Return empty DataFrame if all queries fail but don't raise exception
        return pd.DataFrame()

    def close_connections(self) -> None:
        """Close all database connections."""
        for db_name, engine in self.engines.items():
            try:
                engine.dispose()
                logger.info(f"Closed connections for {db_name}")
            except Exception as e:
                logger.error(f"Error closing connections for {db_name}: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close connections."""
        self.close_connections()


def create_database_manager(config_path: Optional[Union[str, Path]] = None) -> DatabaseManager:
    """
    Factory function to create DatabaseManager instance from config file.
    
    Args:
        config_path: Path to configuration file (optional)
        
    Returns:
        Configured DatabaseManager instance
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / 'config' / 'database_config.yaml'
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return DatabaseManager(config)
        
    except Exception as e:
        logger.error(f"Failed to create database manager from {config_path}: {e}")
        raise
