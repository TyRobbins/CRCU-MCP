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
        query = """
        SELECT TABLE_NAME 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_SCHEMA = ? AND TABLE_TYPE = 'BASE TABLE'
        ORDER BY TABLE_NAME
        """
        
        try:
            result = self.execute_query(database, query, params={'schema': schema})
            return result['TABLE_NAME'].tolist()
            
        except Exception as e:
            logger.error(f"Failed to get tables for {database}.{schema}: {e}")
            raise
    
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
        query = """
        SELECT 
            COLUMN_NAME,
            DATA_TYPE,
            IS_NULLABLE,
            COLUMN_DEFAULT,
            CHARACTER_MAXIMUM_LENGTH,
            NUMERIC_PRECISION,
            NUMERIC_SCALE,
            ORDINAL_POSITION
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
        ORDER BY ORDINAL_POSITION
        """
        
        try:
            result = self.execute_query(
                database, 
                query, 
                params={'schema': schema, 'table_name': table_name}
            )
            
            columns = []
            for _, row in result.iterrows():
                column_info = {
                    'name': row['COLUMN_NAME'],
                    'type': row['DATA_TYPE'],
                    'nullable': row['IS_NULLABLE'] == 'YES',
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
            raise
    
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
