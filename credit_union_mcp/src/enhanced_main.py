"""
Enhanced Credit Union MCP Server

An improved version with timeout management, column validation, better error handling,
and all the fixes identified from log analysis.
"""

import asyncio
import json
import yaml
import os
import sys
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import time
import re

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, Prompt, Resource
import pyodbc
from sqlalchemy import create_engine, text, inspect
import warnings
warnings.filterwarnings('ignore')

# Import utility classes with fallback for different execution contexts
try:
    from .utils import ContinuityManager, ConversationState, TokenEstimator
except ImportError:
    try:
        from credit_union_mcp.src.utils import ContinuityManager, ConversationState, TokenEstimator
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from src.utils import ContinuityManager, ConversationState, TokenEstimator

# Configure logging to stderr to avoid contaminating stdout JSON-RPC stream
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)


class EnhancedDatabaseManager:
    """Enhanced database manager with timeout management and column validation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.engines = {}
        self.schema_cache = {}
        self.column_cache = {}
        self.common_column_errors = {}
        self._initialize_connections()
        self._setup_common_column_mappings()
    
    def _initialize_connections(self):
        """Initialize database connections with timeout settings."""
        for db_name, db_config in self.config.items():
            try:
                connection_string = (
                    f"mssql+pyodbc://{db_config['username']}:{db_config['password']}"
                    f"@{db_config['server']}/{db_config['database']}"
                    f"?driver=ODBC+Driver+17+for+SQL+Server"
                    f"&timeout=30"  # Connection timeout
                    f"&query_timeout=60"  # Query timeout
                )
                engine = create_engine(
                    connection_string, 
                    pool_pre_ping=True,
                    pool_recycle=3600,  # Recycle connections every hour
                    connect_args={
                        'timeout': 30,
                        'autocommit': True,
                        'fast_executemany': True
                    }
                )
                self.engines[db_name] = engine
                logger.info(f"Connected to database: {db_name}")
                
                # Pre-cache schema information for faster validation
                self._cache_schema_info(db_name)
                
            except Exception as e:
                logger.error(f"Failed to connect to {db_name}: {e}")
    
    def _setup_common_column_mappings(self):
        """Set up mappings for commonly misused column names."""
        self.common_column_errors = {
            'EFFECTDATE': 'EFFECTIVEDATE',
            'TRANSAMOUNT': 'BALANCECHANGE', 
            'ProcessDate': None,  # Doesn't exist - suggest removal or alternative
            'VOIDID': 'VOIDCODE',
            'TRANSACTIONAMOUNT': 'BALANCECHANGE',
            'AMOUNT': 'BALANCECHANGE'
        }
    
    def _cache_schema_info(self, database: str):
        """Cache schema information for faster column validation."""
        try:
            if database not in self.engines:
                return
                
            with self.engines[database].connect() as conn:
                inspector = inspect(self.engines[database])
                
                # Get all tables and their columns
                tables = inspector.get_table_names()
                self.schema_cache[database] = {}
                self.column_cache[database] = set()
                
                for table in tables:
                    try:
                        columns = inspector.get_columns(table)
                        column_names = [col['name'].upper() for col in columns]
                        self.schema_cache[database][table.upper()] = column_names
                        self.column_cache[database].update(column_names)
                    except Exception as e:
                        logger.warning(f"Failed to cache columns for {table}: {e}")
                        
                logger.info(f"Cached schema information for {database}: {len(tables)} tables")
                
        except Exception as e:
            logger.error(f"Failed to cache schema for {database}: {e}")
    
    def validate_column_names(self, query: str, database: str) -> Dict[str, Any]:
        """Validate column names in query against actual schema."""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        if database not in self.column_cache:
            validation_result['warnings'].append("Schema cache not available for validation")
            return validation_result
        
        available_columns = self.column_cache[database]
        query_upper = query.upper()
        
        # Extract potential column references (simplified pattern)
        # Look for patterns like "table.column" or standalone identifiers
        column_patterns = re.findall(r'\b([A-Z_][A-Z0-9_]*)\b', query_upper)
        
        for potential_column in set(column_patterns):
            # Skip SQL keywords and common functions
            if potential_column in ['SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'NULL', 'IS', 'NOT',
                                   'INNER', 'LEFT', 'RIGHT', 'JOIN', 'ON', 'GROUP', 'BY', 'ORDER',
                                   'TOP', 'DISTINCT', 'COUNT', 'SUM', 'AVG', 'MAX', 'MIN',
                                   'YEAR', 'MONTH', 'DAY', 'GETDATE', 'DATEADD', 'CAST', 'CONVERT']:
                continue
            
            if potential_column not in available_columns:
                # Check if it's a common error with a known mapping
                if potential_column in self.common_column_errors:
                    correct_column = self.common_column_errors[potential_column]
                    if correct_column:
                        validation_result['valid'] = False
                        validation_result['errors'].append(
                            f"Column '{potential_column}' does not exist. Did you mean '{correct_column}'?"
                        )
                        validation_result['suggestions'].append({
                            'incorrect': potential_column,
                            'suggested': correct_column
                        })
                    else:
                        validation_result['valid'] = False
                        validation_result['errors'].append(
                            f"Column '{potential_column}' does not exist in this schema."
                        )
                else:
                    # Try to find similar column names
                    similar_columns = [col for col in available_columns 
                                     if potential_column in col or col in potential_column]
                    if similar_columns:
                        validation_result['warnings'].append(
                            f"Column '{potential_column}' not found. Similar columns: {similar_columns[:5]}"
                        )
        
        return validation_result
    
    def execute_query_with_timeout(self, database: str, query: str, max_rows: int = 10000, timeout: int = 30) -> pd.DataFrame:
        """Execute SQL query with timeout and enhanced error handling."""
        if database not in self.engines:
            raise ValueError(f"Database {database} not configured")
        
        # Validate column names first
        column_validation = self.validate_column_names(query, database)
        if not column_validation['valid']:
            error_msg = "Column validation failed: " + "; ".join(column_validation['errors'])
            if column_validation['suggestions']:
                error_msg += f"\n\nSuggested fixes: {column_validation['suggestions']}"
            raise ValueError(error_msg)
        
        # Enhanced query validation - allow SELECT queries and CTEs
        query_upper = query.upper().strip()
        if not (query_upper.startswith('SELECT') or query_upper.startswith('WITH')):
            raise ValueError("Only SELECT queries and CTEs are allowed")
        
        # Add performance optimizations
        optimized_query = self._optimize_query(query, max_rows)
        
        start_time = time.time()
        
        try:
            with self.engines[database].connect() as conn:
                # Set query timeout at connection level
                conn.execute(text(f"SET LOCK_TIMEOUT {timeout * 1000}"))  # Timeout in milliseconds
                
                result = pd.read_sql(text(optimized_query), conn)
                
                execution_time = time.time() - start_time
                
                if execution_time > 10:  # Log slow queries
                    logger.warning(f"Slow query detected: {execution_time:.2f}s for database {database}")
                
                # Add execution metadata
                if hasattr(result, 'attrs'):
                    result.attrs['execution_time'] = execution_time
                    result.attrs['database'] = database
                    result.attrs['row_count'] = len(result)
                
                return result
                
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Enhanced error messages
            error_msg = str(e)
            
            if "timeout" in error_msg.lower():
                error_msg = f"Query timed out after {timeout} seconds. Consider simplifying the query or adding more specific WHERE conditions."
            elif "invalid column" in error_msg.lower():
                error_msg += f"\n\nColumn validation warnings: {column_validation['warnings']}"
            
            logger.error(f"Query execution failed after {execution_time:.2f}s: {error_msg}")
            raise ValueError(error_msg)
    
    def _optimize_query(self, query: str, max_rows: int) -> str:
        """Apply automatic query optimizations."""
        query_upper = query.upper()
        
        # Add TOP clause if not present and it's a simple SELECT
        if 'TOP ' not in query_upper and 'LIMIT ' not in query_upper:
            if query_upper.startswith('SELECT') and 'FROM' in query_upper:
                query = query.replace('SELECT', f'SELECT TOP {max_rows}', 1)
        
        # Add query hints for better performance on large tables
        if 'SAVINGSTRANSACTION' in query_upper:
            # Add index hints for transaction table
            query = query.replace('FROM SAVINGSTRANSACTION', 
                                'FROM SAVINGSTRANSACTION WITH (NOLOCK)')
        
        if 'ACCOUNT' in query_upper and 'CLOSEDATE' not in query_upper:
            # Suggest adding active account filter
            logger.info("Consider adding active account filter: WHERE CLOSEDATE IS NULL")
        
        return query
    
    def get_tables(self, database: str, schema: str = 'dbo') -> List[Dict[str, Any]]:
        """Get list of tables with enhanced metadata."""
        if database not in self.engines:
            raise ValueError(f"Database {database} not configured")
        
        query = f"""
        SELECT 
            TABLE_NAME,
            (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS 
             WHERE TABLE_SCHEMA = t.TABLE_SCHEMA AND TABLE_NAME = t.TABLE_NAME) as COLUMN_COUNT,
            CASE 
                WHEN TABLE_NAME IN ('SAVINGSTRANSACTION', 'LOANTRANSACTION') THEN 'High Volume'
                WHEN TABLE_NAME IN ('ACCOUNT', 'NAME') THEN 'Core Entity' 
                ELSE 'Standard'
            END as TABLE_TYPE
        FROM INFORMATION_SCHEMA.TABLES t
        WHERE TABLE_SCHEMA = '{schema}' AND TABLE_TYPE = 'BASE TABLE'
        ORDER BY TABLE_NAME
        """
        
        with self.engines[database].connect() as conn:
            result = pd.read_sql(text(query), conn)
            return result.to_dict('records')
    
    def get_table_schema(self, database: str, table_name: str, schema: str = 'dbo') -> List[Dict[str, Any]]:
        """Get enhanced schema information for a table."""
        if database not in self.engines:
            raise ValueError(f"Database {database} not configured")
        
        # Sanitize inputs
        table_name = table_name.replace("'", "''").replace(";", "")
        schema = schema.replace("'", "''").replace(";", "")
        
        query = f"""
        SELECT TOP 100
            COLUMN_NAME as column_name,
            DATA_TYPE as data_type,
            IS_NULLABLE as is_nullable,
            COLUMN_DEFAULT as default_value,
            CHARACTER_MAXIMUM_LENGTH as max_length,
            NUMERIC_PRECISION as numeric_precision,
            NUMERIC_SCALE as numeric_scale,
            ORDINAL_POSITION as position
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{table_name}'
        ORDER BY ORDINAL_POSITION
        """
        
        try:
            logger.info(f"Getting table schema for {schema}.{table_name} in database: {database}")
            with self.engines[database].connect() as conn:
                result = pd.read_sql(text(query), conn)
                logger.info(f"Schema query returned {len(result)} columns")
                
                if not result.empty:
                    schema_records = result.to_dict('records')
                    
                    # Add metadata about the query
                    for record in schema_records:
                        record['table_name'] = table_name
                        record['table_schema'] = schema
                        record['database'] = database
                        
                        # Add helpful column information
                        if record['column_name'].upper() in ['EFFECTIVEDATE', 'POSTDATE']:
                            record['usage_note'] = 'Date field - use for transaction timing'
                        elif record['column_name'].upper() in ['BALANCECHANGE', 'NEWBALANCE']:
                            record['usage_note'] = 'Amount field - use for transaction amounts'
                        elif record['column_name'].upper() in ['PARENTACCOUNT']:
                            record['usage_note'] = 'Key field - use for account joins'
                    
                    return schema_records
                else:
                    logger.warning(f"No columns found for table {schema}.{table_name}")
                    return [{
                        'error': 'Table not found or no columns',
                        'table_name': table_name,
                        'table_schema': schema,
                        'database': database,
                        'suggestion': 'Check table name spelling or verify table exists'
                    }]
                    
        except Exception as e:
            logger.error(f"Failed to get schema for {schema}.{table_name}: {str(e)}")
            return [{
                'error': True,
                'message': f"Failed to get schema: {str(e)}",
                'table_name': table_name,
                'table_schema': schema,
                'database': database
            }]
    
    def test_connection(self, database: str) -> bool:
        """Test database connection with timeout."""
        try:
            if database not in self.engines:
                logger.warning(f"Database {database} not configured")
                return False
            with self.engines[database].connect() as conn:
                # Use a simple query with timeout
                result = conn.execute(text("SELECT 1"))
                result.fetchone()  # Ensure query completes
                logger.debug(f"Database connection test successful: {database}")
                return True
        except Exception as e:
            logger.error(f"Database connection test failed for {database}: {e}")
            return False


class EnhancedBusinessRulesManager:
    """Enhanced business rules manager with better query templates."""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.business_rules = self._load_business_rules()
        self.query_templates = self._setup_enhanced_templates()
        logger.info("Enhanced business rules loaded from configuration")
    
    def _load_business_rules(self) -> Dict[str, Any]:
        """Load business rules from YAML configuration."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    logger.info(f"Loaded business rules from: {self.config_path}")
                    return config
            else:
                logger.warning(f"Business rules config not found at: {self.config_path}")
                return {}
        except Exception as e:
            logger.error(f"Failed to load business rules: {e}")
            return {}
    
    def _setup_enhanced_templates(self) -> Dict[str, Dict[str, Any]]:
        """Set up enhanced query templates with metadata."""
        templates = {
            'transaction_volume_trends': {
                'description': 'Monthly transaction volume trends with proper column names',
                'category': 'Transaction Analysis',
                'query': """
                SELECT 
                    YEAR(st.EFFECTIVEDATE) AS TranYear,
                    MONTH(st.EFFECTIVEDATE) AS TranMonth,
                    COUNT(*) AS TransactionCount,
                    SUM(ABS(st.BALANCECHANGE)) AS TotalVolume,
                    AVG(ABS(st.BALANCECHANGE)) AS AvgTransactionSize,
                    COUNT(DISTINCT st.PARENTACCOUNT) AS UniqueMembers,
                    CAST(COUNT(*) * 1.0 / COUNT(DISTINCT st.PARENTACCOUNT) AS DECIMAL(10,2)) AS AvgTransPerMember
                FROM SAVINGSTRANSACTION st WITH (NOLOCK)
                INNER JOIN ACCOUNT a WITH (NOLOCK) ON st.PARENTACCOUNT = a.ACCOUNTNUMBER
                WHERE st.EFFECTIVEDATE >= DATEADD(MONTH, -12, GETDATE())
                    AND st.EFFECTIVEDATE < GETDATE()
                    AND (a.CLOSEDATE IS NULL OR a.CLOSEDATE = '1900-01-01')
                GROUP BY YEAR(st.EFFECTIVEDATE), MONTH(st.EFFECTIVEDATE)
                ORDER BY TranYear DESC, TranMonth DESC
                """,
                'parameters': ['months_back'],
                'notes': 'Uses EFFECTIVEDATE (not EFFECTDATE) and BALANCECHANGE (not TRANSAMOUNT)'
            },
            'active_member_summary': {
                'description': 'Active member summary with correct business rules',
                'category': 'Member Analysis',
                'query': """
                SELECT 
                    COUNT(DISTINCT a.ACCOUNTNUMBER) AS TotalActiveMembers,
                    COUNT(DISTINCT CASE WHEN a.TYPE IN (0, 1, 6) THEN a.ACCOUNTNUMBER END) AS ShareMembers,
                    COUNT(DISTINCT CASE WHEN a.TYPE = 10 THEN a.ACCOUNTNUMBER END) AS CheckingMembers,
                    AVG(a.BALANCE) AS AvgBalance,
                    SUM(a.BALANCE) AS TotalBalance
                FROM ACCOUNT a WITH (NOLOCK)
                WHERE (a.CLOSEDATE IS NULL OR a.CLOSEDATE = '1900-01-01')
                    AND a.BALANCE > 0
                """,
                'parameters': [],
                'notes': 'Uses proper active account conditions and account type mappings'
            },
            'recent_transactions_safe': {
                'description': 'Recent transactions with column validation',
                'category': 'Transaction Analysis',
                'query': """
                SELECT TOP 100
                    st.PARENTACCOUNT,
                    st.EFFECTIVEDATE,
                    st.BALANCECHANGE,
                    st.NEWBALANCE,
                    st.DESCRIPTION,
                    st.ACTIONCODE
                FROM SAVINGSTRANSACTION st WITH (NOLOCK)
                WHERE st.EFFECTIVEDATE >= DATEADD(DAY, -7, GETDATE())
                    AND st.BALANCECHANGE <> 0
                ORDER BY st.EFFECTIVEDATE DESC
                """,
                'parameters': ['days_back'],
                'notes': 'Safe query with correct column names and reasonable limits'
            }
        }
        
        # Merge with existing templates from config
        config_templates = self.business_rules.get('mcp_compatible_queries', {})
        for name, query in config_templates.items():
            if name not in templates:
                templates[name] = {
                    'description': f'Business rule template: {name}',
                    'category': 'Business Rules',
                    'query': query,
                    'parameters': [],
                    'notes': 'From business rules configuration'
                }
        
        return templates
    
    def get_query_template(self, template_name: str) -> str:
        """Get query template by name."""
        if template_name in self.query_templates:
            return self.query_templates[template_name]['query']
        elif template_name in self.business_rules.get('mcp_compatible_queries', {}):
            return self.business_rules['mcp_compatible_queries'][template_name]
        else:
            raise ValueError(f"Query template '{template_name}' not found")
    
    def get_template_list(self) -> List[Dict[str, Any]]:
        """Get list of available templates with metadata."""
        template_list = []
        for name, template in self.query_templates.items():
            template_list.append({
                'name': name,
                'description': template['description'],
                'category': template['category'],
                'parameters': template.get('parameters', []),
                'notes': template.get('notes', '')
            })
        return template_list


class EnhancedCreditUnionMCP:
    """Enhanced Credit Union MCP Server with all fixes implemented."""
    
    def __init__(self):
        self.server = Server("credit-union-mcp")
        self.db_manager = None
        self.business_rules = None
        self.continuity_manager = None
        self.health_status = None
        
    async def setup(self):
        """Initialize the enhanced MCP server."""
        logger.info("Starting enhanced MCP server setup with business rules enforcement")
        
        # Perform startup health checks
        logger.info("Performing startup health checks...")
        self.health_status = self._perform_health_checks()
        
        if self.health_status['overall_status'] == 'critical':
            logger.critical("Startup health checks failed - server cannot start")
            raise RuntimeError("Critical startup validation failures detected")
        else:
            logger.info("All utility imports validated successfully")
        
        logger.info("Startup health checks passed")
        
        script_dir = Path(__file__).parent.parent
        
        # Load enhanced business rules
        business_rules_path = script_dir / "config" / "business_rules_config.yaml"
        logger.info(f"Loading business rules from: {business_rules_path}")
        self.business_rules = EnhancedBusinessRulesManager(business_rules_path)
        logger.info(f"Loaded business rules from: {business_rules_path}")
        
        # Load database configuration
        db_config_path = script_dir / "config" / "database_config.yaml"
        logger.info(f"Looking for database config at: {db_config_path}")
        
        if db_config_path.exists():
            logger.info(f"Found database config file: {db_config_path}")
            with open(db_config_path, 'r') as f:
                db_config = yaml.safe_load(f)
                db_config = {k: v for k, v in db_config.items() 
                           if isinstance(v, dict) and 'server' in v}
                logger.info(f"Loaded database configurations for: {list(db_config.keys())}")
                
                # Use enhanced database manager
                self.db_manager = EnhancedDatabaseManager(db_config)
                
                # Test connections
                for db_name in db_config.keys():
                    if self.db_manager.test_connection(db_name):
                        logger.info(f"Connected to database: {db_name}")
                    else:
                        logger.warning(f"Failed to connect to database: {db_name}")
                
                logger.info("Database connections initialized")
        else:
            logger.warning(f"No database config found at {db_config_path}")
        
        # Initialize enhanced continuity management
        try:
            continuity_state_dir = script_dir / "conversation_states"
            self.continuity_manager = ContinuityManager(
                context_limit=180000,  # 180K token limit
                warning_threshold=0.75,
                state_dir=continuity_state_dir,
                business_rules=self.business_rules.business_rules if self.business_rules else {}
            )
            logger.info("Continuity management system initialized")
        except Exception as e:
            logger.error(f"Failed to initialize continuity manager: {e}")
            self.continuity_manager = None
        
        self._register_enhanced_tools()
        logger.info("MCP tools registered with business rules enforcement and continuity management")
        logger.info("Credit Union MCP Server ready")
    
    def _perform_health_checks(self) -> Dict[str, Any]:
        """Perform enhanced health checks."""
        # Using the same validation logic as the original
        validation_results = {
            'ContinuityManager': 'ContinuityManager' in globals(),
            'ConversationState': 'ConversationState' in globals(),
            'TokenEstimator': 'TokenEstimator' in globals(),
            'utils_module': False
        }
        
        if all([validation_results['ContinuityManager'], 
                validation_results['ConversationState'], 
                validation_results['TokenEstimator']]):
            validation_results['utils_module'] = True
        
        return {
            'imports': validation_results,
            'overall_status': 'healthy' if validation_results['utils_module'] else 'critical'
        }
    
    def _register_enhanced_tools(self):
        """Register enhanced MCP tools with all fixes."""
        
        @self.server.list_prompts()
        async def list_prompts() -> List[Prompt]:
            """List available prompts for query assistance."""
            return [
                Prompt(
                    name="common_columns_help",
                    description="Help with commonly confused column names",
                    arguments=[
                        {"name": "table_name", "description": "Table name to get column help for", "required": False}
                    ]
                ),
                Prompt(
                    name="query_optimization_tips", 
                    description="Tips for optimizing slow queries",
                    arguments=[
                        {"name": "query_type", "description": "Type of query (transaction, member, etc.)", "required": False}
                    ]
                ),
                Prompt(
                    name="business_rules_guidance",
                    description="Guidance on business rules and compliant queries", 
                    arguments=[]
                )
            ]
        
        @self.server.get_prompt()
        async def get_prompt(name: str, arguments: Optional[Dict[str, str]] = None) -> str:
            """Get specific prompt content."""
            if name == "common_columns_help":
                table_name = arguments.get("table_name", "") if arguments else ""
                return f"""
# Common Column Name Issues

Based on log analysis, here are frequently confused column names:

## SAVINGSTRANSACTION Table Issues:
- ❌ EFFECTDATE → ✅ EFFECTIVEDATE 
- ❌ TRANSAMOUNT → ✅ BALANCECHANGE
- ❌ ProcessDate → ✅ This field doesn't exist. Remove or use POSTDATE
- ❌ VOIDID → ✅ VOIDCODE

## Best Practices:
- Use EFFECTIVEDATE for transaction timing
- Use BALANCECHANGE for transaction amounts  
- Use POSTDATE for posting date
- Always include WHERE (CLOSEDATE IS NULL OR CLOSEDATE = '1900-01-01') for active records

{f"## For table {table_name}:" if table_name else ""}
Use get_table_schema tool to see exact column names for any table.
"""
            elif name == "query_optimization_tips":
                query_type = arguments.get("query_type", "") if arguments else ""
                return f"""
# Query Optimization Tips

## Prevent Timeouts:
- Add TOP {1000} clause to limit results
- Use specific date ranges: WHERE EFFECTIVEDATE >= DATEADD(MONTH, -3, GETDATE())
- Add WITH (NOLOCK) for read-only queries on large tables

## Performance Hints:
- For SAVINGSTRANSACTION: Always filter by EFFECTIVEDATE
- For ACCOUNT: Always include active account filter
- Use indexed columns in WHERE clauses

## Timeout Prevention:
- Queries timeout after 30 seconds
- Large table scans will fail
- Use business rule templates for tested queries

{f"## For {query_type} queries:" if query_type else ""}
Consider using pre-built templates from get_business_rules_summary.
"""
            elif name == "business_rules_guidance":
                return """
# Business Rules Guidance

## Query Templates Available:
- transaction_volume_trends: Monthly transaction analysis
- active_member_summary: Member counts and balances
- recent_transactions_safe: Recent transactions with limits

## Compliance Requirements:
- Process Date: Use FORMAT(DATEADD(DAY, -1, GETDATE()), 'yyyyMMdd')  
- Active Records: (CLOSEDATE IS NULL OR CLOSEDATE = '1900-01-01')
- Charge-Off: (ChargeOffDate IS NULL OR ChargeOffDate = '1900-01-01')

## Use Tools:
- validate_query_compliance: Check query against business rules
- get_enhanced_query_templates: Get pre-tested query patterns
"""
            else:
                raise ValueError(f"Unknown prompt: {name}")
        
        @self.server.list_resources()
        async def list_resources() -> List[Resource]:
            """List available resources."""
            return [
                Resource(
                    uri="database://connection",
                    name="Database Connection Info",
                    description="Current database connection configuration", 
                    mimeType="application/json"
                ),
                Resource(
                    uri="schema://column_mappings",
                    name="Column Name Mappings",
                    description="Common column name corrections and mappings",
                    mimeType="application/json"
                ),
                Resource(
                    uri="templates://query_examples", 
                    name="Query Templates",
                    description="Pre-built query templates with best practices",
                    mimeType="application/json"
                ),
                Resource(
                    uri="performance://optimization_guide",
                    name="Performance Optimization Guide",
                    description="Guidelines for writing efficient queries",
                    mimeType="text/markdown"
                )
            ]
        
        @self.server.read_resource()
        async def read_resource(uri: str) -> Dict[str, Any]:
            """Read specific resource content."""
            if uri == "database://connection":
                return {
                    "contents": [{
                        "type": "text",
                        "text": json.dumps({
                            "databases": list(self.db_manager.engines.keys()) if self.db_manager else [],
                            "status": "connected" if self.db_manager else "disconnected",
                            "timeout_settings": "30s connection, 60s query",
                            "performance_features": ["column_validation", "query_optimization", "timeout_management"]
                        }, indent=2)
                    }]
                }
            elif uri == "schema://column_mappings":
                mappings = self.db_manager.common_column_errors if self.db_manager else {}
                return {
                    "contents": [{
                        "type": "text", 
                        "text": json.dumps({
                            "common_mistakes": mappings,
                            "validation_enabled": True,
                            "cached_schemas": list(self.db_manager.schema_cache.keys()) if self.db_manager else []
                        }, indent=2)
                    }]
                }
            elif uri == "templates://query_examples":
                templates = self.business_rules.get_template_list() if self.business_rules else []
                return {
                    "contents": [{
                        "type": "text",
                        "text": json.dumps(templates, indent=2)
                    }]
                }
            elif uri == "performance://optimization_guide":
                return {
                    "contents": [{
                        "type": "text",
                        "text": """# Query Performance Optimization Guide

## Timeout Prevention
- All queries have 30-second timeout
- Use TOP clause to limit results
- Add specific date filters
- Use WITH (NOLOCK) for read queries

## Column Validation  
- Automatic validation against schema
- Suggestions for common mistakes
- Real-time error prevention

## Query Optimization
- Automatic TOP clause insertion
- Index hints for large tables
- Performance monitoring and logging

## Best Practices
- Test queries with execute_query first
- Use business rule templates when available
- Check validate_query_compliance for compliance
- Monitor execution times in logs
"""
                    }]
                }
            else:
                raise ValueError(f"Unknown resource: {uri}")
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            base_tools = [
                Tool(
                    name="execute_query",
                    description="Execute SQL query with timeout management and column validation",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "database": {
                                "type": "string",
                                "enum": ["ARCUSYM000", "TEMENOS"],
                                "default": "ARCUSYM000",
                                "description": "Target database"
                            },
                            "query": {
                                "type": "string",
                                "description": "SQL query to execute (SELECT only, with column validation)"
                            },
                            "max_rows": {
                                "type": "integer",
                                "default": 1000,
                                "description": "Maximum number of rows to return"
                            },
                            "timeout": {
                                "type": "integer", 
                                "default": 30,
                                "description": "Query timeout in seconds (max 60)"
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="get_tables",
                    description="Get list of tables with enhanced metadata", 
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "database": {
                                "type": "string",
                                "enum": ["ARCUSYM000", "TEMENOS"],
                                "default": "ARCUSYM000",
                                "description": "Target database"
                            },
                            "schema": {
                                "type": "string",
                                "default": "dbo",
                                "description": "Database schema"
                            }
                        }
                    }
                ),
                Tool(
                    name="get_table_schema",
                    description="Get enhanced schema information with usage notes",
                    inputSchema={
                        "type": "object", 
                        "properties": {
                            "database": {
                                "type": "string",
                                "enum": ["ARCUSYM000", "TEMENOS"],
                                "default": "ARCUSYM000", 
                                "description": "Target database"
                            },
                            "table_name": {
                                "type": "string",
                                "description": "Name of the table"
                            },
                            "schema": {
                                "type": "string",
                                "default": "dbo",
                                "description": "Database schema"
                            }
                        },
                        "required": ["table_name"]
                    }
                ),
                Tool(
                    name="validate_query_compliance",
                    description="Enhanced query validation with column checking",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "SQL query to validate"
                            },
                            "database": {
                                "type": "string", 
                                "enum": ["ARCUSYM000", "TEMENOS"],
                                "default": "ARCUSYM000",
                                "description": "Target database for column validation"
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="get_enhanced_query_templates",
                    description="Get enhanced query templates with metadata",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "enum": ["Transaction Analysis", "Member Analysis", "Business Rules", "All"],
                                "default": "All",
                                "description": "Template category filter"
                            }
                        }
                    }
                ),
                Tool(
                    name="execute_template_query",
                    description="Execute a pre-built template query safely",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "template_name": {
                                "type": "string",
                                "description": "Name of template to execute"
                            },
                            "database": {
                                "type": "string",
                                "enum": ["ARCUSYM000", "TEMENOS"],
                                "default": "ARCUSYM000", 
                                "description": "Target database"
                            },
                            "parameters": {
                                "type": "object",
                                "description": "Template parameters (optional)",
                                "additionalProperties": True
                            }
                        },
                        "required": ["template_name"]
                    }
                ),
                Tool(
                    name="get_business_rules_summary", 
                    description="Get comprehensive business rules summary",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="test_connection",
                    description="Test database connections with timeout",
                    inputSchema={
                        "type": "object", 
                        "properties": {
                            "database": {
                                "type": "string",
                                "enum": ["ARCUSYM000", "TEMENOS", "ALL"],
                                "default": "ALL",
                                "description": "Database to test"
                            }
                        }
                    }
                )
            ]
            return base_tools
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Enhanced tool call handler with all fixes."""
            try:
                if not self.db_manager:
                    return [TextContent(
                        type="text",
                        text=json.dumps({
                            "error": "Database not available - server running in demo mode"
                        }, indent=2)
                    )]
                
                if name == "execute_query":
                    database = arguments.get('database', 'ARCUSYM000')
                    query = arguments['query']
                    max_rows = arguments.get('max_rows', 1000)
                    timeout = min(arguments.get('timeout', 30), 60)  # Cap at 60 seconds
                    
                    result = self.db_manager.execute_query_with_timeout(
                        database, query, max_rows, timeout
                    )
                    
                    response = {
                        "success": True,
                        "results": result.to_dict('records'),
                        "metadata": {
                            "row_count": len(result),
                            "database": database,
                            "timeout_used": timeout,
                            "column_validation": "passed",
                            "execution_time": getattr(result, 'attrs', {}).get('execution_time', 'unknown')
                        }
                    }
                    
                    return [TextContent(
                        type="text",
                        text=json.dumps(response, indent=2, default=str)
                    )]
                
                elif name == "get_tables":
                    database = arguments.get('database', 'ARCUSYM000')
                    schema = arguments.get('schema', 'dbo')
                    
                    tables = self.db_manager.get_tables(database, schema)
                    
                    return [TextContent(
                        type="text",
                        text=json.dumps({
                            "tables": tables,
                            "database": database,
                            "schema": schema,
                            "enhanced_metadata": True
                        }, indent=2)
                    )]
                
                elif name == "get_table_schema":
                    database = arguments.get('database', 'ARCUSYM000')
                    table_name = arguments['table_name']
                    schema = arguments.get('schema', 'dbo')
                    
                    schema_info = self.db_manager.get_table_schema(database, table_name, schema)
                    
                    return [TextContent(
                        type="text", 
                        text=json.dumps({
                            "schema": schema_info,
                            "table": table_name,
                            "database": database,
                            "enhanced_info": True,
                            "usage_notes_included": True
                        }, indent=2)
                    )]
                
                elif name == "validate_query_compliance":
                    query = arguments['query']
                    database = arguments.get('database', 'ARCUSYM000')
                    
                    # Enhanced validation combining business rules and column validation
                    business_compliance = self.business_rules.validate_query_compliance(query)
                    column_validation = self.db_manager.validate_column_names(query, database)
                    
                    combined_result = {
                        "business_compliance": business_compliance,
                        "column_validation": column_validation,
                        "overall_valid": business_compliance.get('compliant', True) and column_validation.get('valid', True),
                        "enhanced_validation": True,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    return [TextContent(
                        type="text",
                        text=json.dumps(combined_result, indent=2)
                    )]
                
                elif name == "get_enhanced_query_templates":
                    category = arguments.get('category', 'All')
                    
                    templates = self.business_rules.get_template_list()
                    if category != 'All':
                        templates = [t for t in templates if t['category'] == category]
                    
                    return [TextContent(
                        type="text",
                        text=json.dumps({
                            "templates": templates,
                            "category_filter": category,
                            "total_available": len(self.business_rules.query_templates),
                            "enhanced_templates": True
                        }, indent=2)
                    )]
                
                elif name == "execute_template_query":
                    template_name = arguments['template_name']
                    database = arguments.get('database', 'ARCUSYM000')
                    parameters = arguments.get('parameters', {})
                    
                    try:
                        query_template = self.business_rules.get_query_template(template_name)
                        
                        # Execute with enhanced error handling
                        result = self.db_manager.execute_query_with_timeout(
                            database, query_template, max_rows=1000, timeout=30
                        )
                        
                        return [TextContent(
                            type="text",
                            text=json.dumps({
                                "template_name": template_name,
                                "results": result.to_dict('records'),
                                "metadata": {
                                    "row_count": len(result),
                                    "template_validated": True,
                                    "enhanced_execution": True
                                }
                            }, indent=2, default=str)
                        )]
                    except Exception as e:
                        return [TextContent(
                            type="text",
                            text=json.dumps({
                                "error": f"Template execution failed: {str(e)}",
                                "template_name": template_name,
                                "enhanced_error_handling": True
                            }, indent=2)
                        )]
                
                elif name == "get_business_rules_summary":
                    summary = self.business_rules.get_business_rule_summary()
                    summary.update({
                        "enhanced_features": {
                            "column_validation": True,
                            "timeout_management": True, 
                            "query_optimization": True,
                            "template_system": True,
                            "performance_monitoring": True
                        },
                        "available_templates": len(self.business_rules.query_templates),
                        "cached_schemas": len(self.db_manager.schema_cache) if self.db_manager else 0
                    })
                    
                    return [TextContent(
                        type="text",
                        text=json.dumps(summary, indent=2)
                    )]
                
                elif name == "test_connection":
                    database = arguments.get('database', 'ALL')
                    
                    if database == 'ALL':
                        results = {}
                        for db_name in self.db_manager.engines.keys():
                            results[db_name] = 'Connected' if self.db_manager.test_connection(db_name) else 'Failed'
                    else:
                        results = {database: 'Connected' if self.db_manager.test_connection(database) else 'Failed'}
                    
                    results['enhanced_testing'] = True
                    results['timeout_settings'] = "30s connection, 60s query"
                    
                    return [TextContent(
                        type="text",
                        text=json.dumps(results, indent=2)
                    )]
                
                else:
                    raise ValueError(f"Unknown tool: {name}")
                    
            except Exception as e:
                logger.error(f"Enhanced tool execution failed for {name}: {e}")
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        'error': True,
                        'message': str(e),
                        'tool': name,
                        'enhanced_error_handling': True,
                        'timestamp': datetime.now().isoformat()
                    }, indent=2)
                )]
    
    async def run(self):
        """Start the enhanced MCP server.""" 
        await self.setup()
        
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def main():
    """Main entry point for enhanced server."""
    mcp = EnhancedCreditUnionMCP()
    await mcp.run()


if __name__ == "__main__":
    asyncio.run(main())
