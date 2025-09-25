"""
Simplified Credit Union MCP Server

A streamlined MCP server providing essential database access and analysis tools
for credit union operations without unnecessary complexity.
"""

import asyncio
import json
import yaml
import os
import sys
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, Prompt, Resource
import pyodbc
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings('ignore')

# Import utility classes with fallback for different execution contexts
try:
    # Try relative import first (when run as module)
    from .utils import ContinuityManager, ConversationState, TokenEstimator
except ImportError:
    try:
        # Try absolute import (when run as script)
        from credit_union_mcp.src.utils import ContinuityManager, ConversationState, TokenEstimator
    except ImportError:
        # Final fallback - add parent directory to path
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


class SimpleDatabaseManager:
    """Simplified database connection manager."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.engines = {}
        self._initialize_connections()
    
    def _initialize_connections(self):
        """Initialize database connections."""
        for db_name, db_config in self.config.items():
            try:
                connection_string = (
                    f"mssql+pyodbc://{db_config['username']}:{db_config['password']}"
                    f"@{db_config['server']}/{db_config['database']}"
                    f"?driver=ODBC+Driver+17+for+SQL+Server"
                )
                engine = create_engine(connection_string, pool_pre_ping=True)
                self.engines[db_name] = engine
                logger.info(f"Connected to database: {db_name}")
            except Exception as e:
                logger.error(f"Failed to connect to {db_name}: {e}")
    
    def execute_query(self, database: str, query: str, max_rows: int = 10000) -> pd.DataFrame:
        """Execute SQL query and return DataFrame."""
        if database not in self.engines:
            raise ValueError(f"Database {database} not configured")
        
        # Simple query validation - allow SELECT queries and CTEs
        query_upper = query.upper().strip()
        if not (query_upper.startswith('SELECT') or query_upper.startswith('WITH')):
            raise ValueError("Only SELECT queries and CTEs are allowed")
        
        # Add row limit if not present (but be more careful about complex queries)
        if 'TOP ' not in query_upper and 'LIMIT ' not in query_upper:
            # For simple queries, add TOP clause
            if query_upper.startswith('SELECT') and 'FROM' in query_upper:
                query = query.replace('SELECT', f'SELECT TOP {max_rows}', 1)
        
        with self.engines[database].connect() as conn:
            result = pd.read_sql(text(query), conn)
            return result
    
    def get_tables(self, database: str, schema: str = 'dbo') -> List[str]:
        """Get list of tables in database."""
        if database not in self.engines:
            raise ValueError(f"Database {database} not configured")
        
        query = f"""
        SELECT TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_SCHEMA = '{schema}' AND TABLE_TYPE = 'BASE TABLE'
        ORDER BY TABLE_NAME
        """
        
        with self.engines[database].connect() as conn:
            result = pd.read_sql(text(query), conn)
            return result['TABLE_NAME'].tolist()
    
    def get_table_schema(self, database: str, table_name: str, schema: str = 'dbo') -> List[Dict[str, Any]]:
        """Get schema information for a table with enhanced error handling."""
        if database not in self.engines:
            raise ValueError(f"Database {database} not configured")
        
        # Sanitize inputs to prevent SQL injection
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
            NUMERIC_SCALE as numeric_scale
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
                    
                    return schema_records
                else:
                    logger.warning(f"No columns found for table {schema}.{table_name}")
                    return []
                    
        except Exception as e:
            logger.error(f"Failed to get schema for {schema}.{table_name}: {str(e)}")
            # Return error information in a structured format
            return [{
                'error': True,
                'message': f"Failed to get schema: {str(e)}",
                'table_name': table_name,
                'table_schema': schema,
                'database': database
            }]
    
    def test_connection(self, database: str) -> bool:
        """Test database connection."""
        try:
            if database not in self.engines:
                logger.warning(f"Database {database} not configured")
                return False
            with self.engines[database].connect() as conn:
                conn.execute(text("SELECT 1"))
                logger.debug(f"Database connection test successful: {database}")
                return True
        except Exception as e:
            logger.error(f"Database connection test failed for {database}: {e}")
            return False
    
    def get_connection_status(self) -> Dict[str, str]:
        """Get status of all database connections."""
        status = {}
        for db_name in self.engines.keys():
            status[db_name] = 'Connected' if self.test_connection(db_name) else 'Failed'
        return status


class BusinessRulesManager:
    """Manages and enforces business rules from configuration."""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.business_rules = self._load_business_rules()
        logger.info("Business rules loaded from configuration")
    
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
    
    def get_query_template(self, template_name: str) -> str:
        """Get MCP-compatible query template by name."""
        try:
            templates = self.business_rules.get('mcp_compatible_queries', {})
            if template_name in templates:
                return templates[template_name]
            else:
                raise ValueError(f"Query template '{template_name}' not found")
        except Exception as e:
            logger.error(f"Failed to get query template {template_name}: {e}")
            raise
    
    def get_process_date_formula(self) -> str:
        """Get the definitive process date calculation formula."""
        try:
            return self.business_rules.get('definitive_logic_components', {}).get(
                'process_date_calculation', {}).get('formula', 
                "FORMAT(DATEADD(DAY, -1, GETDATE()), 'yyyyMMdd')"
            )
        except Exception as e:
            logger.error(f"Failed to get process date formula: {e}")
            return "FORMAT(DATEADD(DAY, -1, GETDATE()), 'yyyyMMdd')"
    
    def validate_query_compliance(self, query: str) -> Dict[str, Any]:
        """Validate query compliance with business rules."""
        compliance_result = {
            'compliant': True,
            'warnings': [],
            'errors': []
        }
        
        query_upper = query.upper()
        
        # Check for process date filter
        process_date_pattern = "PROCESSDATE = FORMAT(DATEADD(DAY, -1, GETDATE()), 'YYYYMMDD')"
        if 'PROCESSDATE' in query_upper and process_date_pattern.upper() not in query_upper:
            compliance_result['warnings'].append(
                "Query should use definitive process date formula: FORMAT(DATEADD(DAY, -1, GETDATE()), 'yyyyMMdd')"
            )
        
        # Check for proper active record conditions
        if 'CLOSEDATE' in query_upper:
            if not ("CLOSEDATE IS NULL" in query_upper or "CLOSEDATE = '1900-01-01'" in query_upper):
                compliance_result['warnings'].append(
                    "Active record condition should be: (CloseDate IS NULL OR CloseDate = '1900-01-01')"
                )
        
        if 'CHARGEOFFDATE' in query_upper:
            if not ("CHARGEOFFDATE IS NULL" in query_upper or "CHARGEOFFDATE = '1900-01-01'" in query_upper):
                compliance_result['warnings'].append(
                    "Charge-off condition should be: (ChargeOffDate IS NULL OR ChargeOffDate = '1900-01-01')"
                )
        
        return compliance_result
    
    def get_account_type_mappings(self) -> Dict[int, str]:
        """Get account type code to description mappings."""
        try:
            mappings = self.business_rules.get('business_rules', {}).get(
                'active_member_definition', {}).get('account_type_mappings', {})
            return {int(k): v for k, v in mappings.items()}
        except Exception as e:
            logger.error(f"Failed to get account type mappings: {e}")
            return {}
    
    def get_business_rule_summary(self) -> Dict[str, Any]:
        """Get summary of loaded business rules."""
        try:
            return {
                'config_loaded': bool(self.business_rules),
                'config_path': str(self.config_path),
                'available_query_templates': list(self.business_rules.get('mcp_compatible_queries', {}).keys()),
                'process_date_formula': self.get_process_date_formula(),
                'account_type_count': len(self.get_account_type_mappings()),
                'last_validated': self.business_rules.get('validation_status', {}).get('last_validated', 'Unknown')
            }
        except Exception as e:
            logger.error(f"Failed to get business rule summary: {e}")
            return {'error': str(e)}


def validate_imports() -> Dict[str, bool]:
    """Validate that all required imports are available."""
    validation_results = {
        'ContinuityManager': False,
        'ConversationState': False,
        'TokenEstimator': False,
        'utils_module': False
    }
    
    # Check if classes are available in globals (they should be imported at module level)
    try:
        # Test if the classes are accessible
        if 'ContinuityManager' in globals():
            validation_results['ContinuityManager'] = True
        if 'ConversationState' in globals():
            validation_results['ConversationState'] = True  
        if 'TokenEstimator' in globals():
            validation_results['TokenEstimator'] = True
        
        # If all classes are available, mark utils_module as successful
        if all([validation_results['ContinuityManager'], 
                validation_results['ConversationState'], 
                validation_results['TokenEstimator']]):
            validation_results['utils_module'] = True
            logger.info("All utility imports validated successfully")
        else:
            logger.warning("Some utility imports are missing from globals")
            
    except Exception as e:
        logger.error(f"Import validation failed: {e}")
    
    return validation_results


def perform_startup_health_checks() -> Dict[str, Any]:
    """Perform comprehensive startup health checks."""
    health_status = {
        'imports': validate_imports(),
        'python_version': {
            'version': sys.version,
            'major': sys.version_info.major,
            'minor': sys.version_info.minor,
            'compatible': sys.version_info >= (3, 8)
        },
        'dependencies': {},
        'overall_status': 'unknown'
    }
    
    # Check key dependencies
    dependencies = {
        'pandas': pd,
        'sqlalchemy': None,
        'pyodbc': pyodbc,
        'yaml': yaml,
        'mcp': None
    }
    
    for dep_name, module in dependencies.items():
        try:
            if module is not None:
                health_status['dependencies'][dep_name] = {
                    'available': True,
                    'version': getattr(module, '__version__', 'unknown')
                }
            else:
                # For modules that don't have a direct reference
                if dep_name == 'sqlalchemy':
                    from sqlalchemy import __version__
                    health_status['dependencies'][dep_name] = {
                        'available': True,
                        'version': __version__
                    }
                elif dep_name == 'mcp':
                    # MCP doesn't have a standard __version__
                    health_status['dependencies'][dep_name] = {
                        'available': True,
                        'version': 'available'
                    }
        except Exception as e:
            health_status['dependencies'][dep_name] = {
                'available': False,
                'error': str(e)
            }
    
    # Determine overall status
    import_failures = [k for k, v in health_status['imports'].items() if not v]
    dependency_failures = [k for k, v in health_status['dependencies'].items() 
                          if not v.get('available', False)]
    
    if not import_failures and not dependency_failures and health_status['python_version']['compatible']:
        health_status['overall_status'] = 'healthy'
    elif import_failures or not health_status['python_version']['compatible']:
        health_status['overall_status'] = 'critical'
    else:
        health_status['overall_status'] = 'warning'
    
    return health_status


class CreditUnionMCP:
    """Business Rules-Driven Credit Union MCP Server."""
    
    def __init__(self):
        self.server = Server("credit-union-mcp")
        self.db_manager = None
        self.business_rules = None
        self.continuity_manager = None
        self.health_status = None
        
    async def setup(self):
        """Initialize the MCP server with business rules enforcement."""
        logger.info("Starting MCP server setup with business rules enforcement")
        
        # Perform startup health checks first
        logger.info("Performing startup health checks...")
        self.health_status = perform_startup_health_checks()
        
        if self.health_status['overall_status'] == 'critical':
            logger.critical("Startup health checks failed - server cannot start")
            logger.critical(f"Import failures: {[k for k, v in self.health_status['imports'].items() if not v]}")
            logger.critical(f"Dependency failures: {[k for k, v in self.health_status['dependencies'].items() if not v.get('available', False)]}")
            raise RuntimeError("Critical startup validation failures detected")
        elif self.health_status['overall_status'] == 'warning':
            logger.warning("Startup health checks detected warnings - continuing with caution")
            logger.warning(f"Dependency issues: {[k for k, v in self.health_status['dependencies'].items() if not v.get('available', False)]}")
        else:
            logger.info("Startup health checks passed")
        
        script_dir = Path(__file__).parent.parent  # Go up to credit_union_mcp directory
        
        # Load business rules first
        business_rules_path = script_dir / "config" / "business_rules_config.yaml"
        logger.info(f"Loading business rules from: {business_rules_path}")
        self.business_rules = BusinessRulesManager(business_rules_path)
        
        # Load database configuration  
        db_config_path = script_dir / "config" / "database_config.yaml"
        logger.info(f"Looking for database config at: {db_config_path}")
        
        if db_config_path.exists():
            logger.info(f"Found database config file: {db_config_path}")
            with open(db_config_path, 'r') as f:
                db_config = yaml.safe_load(f)
                # Filter out non-database entries like 'connection_pool'
                db_config = {k: v for k, v in db_config.items() if isinstance(v, dict) and 'server' in v}
                logger.info(f"Loaded database configurations for: {list(db_config.keys())}")
                self.db_manager = SimpleDatabaseManager(db_config)
                logger.info("Database connections initialized")
        else:
            logger.warning(f"No database config found at {db_config_path} - running in demo mode")
        
        # Initialize continuity management system with enhanced error handling
        try:
            continuity_state_dir = script_dir / "conversation_states"
            self.continuity_manager = ContinuityManager(
                context_limit=180000,  # 180K tokens (90% of 200K)
                warning_threshold=0.75,
                state_dir=continuity_state_dir,
                business_rules=self.business_rules.business_rules if self.business_rules else {}
            )
            logger.info("Continuity management system initialized")
        except Exception as e:
            logger.error(f"Failed to initialize continuity manager: {e}")
            logger.warning("Server will continue without continuity management features")
            self.continuity_manager = None
        
        self._register_tools()
        logger.info("MCP tools registered with business rules enforcement and continuity management")
    
    def _register_tools(self):
        """Register MCP tools."""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="execute_query",
                    description="Execute SQL query on database",
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
                                "description": "SQL query to execute (SELECT only)"
                            },
                            "max_rows": {
                                "type": "integer",
                                "default": 1000,
                                "description": "Maximum number of rows to return"
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="get_tables",
                    description="Get list of tables in database",
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
                    description="Get schema for a specific table",
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
                    name="execute_business_rule_query",
                    description="Execute predefined business rule query from configuration",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "database": {
                                "type": "string",
                                "enum": ["ARCUSYM000", "TEMENOS"],
                                "default": "ARCUSYM000",
                                "description": "Target database"
                            },
                            "template_name": {
                                "type": "string",
                                "enum": [
                                    "simple_active_member_count",
                                    "member_breakdown_by_account_type", 
                                    "financial_summary_active_products"
                                ],
                                "description": "Name of the query template from business rules config"
                            },
                            "max_rows": {
                                "type": "integer",
                                "default": 1000,
                                "description": "Maximum number of rows to return"
                            }
                        },
                        "required": ["template_name"]
                    }
                ),
                Tool(
                    name="validate_query_compliance",
                    description="Validate query compliance with business rules",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "SQL query to validate against business rules"
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="get_business_rules_summary",
                    description="Get summary of loaded business rules and available templates",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="test_connection",
                    description="Test database connections",
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
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls."""
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
                    
                    result = self.db_manager.execute_query(database, query, max_rows)
                    return [TextContent(
                        type="text",
                        text=result.to_json(orient='records', indent=2)
                    )]
                
                elif name == "get_tables":
                    database = arguments.get('database', 'ARCUSYM000')
                    schema = arguments.get('schema', 'dbo')
                    
                    tables = self.db_manager.get_tables(database, schema)
                    return [TextContent(
                        type="text",
                        text=json.dumps(tables, indent=2)
                    )]
                
                elif name == "get_table_schema":
                    database = arguments.get('database', 'ARCUSYM000')
                    table_name = arguments['table_name']
                    schema = arguments.get('schema', 'dbo')
                    
                    schema_info = self.db_manager.get_table_schema(database, table_name, schema)
                    return [TextContent(
                        type="text",
                        text=json.dumps(schema_info, indent=2)
                    )]
                
                elif name == "execute_business_rule_query":
                    database = arguments.get('database', 'ARCUSYM000')
                    template_name = arguments['template_name']
                    max_rows = arguments.get('max_rows', 1000)
                    
                    try:
                        # Get query template from business rules
                        query_template = self.business_rules.get_query_template(template_name)
                        
                        # Execute query using database manager
                        result = self.db_manager.execute_query(database, query_template, max_rows)
                        
                        # Validate compliance
                        compliance = self.business_rules.validate_query_compliance(query_template)
                        
                        response = {
                            'template_name': template_name,
                            'query_executed': 'Business rule query template applied',
                            'compliance_check': compliance,
                            'results': result.to_dict('records') if not result.empty else [],
                            'row_count': len(result),
                            'business_rules_applied': True
                        }
                        
                        return [TextContent(
                            type="text",
                            text=json.dumps(response, indent=2, default=str)
                        )]
                    except Exception as e:
                        return [TextContent(
                            type="text",
                            text=json.dumps({
                                'error': f"Business rule query failed: {str(e)}",
                                'template_name': template_name,
                                'compliance_enforced': True
                            }, indent=2)
                        )]
                
                elif name == "validate_query_compliance":
                    query = arguments['query']
                    
                    try:
                        compliance_result = self.business_rules.validate_query_compliance(query)
                        compliance_result['business_rules_enforced'] = True
                        compliance_result['validation_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        
                        return [TextContent(
                            type="text",
                            text=json.dumps(compliance_result, indent=2)
                        )]
                    except Exception as e:
                        return [TextContent(
                            type="text",
                            text=json.dumps({
                                'error': f"Validation failed: {str(e)}",
                                'query_checked': query[:100] + "..." if len(query) > 100 else query
                            }, indent=2)
                        )]
                
                elif name == "get_business_rules_summary":
                    try:
                        summary = self.business_rules.get_business_rule_summary()
                        summary['server_mode'] = 'Business Rules Enforced'
                        summary['specialized_functions_removed'] = True
                        
                        return [TextContent(
                            type="text",
                            text=json.dumps(summary, indent=2)
                        )]
                    except Exception as e:
                        return [TextContent(
                            type="text",
                            text=json.dumps({
                                'error': f"Failed to get business rules summary: {str(e)}",
                                'fallback_info': 'Business rules configuration may not be loaded properly'
                            }, indent=2)
                        )]
                
                elif name == "test_connection":
                    database = arguments.get('database', 'ALL')
                    
                    if database == 'ALL':
                        results = {}
                        for db_name in self.db_manager.engines.keys():
                            results[db_name] = 'Connected' if self.db_manager.test_connection(db_name) else 'Failed'
                    else:
                        results = {database: 'Connected' if self.db_manager.test_connection(database) else 'Failed'}
                    
                    return [TextContent(
                        type="text",
                        text=json.dumps(results, indent=2)
                    )]
                
                elif name == "get_server_health":
                    try:
                        # Get current health status
                        current_health = self.health_status.copy() if self.health_status else {}
                        
                        # Add runtime status information
                        current_health['runtime_status'] = {
                            'database_manager': 'initialized' if self.db_manager else 'not_available',
                            'business_rules_manager': 'initialized' if self.business_rules else 'not_available',
                            'continuity_manager': 'initialized' if self.continuity_manager else 'not_available',
                            'server_startup_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        
                        # Add database connection status if available
                        if self.db_manager:
                            current_health['database_connections'] = self.db_manager.get_connection_status()
                        
                        # Add continuity manager status if available
                        if self.continuity_manager:
                            current_health['continuity_status'] = self.continuity_manager.get_context_status()
                        
                        return [TextContent(
                            type="text",
                            text=json.dumps(current_health, indent=2)
                        )]
                        
                    except Exception as e:
                        return [TextContent(
                            type="text",
                            text=json.dumps({
                                'error': f"Failed to get server health: {str(e)}",
                                'basic_status': 'server_running_but_health_check_failed'
                            }, indent=2)
                        )]
                
                else:
                    raise ValueError(f"Unknown tool: {name}")
                    
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        'error': True,
                        'message': str(e),
                        'tool': name
                    }, indent=2)
                )]
    
    async def run(self):
        """Start the MCP server."""
        await self.setup()
        logger.info("Credit Union MCP Server ready")
        
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def main():
    """Main entry point."""
    mcp = CreditUnionMCP()
    await mcp.run()


if __name__ == "__main__":
    asyncio.run(main())
