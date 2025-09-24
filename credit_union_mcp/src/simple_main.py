"""
Simplified Credit Union MCP Server

A streamlined MCP server providing essential database access and analysis tools
for credit union operations without unnecessary complexity.
"""

import asyncio
import json
import yaml
import os
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import pyodbc
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings('ignore')


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
                print(f"Connected to {db_name}", flush=True)
            except Exception as e:
                print(f"Failed to connect to {db_name}: {e}", flush=True)
    
    def execute_query(self, database: str, query: str, max_rows: int = 10000) -> pd.DataFrame:
        """Execute SQL query and return DataFrame."""
        if database not in self.engines:
            raise ValueError(f"Database {database} not configured")
        
        # Simple query validation
        query_upper = query.upper().strip()
        if not query_upper.startswith('SELECT'):
            raise ValueError("Only SELECT queries are allowed")
        
        # Add row limit if not present
        if 'TOP ' not in query_upper and 'LIMIT ' not in query_upper:
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
        """Get schema information for a table."""
        if database not in self.engines:
            raise ValueError(f"Database {database} not configured")
        
        query = f"""
        SELECT 
            COLUMN_NAME as column_name,
            DATA_TYPE as data_type,
            IS_NULLABLE as is_nullable,
            COLUMN_DEFAULT as default_value,
            CHARACTER_MAXIMUM_LENGTH as max_length
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{table_name}'
        ORDER BY ORDINAL_POSITION
        """
        
        with self.engines[database].connect() as conn:
            result = pd.read_sql(text(query), conn)
            return result.to_dict('records')
    
    def test_connection(self, database: str) -> bool:
        """Test database connection."""
        try:
            if database not in self.engines:
                return False
            with self.engines[database].connect() as conn:
                conn.execute(text("SELECT 1"))
                return True
        except:
            return False


class SimpleAnalytics:
    """Simplified analytics functions."""
    
    def __init__(self, db_manager: SimpleDatabaseManager):
        self.db_manager = db_manager
    
    def get_active_members(self, database: str = "ARCUSYM000") -> Dict[str, Any]:
        """Get active members count and basic statistics."""
        query = """
        SELECT 
            COUNT(DISTINCT n.PARENTACCOUNT) as active_members,
            COUNT(DISTINCT a.ACCOUNTNUMBER) as total_accounts,
            AVG(CAST(a.BALANCE as FLOAT)) as avg_balance
        FROM NAME n
        INNER JOIN ACCOUNT a ON n.PARENTACCOUNT = a.ACCOUNTNUMBER
        WHERE n.TYPE = 0 
            AND a.CLOSEDATE = '19000101'
            AND n.SSN IS NOT NULL
        """
        
        try:
            result = self.db_manager.execute_query(database, query)
            if not result.empty:
                return {
                    'active_members': int(result.iloc[0]['active_members']),
                    'total_accounts': int(result.iloc[0]['total_accounts']),
                    'avg_balance': float(result.iloc[0]['avg_balance']) if result.iloc[0]['avg_balance'] else 0.0,
                    'analysis_date': datetime.now().strftime('%Y-%m-%d')
                }
        except Exception as e:
            return {'error': f"Analysis failed: {str(e)}"}
        
        return {'error': 'No data found'}
    
    def get_financial_summary(self, database: str = "ARCUSYM000") -> Dict[str, Any]:
        """Get basic financial metrics."""
        query = """
        SELECT 
            SUM(CAST(s.BALANCE as FLOAT)) as total_savings,
            SUM(CAST(l.BALANCE as FLOAT)) as total_loans,
            COUNT(DISTINCT s.PARENTACCOUNT) as savings_accounts,
            COUNT(DISTINCT l.PARENTACCOUNT) as loan_accounts
        FROM SAVINGS s
        FULL OUTER JOIN LOAN l ON s.PARENTACCOUNT = l.PARENTACCOUNT
        WHERE (s.CLOSEDATE = '19000101' OR s.CLOSEDATE IS NULL)
            AND (l.CLOSEDATE = '19000101' OR l.CLOSEDATE IS NULL)
        """
        
        try:
            result = self.db_manager.execute_query(database, query)
            if not result.empty:
                return {
                    'total_savings': float(result.iloc[0]['total_savings']) if result.iloc[0]['total_savings'] else 0.0,
                    'total_loans': float(result.iloc[0]['total_loans']) if result.iloc[0]['total_loans'] else 0.0,
                    'savings_accounts': int(result.iloc[0]['savings_accounts']) if result.iloc[0]['savings_accounts'] else 0,
                    'loan_accounts': int(result.iloc[0]['loan_accounts']) if result.iloc[0]['loan_accounts'] else 0,
                    'analysis_date': datetime.now().strftime('%Y-%m-%d')
                }
        except Exception as e:
            return {'error': f"Analysis failed: {str(e)}"}
        
        return {'error': 'No data found'}


class CreditUnionMCP:
    """Simplified Credit Union MCP Server."""
    
    def __init__(self):
        self.server = Server("credit-union-mcp")
        self.db_manager = None
        self.analytics = None
        
    async def setup(self):
        """Initialize the MCP server."""
        print("Starting simplified MCP server setup...", flush=True)
        
        # Load database configuration
        config_path = Path("config/database_config.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                db_config = yaml.safe_load(f)
                self.db_manager = SimpleDatabaseManager(db_config)
                self.analytics = SimpleAnalytics(self.db_manager)
                print("Database connections initialized", flush=True)
        else:
            print("No database config found - running in demo mode", flush=True)
        
        self._register_tools()
        print("MCP tools registered", flush=True)
    
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
                    name="get_active_members",
                    description="Get active members analysis",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "database": {
                                "type": "string",
                                "enum": ["ARCUSYM000"],
                                "default": "ARCUSYM000",
                                "description": "Target database"
                            }
                        }
                    }
                ),
                Tool(
                    name="get_financial_summary",
                    description="Get basic financial summary",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "database": {
                                "type": "string",
                                "enum": ["ARCUSYM000"],
                                "default": "ARCUSYM000",
                                "description": "Target database"
                            }
                        }
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
                
                elif name == "get_active_members":
                    database = arguments.get('database', 'ARCUSYM000')
                    result = self.analytics.get_active_members(database)
                    return [TextContent(
                        type="text",
                        text=json.dumps(result, indent=2)
                    )]
                
                elif name == "get_financial_summary":
                    database = arguments.get('database', 'ARCUSYM000')
                    result = self.analytics.get_financial_summary(database)
                    return [TextContent(
                        type="text",
                        text=json.dumps(result, indent=2)
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
        print("Credit Union MCP Server ready", flush=True)
        
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
