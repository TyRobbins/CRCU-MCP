"""
Credit Union Analytics MCP Server

Multi-Agentic Credit Union Analytics MCP Server providing comprehensive
financial analysis capabilities through specialized AI agents.
"""

import asyncio
import json
import yaml
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from loguru import logger

import sys
import os
# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.database.connection import DatabaseManager
from src.database.data_catalog import DataCatalog
from src.orchestration.coordinator import AgentCoordinator
from src.agents.base_agent import AnalysisContext


class CreditUnionMCP:
    """
    Main MCP Server for Credit Union Analytics.
    
    Provides a comprehensive suite of financial analysis tools through
    specialized AI agents and intelligent orchestration.
    """
    
    def __init__(self):
        """Initialize the MCP server."""
        self.server = Server("credit-union-analytics")
        self.db_manager: Optional[DatabaseManager] = None
        self.coordinator: Optional[AgentCoordinator] = None
        self._setup_logging()
        
    def _setup_logging(self):
        """Set up logging configuration."""
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Configure loguru
        logger.remove()  # Remove default handler
        logger.add(
            "logs/credit_union_mcp_{time}.log",
            rotation="500 MB",
            retention="10 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
        )
        logger.add(
            lambda msg: None,  # Disable console logging in production
            level="ERROR"
        )
        
        logger.info("Credit Union MCP Server initializing...")
    
    async def setup(self):
        """Initialize database connections and agents."""
        try:
            print("Starting MCP server setup...", file=sys.stderr)
            
            # Load database configuration (relative to project root, not src/)
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "database_config.yaml"
            print(f"Looking for config at: {config_path}", file=sys.stderr)
            
            if not config_path.exists():
                print(f"Warning: Database configuration not found at {config_path}", file=sys.stderr)
                print("Running in demo mode without database connectivity", file=sys.stderr)
                self.db_manager = None
                self.coordinator = None
            else:
                with open(config_path, 'r') as f:
                    db_config = yaml.safe_load(f)
                print("Database config loaded successfully", file=sys.stderr)
                
                try:
                    # Initialize database manager
                    self.db_manager = DatabaseManager(db_config)
                    print("Database manager initialized", file=sys.stderr)
                    logger.info("Database manager initialized")
                    
                    # Test database connections (but don't fail if they don't work)
                    for db_name in ['TEMENOS', 'ARCUSYM000']:
                        if db_name in db_config:
                            try:
                                connection_status = self.db_manager.test_connection(db_name)
                                if connection_status:
                                    print(f"Successfully connected to {db_name} database", file=sys.stderr)
                                    logger.info(f"Successfully connected to {db_name} database")
                                else:
                                    print(f"Failed to connect to {db_name} database", file=sys.stderr)
                                    logger.warning(f"Failed to connect to {db_name} database")
                            except Exception as e:
                                print(f"Error testing {db_name} connection: {e}", file=sys.stderr)
                                logger.error(f"Error testing {db_name} connection: {e}")
                    
                    # Initialize agent coordinator
                    self.coordinator = AgentCoordinator(self.db_manager)
                    print("Agent coordinator initialized", file=sys.stderr)
                    logger.info("Agent coordinator initialized")
                    
                except Exception as db_error:
                    print(f"Database initialization failed: {db_error}", file=sys.stderr)
                    logger.error(f"Database initialization failed: {db_error}")
                    # Continue without database connectivity
                    self.coordinator = None
                    print("Running in limited mode without database connectivity", file=sys.stderr)
            
            # Register MCP tools
            self._register_tools()
            print("MCP tools registered successfully", file=sys.stderr)
            logger.info("MCP tools registered")
            
        except Exception as e:
            print(f"Failed to setup MCP server: {e}", file=sys.stderr)
            logger.error(f"Failed to setup MCP server: {e}")
            raise
    
    def _register_tools(self):
        """Register all MCP tools."""
        
        @self.server.list_resources()
        async def list_resources():
            """List available resources."""
            return [
                {
                    "uri": "database://connection",
                    "name": "Database Connection Info",
                    "description": "Current database connection configuration",
                    "mimeType": "application/json"
                }
            ]
        
        @self.server.read_resource()
        async def read_resource(uri: str):
            """Read a specific resource."""
            if uri == "database://connection":
                return {
                    "contents": [
                        {
                            "type": "text",
                            "text": json.dumps({
                                "databases": list(self.db_manager.engines.keys()) if self.db_manager else [],
                                "status": "connected" if self.db_manager else "disconnected",
                                "available_agents": len(self.coordinator.agents) if self.coordinator else 0
                            }, indent=2)
                        }
                    ]
                }
            else:
                raise ValueError(f"Unknown resource: {uri}")
        
        @self.server.list_prompts()
        async def list_prompts():
            """List available prompts."""
            return []
        
        @self.server.get_prompt()
        async def get_prompt(name: str, arguments: Optional[Dict[str, str]] = None):
            """Get a specific prompt."""
            raise ValueError(f"Unknown prompt: {name}")
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List all available tools."""
            return [
                # Database Tools
                Tool(
                    name="execute_query",
                    description="Execute SQL query on TEMENOS or ARCUSYM000 database",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "database": {
                                "type": "string",
                                "enum": ["TEMENOS", "ARCUSYM000"],
                                "default": "ARCUSYM000",
                                "description": "Target database (TEMENOS or ARCUSYM000)"
                            },
                            "query": {
                                "type": "string",
                                "description": "SQL query to execute"
                            },
                            "max_rows": {
                                "type": "integer",
                                "default": 10000,
                                "description": "Maximum number of rows to return"
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="get_tables",
                    description="Get list of tables in the database",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "database": {
                                "type": "string",
                                "enum": ["TEMENOS", "ARCUSYM000"],
                                "default": "ARCUSYM000",
                                "description": "Target database (ARCUSYM000 primary, TEMENOS for specialized workflows)"
                            },
                            "schema": {
                                "type": "string",
                                "default": "dbo",
                                "description": "Database schema name"
                            }
                        }
                    }
                ),
                Tool(
                    name="get_table_schema",
                    description="Get schema information for a specific table",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "database": {
                                "type": "string",
                                "enum": ["TEMENOS", "ARCUSYM000"],
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
                                "description": "Database schema name"
                            }
                        },
                        "required": ["table_name"]
                    }
                ),
                
                # Agent Analysis Tools
                Tool(
                    name="analyze_financial_performance",
                    description="Analyze financial performance metrics (ROA, ROE, NIM, etc.)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "database": {
                                "type": "string",
                                "enum": ["TEMENOS", "ARCUSYM000"],
                                "default": "ARCUSYM000",
                                "description": "Target database"
                            },
                            "metric_type": {
                                "type": "string",
                                "enum": ["profitability", "capital", "asset_quality"],
                                "default": "profitability",
                                "description": "Type of financial metrics to analyze"
                            },
                            "as_of_date": {
                                "type": "string",
                                "description": "Analysis date (YYYY-MM-DD format). Defaults to current date if not provided"
                            }
                        }
                    }
                ),
                Tool(
                    name="analyze_portfolio_risk",
                    description="Analyze loan portfolio risk (concentration, delinquency, stress testing)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "database": {
                                "type": "string",
                                "enum": ["TEMENOS", "ARCUSYM000"],
                                "default": "ARCUSYM000",
                                "description": "Target database"
                            },
                            "analysis_type": {
                                "type": "string",
                                "enum": ["concentration", "delinquency", "stress_test"],
                                "default": "concentration",
                                "description": "Type of risk analysis to perform"
                            },
                            "as_of_date": {
                                "type": "string",
                                "description": "Analysis date (YYYY-MM-DD format)"
                            }
                        }
                    }
                ),
                Tool(
                    name="analyze_member_segments",
                    description="Member segmentation and behavior analysis",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "database": {
                                "type": "string",
                                "enum": ["ARCUSYM000"],
                                "default": "ARCUSYM000",
                                "description": "Target database (ARCUSYM000 for member data)"
                            },
                            "method": {
                                "type": "string",
                                "enum": ["rfm", "clustering", "lifetime_value", "churn_prediction", "active_members"],
                                "default": "active_members",
                                "description": "Analysis method to use"
                            },
                            "as_of_date": {
                                "type": "string",
                                "description": "Analysis date (YYYY-MM-DD format)"
                            }
                        }
                    }
                ),
                Tool(
                    name="get_active_members",
                    description="Get current active members using CRCU business rules with PII protection",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "database": {
                                "type": "string",
                                "enum": ["ARCUSYM000"],
                                "default": "ARCUSYM000",
                                "description": "Target database (ARCUSYM000 for CRCU active members query)"
                            },
                            "as_of_date": {
                                "type": "string",
                                "description": "Analysis date (YYYY-MM-DD format). Defaults to current date if not provided"
                            },
                            "include_insights": {
                                "type": "boolean",
                                "default": True,
                                "description": "Include analysis insights and recommendations"
                            }
                        }
                    }
                ),
                Tool(
                    name="check_compliance",
                    description="Regulatory compliance monitoring and analysis",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "database": {
                                "type": "string",
                                "enum": ["ARCUSYM000"],
                                "default": "ARCUSYM000",
                                "description": "Target database"
                            },
                            "check_type": {
                                "type": "string",
                                "enum": ["capital", "bsa_aml", "lending_limits"],
                                "default": "capital",
                                "description": "Type of compliance check to perform"
                            },
                            "as_of_date": {
                                "type": "string",
                                "description": "Analysis date (YYYY-MM-DD format)"
                            }
                        }
                    }
                ),
                Tool(
                    name="analyze_operations",
                    description="Operations efficiency and performance analysis",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "database": {
                                "type": "string",
                                "enum": ["ARCUSYM000"],
                                "default": "ARCUSYM000",
                                "description": "Target database"
                            },
                            "focus_area": {
                                "type": "string",
                                "enum": ["branch", "channel", "staff"],
                                "default": "branch",
                                "description": "Operational area to focus on"
                            },
                            "as_of_date": {
                                "type": "string",
                                "description": "Analysis date (YYYY-MM-DD format)"
                            }
                        }
                    }
                ),
                
                # Multi-Agent Analysis
                Tool(
                    name="comprehensive_analysis",
                    description="Run comprehensive analysis across all agents",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "database": {
                                "type": "string",
                                "enum": ["ARCUSYM000"],
                                "default": "ARCUSYM000",
                                "description": "Target database"
                            },
                            "as_of_date": {
                                "type": "string",
                                "description": "Analysis date (YYYY-MM-DD format)"
                            },
                            "agents": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["member_analytics", "compliance", "operations"]
                                },
                                "description": "Specific agents to run (optional, runs all if not specified)"
                            }
                        }
                    }
                ),
                
                # Utility Tools
                Tool(
                    name="test_connection",
                    description="Test database connections to both systems",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "database": {
                                "type": "string",
                                "enum": ["TEMENOS", "ARCUSYM000", "ALL"],
                                "default": "ALL",
                                "description": "Database to test (or ALL for both)"
                            }
                        }
                    }
                ),
                Tool(
                    name="get_agent_capabilities",
                    description="Get capabilities of all available agents",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="health_check",
                    description="Perform health check on all system components",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls."""
            try:
                logger.info(f"Tool called: {name} with arguments: {arguments}")
                
                # Database tools
                if name == "execute_query":
                    return await self._handle_execute_query(arguments)
                elif name == "get_tables":
                    return await self._handle_get_tables(arguments)
                elif name == "get_table_schema":
                    return await self._handle_get_table_schema(arguments)
                elif name == "test_connection":
                    return await self._handle_test_connection(arguments)
                
                # Agent analysis tools
                elif name == "analyze_financial_performance":
                    return await self._handle_agent_analysis("financial_performance", arguments)
                elif name == "analyze_portfolio_risk":
                    return await self._handle_agent_analysis("portfolio_risk", arguments)
                elif name == "analyze_member_segments":
                    return await self._handle_agent_analysis("member_analytics", arguments)
                elif name == "get_active_members":
                    return await self._handle_active_members_analysis(arguments)
                elif name == "check_compliance":
                    return await self._handle_agent_analysis("compliance", arguments)
                elif name == "analyze_operations":
                    return await self._handle_agent_analysis("operations", arguments)
                elif name == "comprehensive_analysis":
                    return await self._handle_comprehensive_analysis(arguments)
                
                # Utility tools
                elif name == "get_agent_capabilities":
                    return await self._handle_get_capabilities()
                elif name == "health_check":
                    return await self._handle_health_check()
                
                else:
                    raise ValueError(f"Unknown tool: {name}")
                    
            except Exception as e:
                logger.error(f"Tool execution failed for {name}: {e}")
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        'error': True,
                        'message': str(e),
                        'tool': name,
                        'timestamp': datetime.now().isoformat()
                    }, indent=2)
                )]
    
    async def _handle_execute_query(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle SQL query execution."""
        if not self.db_manager:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": "Database not available",
                    "message": "Server running in demo mode without database connectivity"
                }, indent=2)
            )]
        
        database = arguments.get('database', 'ARCUSYM000')
        query = arguments['query']
        max_rows = arguments.get('max_rows', 10000)
        
        # Validate query is read-only
        if not self.db_manager.validate_query(query):
            raise ValueError("Only SELECT queries are allowed")
        
        result = self.db_manager.execute_query(database, query, max_rows=max_rows)
        
        return [TextContent(
            type="text",
            text=result.to_json(orient='records', indent=2)
        )]
    
    async def _handle_get_tables(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle get tables request."""
        database = arguments.get('database', 'ARCUSYM000')
        schema = arguments.get('schema', 'dbo')
        
        tables = self.db_manager.get_tables(database, schema)
        
        return [TextContent(
            type="text",
            text=json.dumps(tables, indent=2)
        )]
    
    async def _handle_get_table_schema(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle get table schema request."""
        database = arguments.get('database', 'TEMENOS')
        table_name = arguments['table_name']
        schema = arguments.get('schema', 'dbo')
        
        table_schema = self.db_manager.get_table_schema(database, table_name, schema)
        
        return [TextContent(
            type="text",
            text=json.dumps(table_schema, indent=2)
        )]
    
    async def _handle_test_connection(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle connection test request."""
        database = arguments.get('database', 'ALL')
        
        results = {}
        
        if database == 'ALL':
            databases_to_test = ['TEMENOS', 'ARCUSYM000']
        else:
            databases_to_test = [database]
        
        for db in databases_to_test:
            try:
                status = self.db_manager.test_connection(db)
                results[db] = 'Connected' if status else 'Failed'
            except Exception as e:
                results[db] = f'Error: {str(e)}'
        
        return [TextContent(
            type="text",
            text=json.dumps(results, indent=2)
        )]
    
    async def _handle_agent_analysis(self, agent_type: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle single agent analysis request."""
        database = arguments.get('database', 'TEMENOS')
        as_of_date = arguments.get('as_of_date')
        
        # Create analysis context
        context = AnalysisContext(
            agent_type=agent_type,
            database=database,
            as_of_date=as_of_date,
            parameters={k: v for k, v in arguments.items() if k not in ['database', 'as_of_date']}
        )
        
        # Execute analysis
        result = await self.coordinator.route_request(context)
        
        # Convert result to JSON
        result_dict = {
            'agent': result.agent,
            'analysis_type': result.analysis_type,
            'timestamp': result.timestamp,
            'success': result.success,
            'data': result.data,
            'metrics': result.metrics,
            'warnings': result.warnings,
            'errors': result.errors,
            'metadata': result.metadata
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(result_dict, indent=2, default=str)
        )]
    
    async def _handle_active_members_analysis(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle active members analysis request."""
        database = arguments.get('database', 'ARCUSYM000')
        as_of_date = arguments.get('as_of_date')
        include_insights = arguments.get('include_insights', True)
        
        # Create analysis context for active members analysis
        context = AnalysisContext(
            agent_type='member_analytics',
            database=database,
            as_of_date=as_of_date,
            parameters={
                'method': 'active_members',
                'include_insights': include_insights
            }
        )
        
        # Execute active members analysis
        result = await self.coordinator.route_request(context)
        
        # Convert result to JSON
        result_dict = {
            'agent': result.agent,
            'analysis_type': result.analysis_type,
            'timestamp': result.timestamp,
            'success': result.success,
            'data': result.data,
            'metrics': result.metrics,
            'warnings': result.warnings,
            'errors': result.errors,
            'metadata': result.metadata
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(result_dict, indent=2, default=str)
        )]
    
    async def _handle_comprehensive_analysis(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle comprehensive multi-agent analysis."""
        database = arguments.get('database', 'TEMENOS')
        as_of_date = arguments.get('as_of_date')
        specific_agents = arguments.get('agents')
        
        # Create analysis context for multi-agent analysis
        context = AnalysisContext(
            agent_type='multi_agent',
            database=database,
            as_of_date=as_of_date,
            parameters={'agents': specific_agents} if specific_agents else {}
        )
        
        # Execute comprehensive analysis
        result = await self.coordinator.route_request(context)
        
        # Convert result to JSON
        result_dict = {
            'agent': result.agent,
            'analysis_type': result.analysis_type,
            'timestamp': result.timestamp,
            'success': result.success,
            'data': result.data,
            'metrics': result.metrics,
            'warnings': result.warnings,
            'errors': result.errors,
            'metadata': result.metadata
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(result_dict, indent=2, default=str)
        )]
    
    async def _handle_get_capabilities(self) -> List[TextContent]:
        """Handle get agent capabilities request."""
        capabilities = self.coordinator.get_agent_capabilities()
        
        return [TextContent(
            type="text",
            text=json.dumps(capabilities, indent=2)
        )]
    
    async def _handle_health_check(self) -> List[TextContent]:
        """Handle health check request."""
        health_status = await self.coordinator.health_check()
        
        return [TextContent(
            type="text",
            text=json.dumps(health_status, indent=2)
        )]
    
    async def run(self):
        """Start the MCP server."""
        try:
            await self.setup()
            logger.info("Credit Union MCP Server setup complete")
            
            async with stdio_server() as (read_stream, write_stream):
                logger.info("MCP Server starting...")
                await self.server.run(
                    read_stream,
                    write_stream,
                    self.server.create_initialization_options()
                )
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            raise


async def main():
    """Main entry point."""
    mcp = CreditUnionMCP()
    await mcp.run()


if __name__ == "__main__":
    asyncio.run(main())
