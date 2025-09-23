"""
Base Agent Framework for Credit Union Analytics

Provides abstract base class and common functionality for all specialized agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime
from loguru import logger
from pydantic import BaseModel, Field, validator
import json

from ..database.connection import DatabaseManager


class AnalysisContext(BaseModel):
    """
    Standard context structure for agent analysis requests.
    
    Provides validation and standardization of input parameters across all agents.
    """
    agent_type: Optional[str] = None
    analysis_type: Optional[str] = None
    as_of_date: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    database: str = Field(default="TEMENOS", description="Target database")
    
    @validator('as_of_date')
    def validate_date(cls, v):
        if v is not None:
            try:
                datetime.strptime(v, '%Y-%m-%d')
            except ValueError:
                raise ValueError('as_of_date must be in YYYY-MM-DD format')
        return v
    
    @validator('database')
    def validate_database(cls, v):
        if v not in ['TEMENOS', 'ARCUSYM000']:
            raise ValueError('database must be either TEMENOS or ARCUSYM000')
        return v


class AnalysisResult(BaseModel):
    """
    Standard result structure for agent analysis responses.
    
    Ensures consistent output format across all agents.
    """
    agent: str
    analysis_type: str
    timestamp: str
    success: bool
    data: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base class for all Credit Union analytics agents.
    
    Provides common functionality including:
    - Database connection management
    - Input validation
    - Output formatting
    - Error handling
    - Logging
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize the agent with database manager.
        
        Args:
            db_manager: DatabaseManager instance for data access
        """
        self.db = db_manager
        self.name = self.__class__.__name__
        self.logger = logger.bind(agent=self.name)
        
        # Cache for frequently used data
        self._cache: Dict[str, Any] = {}
        
        self.logger.info(f"Initialized {self.name}")
    
    @abstractmethod
    def analyze(self, context: AnalysisContext) -> AnalysisResult:
        """
        Main analysis method - must be implemented by each agent.
        
        Args:
            context: AnalysisContext containing request parameters
            
        Returns:
            AnalysisResult containing analysis results
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Return list of analysis capabilities supported by this agent.
        
        Returns:
            List of capability descriptions
        """
        pass
    
    def validate_context(self, context: AnalysisContext) -> bool:
        """
        Validate input context parameters.
        
        Args:
            context: AnalysisContext to validate
            
        Returns:
            True if context is valid
            
        Raises:
            ValueError: If validation fails
        """
        try:
            # Basic validation is handled by Pydantic model
            # Subclasses can override for additional validation
            return True
        except Exception as e:
            self.logger.error(f"Context validation failed: {e}")
            raise ValueError(f"Invalid context: {e}")
    
    def create_result(self, analysis_type: str, success: bool = True, 
                     data: Optional[Dict[str, Any]] = None,
                     metrics: Optional[Dict[str, float]] = None,
                     warnings: Optional[List[str]] = None,
                     errors: Optional[List[str]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> AnalysisResult:
        """
        Create standardized analysis result.
        
        Args:
            analysis_type: Type of analysis performed
            success: Whether analysis was successful
            data: Analysis results data
            metrics: Calculated metrics
            warnings: Warning messages
            errors: Error messages
            metadata: Additional metadata
            
        Returns:
            AnalysisResult instance
        """
        return AnalysisResult(
            agent=self.name,
            analysis_type=analysis_type,
            timestamp=datetime.now().isoformat(),
            success=success,
            data=data or {},
            metrics=metrics or {},
            warnings=warnings or [],
            errors=errors or [],
            metadata=metadata or {}
        )
    
    def execute_query(self, query: str, database: str = "TEMENOS", 
                     params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Execute database query with error handling and logging.
        
        Args:
            query: SQL query string
            database: Target database
            params: Query parameters
            
        Returns:
            DataFrame with query results
        """
        try:
            self.logger.debug(f"Executing query on {database}")
            result = self.db.execute_query(database, query, params)
            self.logger.info(f"Query returned {len(result)} rows")
            return result
            
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            raise
    
    def get_cached_data(self, key: str) -> Optional[Any]:
        """
        Retrieve data from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data or None if not found
        """
        return self._cache.get(key)
    
    def set_cached_data(self, key: str, data: Any, ttl_minutes: int = 30) -> None:
        """
        Store data in cache with TTL.
        
        Args:
            key: Cache key
            data: Data to cache
            ttl_minutes: Time to live in minutes
        """
        self._cache[key] = {
            'data': data,
            'timestamp': datetime.now(),
            'ttl_minutes': ttl_minutes
        }
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self.logger.info("Cache cleared")
    
    def _is_cache_valid(self, key: str) -> bool:
        """
        Check if cached data is still valid.
        
        Args:
            key: Cache key
            
        Returns:
            True if cache is valid
        """
        if key not in self._cache:
            return False
        
        cache_entry = self._cache[key]
        cache_time = cache_entry['timestamp']
        ttl_minutes = cache_entry['ttl_minutes']
        
        elapsed_minutes = (datetime.now() - cache_time).total_seconds() / 60
        return elapsed_minutes < ttl_minutes
    
    def calculate_percentage_change(self, current: float, previous: float) -> float:
        """
        Calculate percentage change between two values.
        
        Args:
            current: Current value
            previous: Previous value
            
        Returns:
            Percentage change
        """
        if previous == 0:
            return float('inf') if current > 0 else float('-inf') if current < 0 else 0
        return ((current - previous) / previous) * 100
    
    def format_currency(self, amount: float) -> str:
        """
        Format amount as currency string.
        
        Args:
            amount: Amount to format
            
        Returns:
            Formatted currency string
        """
        return f"${amount:,.2f}"
    
    def format_percentage(self, value: float, decimal_places: int = 2) -> str:
        """
        Format value as percentage string.
        
        Args:
            value: Value to format (as decimal, e.g., 0.05 for 5%)
            decimal_places: Number of decimal places
            
        Returns:
            Formatted percentage string
        """
        return f"{value:.{decimal_places}f}%"
    
    def safe_divide(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """
        Perform safe division with default value for division by zero.
        
        Args:
            numerator: Numerator value
            denominator: Denominator value  
            default: Default value if denominator is zero
            
        Returns:
            Division result or default value
        """
        return numerator / denominator if denominator != 0 else default
    
    def handle_analysis_error(self, analysis_type: str, error: Exception) -> AnalysisResult:
        """
        Handle analysis errors and create error result.
        
        Args:
            analysis_type: Type of analysis that failed
            error: Exception that occurred
            
        Returns:
            AnalysisResult with error information
        """
        self.logger.error(f"{analysis_type} analysis failed: {error}")
        
        return self.create_result(
            analysis_type=analysis_type,
            success=False,
            errors=[str(error)],
            metadata={'error_type': type(error).__name__}
        )
    
    def log_analysis_start(self, analysis_type: str, context: AnalysisContext) -> None:
        """
        Log the start of an analysis.
        
        Args:
            analysis_type: Type of analysis starting
            context: Analysis context
        """
        self.logger.info(f"Starting {analysis_type} analysis", 
                        extra={'context': context.dict()})
    
    def log_analysis_complete(self, analysis_type: str, result: AnalysisResult) -> None:
        """
        Log the completion of an analysis.
        
        Args:
            analysis_type: Type of analysis completed
            result: Analysis result
        """
        self.logger.info(f"Completed {analysis_type} analysis", 
                        extra={
                            'success': result.success,
                            'metrics_count': len(result.metrics),
                            'warnings_count': len(result.warnings),
                            'errors_count': len(result.errors)
                        })
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert agent to dictionary representation.
        
        Returns:
            Dictionary containing agent information
        """
        return {
            'name': self.name,
            'capabilities': self.get_capabilities(),
            'cache_size': len(self._cache)
        }
    
    def __str__(self) -> str:
        """String representation of agent."""
        return f"{self.name}(capabilities={len(self.get_capabilities())})"
    
    def __repr__(self) -> str:
        """Detailed string representation of agent."""
        return f"{self.name}(db_connected={bool(self.db)}, cache_entries={len(self._cache)})"


class AgentRegistry:
    """
    Registry for managing and discovering available agents.
    """
    
    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
    
    def register(self, agent: BaseAgent) -> None:
        """
        Register an agent in the registry.
        
        Args:
            agent: Agent instance to register
        """
        self._agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """
        Get agent by name.
        
        Args:
            name: Agent name
            
        Returns:
            Agent instance or None if not found
        """
        return self._agents.get(name)
    
    def list_agents(self) -> List[str]:
        """
        Get list of registered agent names.
        
        Returns:
            List of agent names
        """
        return list(self._agents.keys())
    
    def get_all_capabilities(self) -> Dict[str, List[str]]:
        """
        Get capabilities for all registered agents.
        
        Returns:
            Dictionary mapping agent names to their capabilities
        """
        return {
            name: agent.get_capabilities() 
            for name, agent in self._agents.items()
        }
