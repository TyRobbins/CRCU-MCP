"""
Configuration Management for Credit Union MCP Server

Provides centralized configuration management with environment support,
secure credential handling, and dynamic configuration updates.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import threading
from loguru import logger
import base64
from cryptography.fernet import Fernet
from datetime import datetime


class Environment(Enum):
    """Supported environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    server: str
    port: int = 1433
    database: str = ""
    windows_auth: bool = True
    username: Optional[str] = None
    password: Optional[str] = None
    connection_timeout: int = 30
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600


@dataclass
class AgentConfig:
    """Agent-specific configuration settings."""
    cache_ttl: int = 1800
    max_retries: int = 3
    timeout: int = 60
    batch_size: int = 1000
    thresholds: Dict[str, float] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class PerformanceConfig:
    """Performance and optimization settings."""
    enable_caching: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600
    enable_performance_tracking: bool = True
    enable_circuit_breakers: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    memory_threshold_mb: float = 500.0
    enable_connection_pooling: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    rotation: str = "500 MB"
    retention: str = "10 days"
    enable_console: bool = False
    log_directory: str = "logs"


@dataclass
class SecurityConfig:
    """Security and compliance settings."""
    encrypt_credentials: bool = True
    encryption_key_file: str = ".encryption_key"
    max_query_rows: int = 10000
    allowed_schemas: List[str] = field(default_factory=lambda: ["dbo"])
    enable_query_validation: bool = True
    session_timeout: int = 3600


@dataclass
class MCPConfig:
    """MCP server configuration."""
    server_name: str = "credit-union-analytics"
    max_concurrent_tools: int = 10
    tool_timeout: int = 300
    enable_resource_caching: bool = True
    enable_prompt_templates: bool = False


class ConfigManager:
    """
    Centralized configuration management with environment support and security.
    """
    
    def __init__(self, config_dir: Optional[Path] = None, 
                 environment: Optional[Environment] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Configuration directory path
            environment: Target environment
        """
        self.config_dir = config_dir or Path("config")
        self.environment = environment or self._detect_environment()
        self._config_cache: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._file_watchers: Dict[str, float] = {}
        self._encryption_key: Optional[bytes] = None
        
        # Load configurations
        self._load_all_configs()
        
    def _detect_environment(self) -> Environment:
        """
        Detect current environment from environment variables.
        
        Returns:
            Detected environment
        """
        env_name = os.getenv("CREDIT_UNION_ENV", "development").lower()
        try:
            return Environment(env_name)
        except ValueError:
            logger.warning(f"Unknown environment '{env_name}', defaulting to development")
            return Environment.DEVELOPMENT
    
    def _load_all_configs(self) -> None:
        """Load all configuration files."""
        try:
            # Load base configurations
            self._load_database_config()
            self._load_agent_configs()
            self._load_performance_config()
            self._load_logging_config()
            self._load_security_config()
            self._load_mcp_config()
            
            # Load environment-specific overrides
            self._load_environment_overrides()
            
            logger.info(f"Configuration loaded for environment: {self.environment.value}")
            
        except Exception as e:
            logger.error(f"Failed to load configurations: {e}")
            raise
    
    def _load_database_config(self) -> None:
        """Load database configuration."""
        config_file = self.config_dir / "database_config.yaml"
        
        if not config_file.exists():
            logger.warning(f"Database config file not found: {config_file}")
            return
            
        try:
            with open(config_file, 'r') as f:
                raw_config = yaml.safe_load(f)
            
            databases = {}
            for db_name, db_config in raw_config.items():
                if isinstance(db_config, dict):
                    # Decrypt password if encrypted
                    if db_config.get('password') and self._is_encrypted(db_config['password']):
                        db_config['password'] = self._decrypt_value(db_config['password'])
                    
                    databases[db_name] = DatabaseConfig(**db_config)
            
            self._config_cache['databases'] = databases
            self._file_watchers[str(config_file)] = config_file.stat().st_mtime
            
        except Exception as e:
            logger.error(f"Failed to load database config: {e}")
            raise
    
    def _load_agent_configs(self) -> None:
        """Load agent-specific configurations."""
        config_file = self.config_dir / "agent_config.yaml"
        
        if not config_file.exists():
            # Create default agent config
            self._create_default_agent_config(config_file)
        
        try:
            with open(config_file, 'r') as f:
                raw_config = yaml.safe_load(f)
            
            agents = {}
            for agent_name, agent_config in raw_config.items():
                if isinstance(agent_config, dict):
                    agents[agent_name] = AgentConfig(**agent_config)
            
            self._config_cache['agents'] = agents
            self._file_watchers[str(config_file)] = config_file.stat().st_mtime
            
        except Exception as e:
            logger.error(f"Failed to load agent config: {e}")
            # Use defaults
            self._config_cache['agents'] = self._get_default_agent_configs()
    
    def _load_performance_config(self) -> None:
        """Load performance configuration."""
        config_file = self.config_dir / "performance_config.yaml"
        
        if not config_file.exists():
            self._create_default_performance_config(config_file)
        
        try:
            with open(config_file, 'r') as f:
                raw_config = yaml.safe_load(f)
            
            self._config_cache['performance'] = PerformanceConfig(**raw_config)
            self._file_watchers[str(config_file)] = config_file.stat().st_mtime
            
        except Exception as e:
            logger.error(f"Failed to load performance config: {e}")
            self._config_cache['performance'] = PerformanceConfig()
    
    def _load_logging_config(self) -> None:
        """Load logging configuration."""
        config_file = self.config_dir / "logging_config.yaml"
        
        if not config_file.exists():
            self._create_default_logging_config(config_file)
        
        try:
            with open(config_file, 'r') as f:
                raw_config = yaml.safe_load(f)
            
            self._config_cache['logging'] = LoggingConfig(**raw_config)
            self._file_watchers[str(config_file)] = config_file.stat().st_mtime
            
        except Exception as e:
            logger.error(f"Failed to load logging config: {e}")
            self._config_cache['logging'] = LoggingConfig()
    
    def _load_security_config(self) -> None:
        """Load security configuration."""
        config_file = self.config_dir / "security_config.yaml"
        
        if not config_file.exists():
            self._create_default_security_config(config_file)
        
        try:
            with open(config_file, 'r') as f:
                raw_config = yaml.safe_load(f)
            
            self._config_cache['security'] = SecurityConfig(**raw_config)
            self._file_watchers[str(config_file)] = config_file.stat().st_mtime
            
        except Exception as e:
            logger.error(f"Failed to load security config: {e}")
            self._config_cache['security'] = SecurityConfig()
    
    def _load_mcp_config(self) -> None:
        """Load MCP server configuration."""
        config_file = self.config_dir / "mcp_config.yaml"
        
        if not config_file.exists():
            self._create_default_mcp_config(config_file)
        
        try:
            with open(config_file, 'r') as f:
                raw_config = yaml.safe_load(f)
            
            self._config_cache['mcp'] = MCPConfig(**raw_config)
            self._file_watchers[str(config_file)] = config_file.stat().st_mtime
            
        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")
            self._config_cache['mcp'] = MCPConfig()
    
    def _load_environment_overrides(self) -> None:
        """Load environment-specific configuration overrides."""
        override_file = self.config_dir / f"{self.environment.value}.yaml"
        
        if not override_file.exists():
            return
        
        try:
            with open(override_file, 'r') as f:
                overrides = yaml.safe_load(f)
            
            # Apply overrides to cached configurations
            self._apply_overrides(overrides)
            self._file_watchers[str(override_file)] = override_file.stat().st_mtime
            
            logger.info(f"Applied environment overrides from {override_file}")
            
        except Exception as e:
            logger.error(f"Failed to load environment overrides: {e}")
    
    def _apply_overrides(self, overrides: Dict[str, Any]) -> None:
        """
        Apply configuration overrides.
        
        Args:
            overrides: Override values to apply
        """
        with self._lock:
            for section, values in overrides.items():
                if section in self._config_cache and isinstance(values, dict):
                    if hasattr(self._config_cache[section], '__dict__'):
                        # Update dataclass fields
                        for key, value in values.items():
                            if hasattr(self._config_cache[section], key):
                                setattr(self._config_cache[section], key, value)
                    elif isinstance(self._config_cache[section], dict):
                        # Update dictionary values
                        self._config_cache[section].update(values)
    
    def get_database_config(self, database_name: str) -> Optional[DatabaseConfig]:
        """
        Get database configuration.
        
        Args:
            database_name: Name of the database
            
        Returns:
            Database configuration or None if not found
        """
        databases = self._config_cache.get('databases', {})
        return databases.get(database_name)
    
    def get_agent_config(self, agent_name: str) -> AgentConfig:
        """
        Get agent configuration.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Agent configuration
        """
        agents = self._config_cache.get('agents', {})
        return agents.get(agent_name, AgentConfig())
    
    def get_performance_config(self) -> PerformanceConfig:
        """Get performance configuration."""
        return self._config_cache.get('performance', PerformanceConfig())
    
    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration."""
        return self._config_cache.get('logging', LoggingConfig())
    
    def get_security_config(self) -> SecurityConfig:
        """Get security configuration."""
        return self._config_cache.get('security', SecurityConfig())
    
    def get_mcp_config(self) -> MCPConfig:
        """Get MCP configuration."""
        return self._config_cache.get('mcp', MCPConfig())
    
    def set_config(self, section: str, key: str, value: Any) -> None:
        """
        Set configuration value at runtime.
        
        Args:
            section: Configuration section
            key: Configuration key
            value: New value
        """
        with self._lock:
            if section in self._config_cache:
                config_obj = self._config_cache[section]
                if hasattr(config_obj, key):
                    setattr(config_obj, key, value)
                    logger.info(f"Updated config {section}.{key} = {value}")
                else:
                    logger.warning(f"Configuration key {section}.{key} does not exist")
            else:
                logger.warning(f"Configuration section {section} does not exist")
    
    def reload_configs(self) -> None:
        """Reload all configurations from files."""
        logger.info("Reloading configurations...")
        self._config_cache.clear()
        self._file_watchers.clear()
        self._load_all_configs()
    
    def check_for_updates(self) -> bool:
        """
        Check if any configuration files have been updated.
        
        Returns:
            True if any files were updated and reloaded
        """
        updated = False
        
        for file_path, last_mtime in self._file_watchers.items():
            try:
                current_mtime = Path(file_path).stat().st_mtime
                if current_mtime > last_mtime:
                    logger.info(f"Configuration file updated: {file_path}")
                    updated = True
            except (OSError, FileNotFoundError):
                logger.warning(f"Configuration file no longer exists: {file_path}")
        
        if updated:
            self.reload_configs()
            
        return updated
    
    def get_all_configs(self) -> Dict[str, Any]:
        """
        Get all current configurations.
        
        Returns:
            Dictionary with all configurations
        """
        with self._lock:
            return {
                'environment': self.environment.value,
                'databases': {name: self._config_to_dict(config) 
                             for name, config in self._config_cache.get('databases', {}).items()},
                'agents': {name: self._config_to_dict(config)
                          for name, config in self._config_cache.get('agents', {}).items()},
                'performance': self._config_to_dict(self._config_cache.get('performance')),
                'logging': self._config_to_dict(self._config_cache.get('logging')),
                'security': self._config_to_dict(self._config_cache.get('security')),
                'mcp': self._config_to_dict(self._config_cache.get('mcp'))
            }
    
    def _config_to_dict(self, config_obj: Any) -> Dict[str, Any]:
        """Convert configuration object to dictionary."""
        if hasattr(config_obj, '__dict__'):
            return {k: v for k, v in config_obj.__dict__.items() 
                   if not k.startswith('_')}
        return config_obj or {}
    
    def encrypt_value(self, value: str) -> str:
        """
        Encrypt a configuration value.
        
        Args:
            value: Value to encrypt
            
        Returns:
            Encrypted value
        """
        if not self._encryption_key:
            self._load_or_create_encryption_key()
        
        fernet = Fernet(self._encryption_key)
        encrypted = fernet.encrypt(value.encode())
        return base64.b64encode(encrypted).decode()
    
    def decrypt_value(self, encrypted_value: str) -> str:
        """
        Decrypt a configuration value.
        
        Args:
            encrypted_value: Encrypted value
            
        Returns:
            Decrypted value
        """
        return self._decrypt_value(encrypted_value)
    
    def _decrypt_value(self, encrypted_value: str) -> str:
        """Internal decrypt method."""
        if not self._encryption_key:
            self._load_or_create_encryption_key()
        
        try:
            fernet = Fernet(self._encryption_key)
            decoded = base64.b64decode(encrypted_value.encode())
            return fernet.decrypt(decoded).decode()
        except Exception as e:
            logger.error(f"Failed to decrypt value: {e}")
            return encrypted_value
    
    def _is_encrypted(self, value: str) -> bool:
        """Check if a value appears to be encrypted."""
        try:
            # Simple heuristic: encrypted values are base64 encoded
            base64.b64decode(value)
            return len(value) > 20 and '=' in value
        except:
            return False
    
    def _load_or_create_encryption_key(self) -> None:
        """Load or create encryption key."""
        key_file = Path(self._config_cache.get('security', SecurityConfig()).encryption_key_file)
        
        if key_file.exists():
            self._encryption_key = key_file.read_bytes()
        else:
            self._encryption_key = Fernet.generate_key()
            key_file.write_bytes(self._encryption_key)
            logger.info(f"Created new encryption key: {key_file}")
    
    def _create_default_agent_config(self, config_file: Path) -> None:
        """Create default agent configuration file."""
        default_config = self._get_default_agent_configs()
        
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            # Convert dataclasses to dict for YAML serialization
            yaml_config = {}
            for name, config in default_config.items():
                yaml_config[name] = self._config_to_dict(config)
            yaml.dump(yaml_config, f, default_flow_style=False)
    
    def _get_default_agent_configs(self) -> Dict[str, AgentConfig]:
        """Get default agent configurations."""
        return {
            'financial_performance': AgentConfig(
                thresholds={
                    'roa_excellent': 1.0,
                    'roa_good': 0.75,
                    'roe_excellent': 10.0,
                    'roe_good': 8.0,
                    'nim_excellent': 3.5,
                    'nim_good': 3.0
                }
            ),
            'portfolio_risk': AgentConfig(
                thresholds={
                    'concentration_limit': 25.0,
                    'delinquency_warning': 2.0,
                    'delinquency_critical': 5.0,
                    'hhi_warning': 0.15,
                    'hhi_critical': 0.25
                }
            ),
            'member_analytics': AgentConfig(
                thresholds={
                    'high_value_threshold': 10000.0,
                    'churn_risk_threshold': 0.7,
                    'engagement_low': 0.3,
                    'ltv_high': 50000.0
                }
            ),
            'compliance': AgentConfig(
                thresholds={
                    'capital_adequacy_min': 7.0,
                    'leverage_ratio_min': 5.0,
                    'liquidity_ratio_min': 10.0,
                    'prompt_corrective_action': 6.0
                }
            ),
            'operations': AgentConfig(
                thresholds={
                    'efficiency_excellent': 85.0,
                    'efficiency_good': 75.0,
                    'cost_per_member_target': 100.0,
                    'digital_adoption_target': 70.0
                }
            )
        }
    
    def _create_default_performance_config(self, config_file: Path) -> None:
        """Create default performance configuration file."""
        default_config = PerformanceConfig()
        
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            yaml.dump(self._config_to_dict(default_config), f, default_flow_style=False)
    
    def _create_default_logging_config(self, config_file: Path) -> None:
        """Create default logging configuration file."""
        default_config = LoggingConfig()
        if self.environment == Environment.DEVELOPMENT:
            default_config.enable_console = True
            default_config.level = "DEBUG"
        
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            yaml.dump(self._config_to_dict(default_config), f, default_flow_style=False)
    
    def _create_default_security_config(self, config_file: Path) -> None:
        """Create default security configuration file."""
        default_config = SecurityConfig()
        
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            yaml.dump(self._config_to_dict(default_config), f, default_flow_style=False)
    
    def _create_default_mcp_config(self, config_file: Path) -> None:
        """Create default MCP configuration file."""
        default_config = MCPConfig()
        
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            yaml.dump(self._config_to_dict(default_config), f, default_flow_style=False)


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_dir: Optional[Path] = None,
                      environment: Optional[Environment] = None) -> ConfigManager:
    """
    Get or create global configuration manager instance.
    
    Args:
        config_dir: Configuration directory path
        environment: Target environment
        
    Returns:
        ConfigManager instance
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager(config_dir, environment)
    
    return _config_manager


def reset_config_manager() -> None:
    """Reset global configuration manager (mainly for testing)."""
    global _config_manager
    _config_manager = None
