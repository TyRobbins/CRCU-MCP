"""
Test Suite for Credit Union MCP Server

Comprehensive testing infrastructure for all components including
unit tests, integration tests, and performance benchmarks.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration
TEST_CONFIG = {
    'database': {
        'use_mock_data': True,
        'mock_data_size': 1000,
        'test_databases': ['TEST_ARCUSYM000', 'TEST_TEMENOS']
    },
    'agents': {
        'test_all_agents': True,
        'enable_performance_tests': True,
        'mock_external_calls': True
    },
    'performance': {
        'benchmark_iterations': 10,
        'load_test_duration': 60,
        'max_response_time_ms': 5000
    }
}
