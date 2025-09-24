"""
Performance Utilities for Credit Union MCP Server

Provides performance tracking, caching decorators, and optimization utilities
to improve system responsiveness and monitoring capabilities.
"""

import time
import functools
import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import weakref
import gc
from loguru import logger
import asyncio
from contextlib import contextmanager


class PerformanceTracker:
    """
    Tracks performance metrics for operations and provides statistics.
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize performance tracker.
        
        Args:
            max_history: Maximum number of operations to keep in history
        """
        self.max_history = max_history
        self.operation_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.last_reset = datetime.now()
        self._lock = threading.RLock()
        
    @contextmanager
    def track_operation(self, operation_name: str):
        """
        Context manager to track operation performance.
        
        Args:
            operation_name: Name of the operation being tracked
        """
        start_time = time.time()
        error_occurred = False
        
        try:
            yield
        except Exception as e:
            error_occurred = True
            with self._lock:
                self.error_counts[operation_name] += 1
            raise
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            with self._lock:
                self.operation_times[operation_name].append(duration)
                self.operation_counts[operation_name] += 1
                
            if error_occurred:
                logger.warning(f"Operation {operation_name} failed after {duration:.3f}s")
            else:
                logger.debug(f"Operation {operation_name} completed in {duration:.3f}s")
    
    def get_stats(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Args:
            operation_name: Specific operation to get stats for, or None for all
            
        Returns:
            Dictionary with performance statistics
        """
        with self._lock:
            if operation_name:
                if operation_name not in self.operation_times:
                    return {"error": f"No data for operation '{operation_name}'"}
                    
                times = list(self.operation_times[operation_name])
                return self._calculate_stats(operation_name, times)
            else:
                stats = {}
                for op_name, times in self.operation_times.items():
                    stats[op_name] = self._calculate_stats(op_name, list(times))
                return stats
    
    def _calculate_stats(self, operation_name: str, times: List[float]) -> Dict[str, Any]:
        """
        Calculate statistics for a list of operation times.
        
        Args:
            operation_name: Name of the operation
            times: List of operation times in seconds
            
        Returns:
            Dictionary with calculated statistics
        """
        if not times:
            return {
                "count": 0,
                "error_count": self.error_counts.get(operation_name, 0),
                "error_rate": 0.0
            }
        
        times.sort()
        count = len(times)
        total_count = self.operation_counts[operation_name]
        error_count = self.error_counts[operation_name]
        
        return {
            "count": count,
            "total_count": total_count,
            "error_count": error_count,
            "error_rate": error_count / total_count if total_count > 0 else 0.0,
            "min_time": min(times),
            "max_time": max(times),
            "avg_time": sum(times) / count,
            "median_time": times[count // 2] if count % 2 == 1 else (times[count // 2 - 1] + times[count // 2]) / 2,
            "p95_time": times[int(count * 0.95)] if count > 0 else 0,
            "p99_time": times[int(count * 0.99)] if count > 0 else 0,
            "total_time": sum(times)
        }
    
    def get_slow_operations(self, threshold: float = 1.0) -> Dict[str, Dict[str, Any]]:
        """
        Get operations that are slower than the threshold.
        
        Args:
            threshold: Time threshold in seconds
            
        Returns:
            Dictionary of slow operations and their stats
        """
        slow_ops = {}
        
        for op_name in self.operation_times:
            stats = self.get_stats(op_name)
            if stats.get("avg_time", 0) > threshold:
                slow_ops[op_name] = stats
                
        return slow_ops
    
    def reset_stats(self, operation_name: Optional[str] = None) -> None:
        """
        Reset statistics for specific operation or all operations.
        
        Args:
            operation_name: Operation to reset, or None for all
        """
        with self._lock:
            if operation_name:
                if operation_name in self.operation_times:
                    self.operation_times[operation_name].clear()
                    self.operation_counts[operation_name] = 0
                    self.error_counts[operation_name] = 0
            else:
                self.operation_times.clear()
                self.operation_counts.clear()
                self.error_counts.clear()
                self.last_reset = datetime.now()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get overall performance summary.
        
        Returns:
            Summary of all tracked operations
        """
        with self._lock:
            total_operations = sum(self.operation_counts.values())
            total_errors = sum(self.error_counts.values())
            
            if not self.operation_times:
                return {
                    "total_operations": 0,
                    "total_errors": 0,
                    "error_rate": 0.0,
                    "tracked_operations": 0,
                    "tracking_since": self.last_reset.isoformat()
                }
            
            all_times = []
            for times in self.operation_times.values():
                all_times.extend(times)
            
            all_times.sort()
            
            return {
                "total_operations": total_operations,
                "total_errors": total_errors,
                "error_rate": total_errors / total_operations if total_operations > 0 else 0.0,
                "tracked_operations": len(self.operation_times),
                "tracking_since": self.last_reset.isoformat(),
                "overall_stats": {
                    "total_time": sum(all_times),
                    "avg_time": sum(all_times) / len(all_times) if all_times else 0,
                    "median_time": all_times[len(all_times) // 2] if all_times else 0,
                    "p95_time": all_times[int(len(all_times) * 0.95)] if all_times else 0
                } if all_times else {}
            }


class CacheManager:
    """
    Thread-safe cache manager with TTL support and memory management.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize cache manager.
        
        Args:
            max_size: Maximum number of cache entries
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None
            
            entry = self._cache[key]
            current_time = time.time()
            
            # Check if entry has expired
            if current_time > entry['expires_at']:
                del self._cache[key]
                del self._access_times[key]
                self.misses += 1
                return None
            
            # Update access time and return value
            self._access_times[key] = current_time
            self.hits += 1
            return entry['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds, or None for default
        """
        if ttl is None:
            ttl = self.default_ttl
            
        current_time = time.time()
        expires_at = current_time + ttl
        
        with self._lock:
            # Check if we need to evict entries
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()
            
            self._cache[key] = {
                'value': value,
                'expires_at': expires_at,
                'created_at': current_time
            }
            self._access_times[key] = current_time
    
    def delete(self, key: str) -> bool:
        """
        Delete entry from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key was found and deleted
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._access_times[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self.hits = 0
            self.misses = 0
            self.evictions = 0
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries from cache.
        
        Returns:
            Number of entries removed
        """
        current_time = time.time()
        expired_keys = []
        
        with self._lock:
            for key, entry in self._cache.items():
                if current_time > entry['expires_at']:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
                del self._access_times[key]
        
        return len(expired_keys)
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._access_times:
            return
            
        # Find least recently used key
        lru_key = min(self._access_times.items(), key=lambda x: x[1])[0]
        
        del self._cache[lru_key]
        del self._access_times[lru_key]
        self.evictions += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "evictions": self.evictions,
                "default_ttl": self.default_ttl
            }


# Global cache instance
_global_cache = CacheManager()


def cache_with_ttl(ttl: int = 3600, cache_manager: Optional[CacheManager] = None):
    """
    Decorator to cache function results with TTL.
    
    Args:
        ttl: Cache TTL in seconds
        cache_manager: Optional cache manager instance
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        cache = cache_manager or _global_cache
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = "|".join(key_parts)
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            logger.debug(f"Cache miss for {func.__name__}, executing function")
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            
            return result
        
        # Add cache management methods to wrapper
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_info = lambda: cache.get_stats()
        
        return wrapper
    
    return decorator


def async_cache_with_ttl(ttl: int = 3600, cache_manager: Optional[CacheManager] = None):
    """
    Decorator to cache async function results with TTL.
    
    Args:
        ttl: Cache TTL in seconds
        cache_manager: Optional cache manager instance
        
    Returns:
        Decorated async function
    """
    def decorator(func: Callable) -> Callable:
        cache = cache_manager or _global_cache
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = "|".join(key_parts)
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for async {func.__name__}")
                return cached_result
            
            # Execute async function and cache result
            logger.debug(f"Cache miss for async {func.__name__}, executing function")
            result = await func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            
            return result
        
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_info = lambda: cache.get_stats()
        
        return wrapper
    
    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for resilient error handling.
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, 
                 expected_exception: type = Exception):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time in seconds before attempting to close circuit
            expected_exception: Exception type to trigger circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        with self._lock:
            if self.state == 'OPEN':
                if self._should_attempt_reset():
                    self.state = 'HALF_OPEN'
                else:
                    raise Exception(f"Circuit breaker is OPEN. Last failure: {self.last_failure_time}")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self) -> None:
        """Handle successful function execution."""
        with self._lock:
            self.failure_count = 0
            self.state = 'CLOSED'
    
    def _on_failure(self) -> None:
        """Handle function execution failure."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current circuit breaker state.
        
        Returns:
            Dictionary with circuit breaker state information
        """
        with self._lock:
            return {
                'state': self.state,
                'failure_count': self.failure_count,
                'failure_threshold': self.failure_threshold,
                'last_failure_time': self.last_failure_time,
                'recovery_timeout': self.recovery_timeout
            }
    
    def reset(self) -> None:
        """Manually reset circuit breaker."""
        with self._lock:
            self.failure_count = 0
            self.last_failure_time = None
            self.state = 'CLOSED'


def circuit_breaker(failure_threshold: int = 5, recovery_timeout: int = 60,
                   expected_exception: type = Exception):
    """
    Decorator to add circuit breaker pattern to functions.
    
    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Time in seconds before attempting to close circuit
        expected_exception: Exception type to trigger circuit breaker
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        breaker = CircuitBreaker(failure_threshold, recovery_timeout, expected_exception)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        
        wrapper.circuit_breaker = breaker
        wrapper.get_circuit_state = breaker.get_state
        wrapper.reset_circuit = breaker.reset
        
        return wrapper
    
    return decorator


class ConnectionPool:
    """
    Generic connection pool with health checking and automatic recovery.
    """
    
    def __init__(self, create_connection: Callable, max_connections: int = 10,
                 health_check: Optional[Callable] = None, health_check_interval: int = 300):
        """
        Initialize connection pool.
        
        Args:
            create_connection: Function to create new connections
            max_connections: Maximum number of connections in pool
            health_check: Function to check connection health
            health_check_interval: Health check interval in seconds
        """
        self.create_connection = create_connection
        self.max_connections = max_connections
        self.health_check = health_check
        self.health_check_interval = health_check_interval
        
        self.pool: deque = deque()
        self.active_connections: int = 0
        self.last_health_check = 0
        self._lock = threading.RLock()
    
    @contextmanager
    def get_connection(self):
        """
        Get connection from pool with automatic return.
        
        Yields:
            Database connection
        """
        conn = None
        try:
            conn = self._get_connection()
            yield conn
        finally:
            if conn:
                self._return_connection(conn)
    
    def _get_connection(self):
        """Get connection from pool or create new one."""
        with self._lock:
            # Perform periodic health checks
            current_time = time.time()
            if (current_time - self.last_health_check) > self.health_check_interval:
                self._perform_health_check()
                self.last_health_check = current_time
            
            # Try to get existing connection
            while self.pool:
                conn = self.pool.popleft()
                if self._is_connection_healthy(conn):
                    return conn
                else:
                    self.active_connections -= 1
            
            # Create new connection if under limit
            if self.active_connections < self.max_connections:
                conn = self.create_connection()
                self.active_connections += 1
                return conn
            
            raise Exception("Connection pool exhausted")
    
    def _return_connection(self, connection):
        """Return connection to pool."""
        with self._lock:
            if self._is_connection_healthy(connection):
                self.pool.append(connection)
            else:
                self.active_connections -= 1
    
    def _is_connection_healthy(self, connection) -> bool:
        """Check if connection is healthy."""
        if not self.health_check:
            return True
        
        try:
            return self.health_check(connection)
        except Exception as e:
            logger.warning(f"Connection health check failed: {e}")
            return False
    
    def _perform_health_check(self) -> None:
        """Perform health check on all pooled connections."""
        if not self.health_check:
            return
        
        healthy_connections = deque()
        removed_count = 0
        
        while self.pool:
            conn = self.pool.popleft()
            if self._is_connection_healthy(conn):
                healthy_connections.append(conn)
            else:
                removed_count += 1
                self.active_connections -= 1
        
        self.pool = healthy_connections
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} unhealthy connections from pool")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get connection pool statistics.
        
        Returns:
            Dictionary with pool statistics
        """
        with self._lock:
            return {
                'pool_size': len(self.pool),
                'active_connections': self.active_connections,
                'max_connections': self.max_connections,
                'utilization': self.active_connections / self.max_connections if self.max_connections > 0 else 0
            }


class MemoryMonitor:
    """
    Monitor memory usage and trigger garbage collection when needed.
    """
    
    def __init__(self, threshold_mb: float = 500.0):
        """
        Initialize memory monitor.
        
        Args:
            threshold_mb: Memory threshold in MB to trigger GC
        """
        self.threshold_mb = threshold_mb
        self.last_gc_time = time.time()
        
    def check_memory(self, force_gc: bool = False) -> Dict[str, Any]:
        """
        Check current memory usage and perform GC if needed.
        
        Args:
            force_gc: Force garbage collection regardless of threshold
            
        Returns:
            Dictionary with memory statistics
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            stats = {
                'memory_mb': memory_mb,
                'threshold_mb': self.threshold_mb,
                'gc_triggered': False,
                'time_since_last_gc': time.time() - self.last_gc_time
            }
            
            if force_gc or memory_mb > self.threshold_mb:
                collected = gc.collect()
                self.last_gc_time = time.time()
                stats['gc_triggered'] = True
                stats['objects_collected'] = collected
                
                # Get memory info after GC
                memory_info_after = process.memory_info()
                memory_mb_after = memory_info_after.rss / 1024 / 1024
                stats['memory_mb_after_gc'] = memory_mb_after
                stats['memory_freed_mb'] = memory_mb - memory_mb_after
                
                logger.info(f"GC triggered: freed {stats['memory_freed_mb']:.1f}MB, collected {collected} objects")
            
            return stats
            
        except ImportError:
            logger.warning("psutil not available for memory monitoring")
            return {'error': 'psutil not available'}
        except Exception as e:
            logger.error(f"Memory monitoring failed: {e}")
            return {'error': str(e)}


# Global instances
performance_tracker = PerformanceTracker()
memory_monitor = MemoryMonitor()


def get_global_cache() -> CacheManager:
    """Get the global cache manager instance."""
    return _global_cache


def get_performance_tracker() -> PerformanceTracker:
    """Get the global performance tracker instance."""
    return performance_tracker


def get_memory_monitor() -> MemoryMonitor:
    """Get the global memory monitor instance."""
    return memory_monitor
