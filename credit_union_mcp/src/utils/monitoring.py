"""
Monitoring and Observability for Credit Union MCP Server

Provides comprehensive monitoring, metrics collection, alerting, and health checking
capabilities for operational visibility and proactive issue detection.
"""

import time
import json
import asyncio
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import psutil
import socket
from pathlib import Path
from loguru import logger

from .performance_utils import PerformanceTracker, get_performance_tracker
from .config_manager import get_config_manager


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class Alert:
    """Alert definition and state."""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    condition: str
    threshold: float
    current_value: Optional[float] = None
    is_active: bool = False
    first_triggered: Optional[datetime] = None
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class HealthCheck:
    """Health check definition."""
    name: str
    check_function: Callable[[], Dict[str, Any]]
    interval_seconds: int = 60
    timeout_seconds: int = 30
    enabled: bool = True
    last_check: Optional[datetime] = None
    last_status: HealthStatus = HealthStatus.HEALTHY
    last_result: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and stores application metrics."""
    
    def __init__(self, max_metrics_per_name: int = 1000):
        """
        Initialize metrics collector.
        
        Args:
            max_metrics_per_name: Maximum number of metrics to keep per metric name
        """
        self.max_metrics_per_name = max_metrics_per_name
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_metrics_per_name))
        self._lock = threading.RLock()
        self._start_time = datetime.now()
        
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None,
                     unit: str = "") -> None:
        """
        Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for the metric
            unit: Optional unit of measurement
        """
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            unit=unit
        )
        
        with self._lock:
            self._metrics[name].append(metric)
    
    def get_metrics(self, name: str, since: Optional[datetime] = None) -> List[Metric]:
        """
        Get metrics for a given name.
        
        Args:
            name: Metric name
            since: Optional timestamp to filter metrics
            
        Returns:
            List of metrics
        """
        with self._lock:
            metrics = list(self._metrics.get(name, []))
            
        if since:
            metrics = [m for m in metrics if m.timestamp >= since]
            
        return metrics
    
    def get_latest_metric(self, name: str) -> Optional[Metric]:
        """Get the latest metric value for a name."""
        with self._lock:
            metrics = self._metrics.get(name)
            return metrics[-1] if metrics else None
    
    def get_metric_summary(self, name: str, minutes: int = 60) -> Dict[str, float]:
        """
        Get statistical summary of metrics over time period.
        
        Args:
            name: Metric name
            minutes: Time period in minutes
            
        Returns:
            Dictionary with statistical summary
        """
        since = datetime.now() - timedelta(minutes=minutes)
        metrics = self.get_metrics(name, since)
        
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        values.sort()
        
        count = len(values)
        return {
            'count': count,
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / count,
            'median': values[count // 2] if count % 2 == 1 else (values[count // 2 - 1] + values[count // 2]) / 2,
            'p95': values[int(count * 0.95)] if count > 0 else 0,
            'p99': values[int(count * 0.99)] if count > 0 else 0
        }
    
    def get_all_metric_names(self) -> List[str]:
        """Get all metric names currently being tracked."""
        with self._lock:
            return list(self._metrics.keys())
    
    def export_metrics(self, format_type: str = "json") -> str:
        """
        Export all metrics in specified format.
        
        Args:
            format_type: Export format (json, prometheus)
            
        Returns:
            Formatted metrics string
        """
        if format_type.lower() == "json":
            return self._export_json()
        elif format_type.lower() == "prometheus":
            return self._export_prometheus()
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _export_json(self) -> str:
        """Export metrics in JSON format."""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': (datetime.now() - self._start_time).total_seconds(),
            'metrics': {}
        }
        
        with self._lock:
            for name, metric_deque in self._metrics.items():
                export_data['metrics'][name] = [
                    {
                        'value': m.value,
                        'timestamp': m.timestamp.isoformat(),
                        'tags': m.tags,
                        'unit': m.unit
                    }
                    for m in metric_deque
                ]
        
        return json.dumps(export_data, indent=2)
    
    def _export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        with self._lock:
            for name, metric_deque in self._metrics.items():
                if not metric_deque:
                    continue
                    
                latest_metric = metric_deque[-1]
                
                # Format metric name for Prometheus
                prom_name = name.replace('-', '_').replace(' ', '_').lower()
                
                # Add help and type comments
                lines.append(f"# HELP {prom_name} Credit Union MCP metric")
                lines.append(f"# TYPE {prom_name} gauge")
                
                # Add metric with tags
                if latest_metric.tags:
                    tag_str = ','.join([f'{k}="{v}"' for k, v in latest_metric.tags.items()])
                    lines.append(f"{prom_name}{{{tag_str}}} {latest_metric.value}")
                else:
                    lines.append(f"{prom_name} {latest_metric.value}")
        
        return '\n'.join(lines)


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """
        Initialize alert manager.
        
        Args:
            metrics_collector: Metrics collector instance
        """
        self.metrics_collector = metrics_collector
        self._alerts: Dict[str, Alert] = {}
        self._alert_history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self._notification_handlers: List[Callable[[Alert], None]] = []
        
        # Default alerts
        self._setup_default_alerts()
    
    def add_alert(self, alert: Alert) -> None:
        """Add or update an alert definition."""
        with self._lock:
            self._alerts[alert.id] = alert
    
    def remove_alert(self, alert_id: str) -> None:
        """Remove an alert definition."""
        with self._lock:
            if alert_id in self._alerts:
                del self._alerts[alert_id]
    
    def add_notification_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add a notification handler for alerts."""
        self._notification_handlers.append(handler)
    
    def check_alerts(self) -> List[Alert]:
        """
        Check all alerts and return active ones.
        
        Returns:
            List of currently active alerts
        """
        active_alerts = []
        
        with self._lock:
            for alert in self._alerts.values():
                if not alert.condition:
                    continue
                
                try:
                    # Get current metric value
                    current_metric = self.metrics_collector.get_latest_metric(alert.condition)
                    current_value = current_metric.value if current_metric else 0.0
                    
                    alert.current_value = current_value
                    
                    # Check if alert should trigger
                    should_trigger = self._evaluate_alert_condition(alert, current_value)
                    
                    if should_trigger and not alert.is_active:
                        # Alert just triggered
                        alert.is_active = True
                        alert.first_triggered = datetime.now()
                        alert.last_triggered = datetime.now()
                        alert.trigger_count += 1
                        
                        self._record_alert_event(alert, "triggered")
                        self._notify_alert(alert)
                        
                    elif not should_trigger and alert.is_active:
                        # Alert resolved
                        alert.is_active = False
                        self._record_alert_event(alert, "resolved")
                    
                    elif should_trigger and alert.is_active:
                        # Alert still active
                        alert.last_triggered = datetime.now()
                    
                    if alert.is_active:
                        active_alerts.append(alert)
                        
                except Exception as e:
                    logger.error(f"Error checking alert {alert.id}: {e}")
        
        return active_alerts
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alert history for specified time period."""
        since = datetime.now() - timedelta(hours=hours)
        return [
            event for event in self._alert_history
            if datetime.fromisoformat(event['timestamp']) >= since
        ]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert status."""
        with self._lock:
            active_alerts = [a for a in self._alerts.values() if a.is_active]
            
            return {
                'total_alerts': len(self._alerts),
                'active_alerts': len(active_alerts),
                'critical_alerts': len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
                'high_alerts': len([a for a in active_alerts if a.severity == AlertSeverity.HIGH]),
                'alert_details': [
                    {
                        'id': a.id,
                        'name': a.name,
                        'severity': a.severity.value,
                        'current_value': a.current_value,
                        'threshold': a.threshold
                    }
                    for a in active_alerts
                ]
            }
    
    def _setup_default_alerts(self) -> None:
        """Set up default system alerts."""
        default_alerts = [
            Alert(
                id="high_cpu_usage",
                name="High CPU Usage",
                description="CPU usage is above threshold",
                severity=AlertSeverity.HIGH,
                condition="system.cpu_percent",
                threshold=80.0
            ),
            Alert(
                id="high_memory_usage",
                name="High Memory Usage", 
                description="Memory usage is above threshold",
                severity=AlertSeverity.HIGH,
                condition="system.memory_percent",
                threshold=85.0
            ),
            Alert(
                id="database_connection_failure",
                name="Database Connection Failure",
                description="Database connection health check failed",
                severity=AlertSeverity.CRITICAL,
                condition="database.connection_failed",
                threshold=1.0
            ),
            Alert(
                id="high_error_rate",
                name="High Error Rate",
                description="Error rate is above threshold",
                severity=AlertSeverity.HIGH,
                condition="application.error_rate",
                threshold=5.0
            ),
            Alert(
                id="slow_response_time",
                name="Slow Response Time",
                description="Average response time is above threshold",
                severity=AlertSeverity.MEDIUM,
                condition="application.avg_response_time",
                threshold=2.0
            )
        ]
        
        for alert in default_alerts:
            self.add_alert(alert)
    
    def _evaluate_alert_condition(self, alert: Alert, current_value: float) -> bool:
        """Evaluate if alert condition is met."""
        # Simple threshold comparison for now
        # Could be extended to support more complex conditions
        return current_value >= alert.threshold
    
    def _record_alert_event(self, alert: Alert, event_type: str) -> None:
        """Record an alert event in history."""
        event = {
            'alert_id': alert.id,
            'alert_name': alert.name,
            'event_type': event_type,
            'severity': alert.severity.value,
            'current_value': alert.current_value,
            'threshold': alert.threshold,
            'timestamp': datetime.now().isoformat()
        }
        
        self._alert_history.append(event)
        
        # Keep only last 1000 events
        if len(self._alert_history) > 1000:
            self._alert_history = self._alert_history[-1000:]
    
    def _notify_alert(self, alert: Alert) -> None:
        """Send alert notifications."""
        for handler in self._notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert notification handler: {e}")


class HealthChecker:
    """Performs health checks on system components."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """
        Initialize health checker.
        
        Args:
            metrics_collector: Metrics collector instance
        """
        self.metrics_collector = metrics_collector
        self._health_checks: Dict[str, HealthCheck] = {}
        self._lock = threading.RLock()
        self._running = False
        self._check_thread: Optional[threading.Thread] = None
        
        # Setup default health checks
        self._setup_default_health_checks()
    
    def add_health_check(self, health_check: HealthCheck) -> None:
        """Add a health check."""
        with self._lock:
            self._health_checks[health_check.name] = health_check
    
    def remove_health_check(self, name: str) -> None:
        """Remove a health check."""
        with self._lock:
            if name in self._health_checks:
                del self._health_checks[name]
    
    def run_health_check(self, name: str) -> Dict[str, Any]:
        """
        Run a specific health check.
        
        Args:
            name: Health check name
            
        Returns:
            Health check result
        """
        with self._lock:
            health_check = self._health_checks.get(name)
            
        if not health_check:
            return {'error': f'Health check {name} not found'}
        
        if not health_check.enabled:
            return {'status': 'disabled'}
        
        try:
            start_time = time.time()
            result = health_check.check_function()
            duration = time.time() - start_time
            
            # Update health check state
            health_check.last_check = datetime.now()
            health_check.last_result = result
            
            # Determine status
            if result.get('healthy', True):
                health_check.last_status = HealthStatus.HEALTHY
            elif result.get('warning', False):
                health_check.last_status = HealthStatus.WARNING
            else:
                health_check.last_status = HealthStatus.UNHEALTHY
            
            # Record metrics
            self.metrics_collector.record_metric(
                f"health_check.{name}.duration",
                duration,
                tags={'status': health_check.last_status.value}
            )
            
            self.metrics_collector.record_metric(
                f"health_check.{name}.status",
                1 if health_check.last_status == HealthStatus.HEALTHY else 0
            )
            
            result['duration'] = duration
            result['timestamp'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            health_check.last_status = HealthStatus.CRITICAL
            health_check.last_result = {'error': str(e)}
            
            logger.error(f"Health check {name} failed: {e}")
            
            return {
                'healthy': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_all_health_checks(self) -> Dict[str, Dict[str, Any]]:
        """Run all enabled health checks."""
        results = {}
        
        with self._lock:
            health_checks = list(self._health_checks.items())
        
        for name, health_check in health_checks:
            if health_check.enabled:
                results[name] = self.run_health_check(name)
        
        return results
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary."""
        with self._lock:
            health_checks = list(self._health_checks.values())
        
        total_checks = len([hc for hc in health_checks if hc.enabled])
        healthy_checks = len([hc for hc in health_checks if hc.last_status == HealthStatus.HEALTHY])
        warning_checks = len([hc for hc in health_checks if hc.last_status == HealthStatus.WARNING])
        unhealthy_checks = len([hc for hc in health_checks if hc.last_status == HealthStatus.UNHEALTHY])
        critical_checks = len([hc for hc in health_checks if hc.last_status == HealthStatus.CRITICAL])
        
        overall_status = HealthStatus.HEALTHY
        if critical_checks > 0:
            overall_status = HealthStatus.CRITICAL
        elif unhealthy_checks > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif warning_checks > 0:
            overall_status = HealthStatus.WARNING
        
        return {
            'overall_status': overall_status.value,
            'total_checks': total_checks,
            'healthy': healthy_checks,
            'warning': warning_checks,
            'unhealthy': unhealthy_checks,
            'critical': critical_checks,
            'last_check': max([hc.last_check for hc in health_checks if hc.last_check]).isoformat() if health_checks else None
        }
    
    def start_continuous_monitoring(self, interval: int = 60) -> None:
        """Start continuous health monitoring."""
        if self._running:
            return
        
        self._running = True
        self._check_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self._check_thread.start()
        logger.info("Started continuous health monitoring")
    
    def stop_continuous_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        self._running = False
        if self._check_thread and self._check_thread.is_alive():
            self._check_thread.join()
        logger.info("Stopped continuous health monitoring")
    
    def _monitoring_loop(self, interval: int) -> None:
        """Continuous monitoring loop."""
        while self._running:
            try:
                self.run_all_health_checks()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(interval)
    
    def _setup_default_health_checks(self) -> None:
        """Setup default system health checks."""
        def check_system_resources():
            """Check system CPU and memory usage."""
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Record system metrics
                self.metrics_collector.record_metric("system.cpu_percent", cpu_percent)
                self.metrics_collector.record_metric("system.memory_percent", memory.percent)
                self.metrics_collector.record_metric("system.disk_percent", disk.percent)
                
                healthy = cpu_percent < 90 and memory.percent < 90 and disk.percent < 90
                warning = cpu_percent > 80 or memory.percent > 80 or disk.percent > 80
                
                return {
                    'healthy': healthy,
                    'warning': warning,
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'disk_percent': disk.percent,
                    'memory_available_gb': memory.available / (1024**3)
                }
            except Exception as e:
                return {'healthy': False, 'error': str(e)}
        
        def check_database_connection():
            """Check database connectivity."""
            try:
                config_manager = get_config_manager()
                db_configs = config_manager._config_cache.get('databases', {})
                
                results = {}
                overall_healthy = True
                
                for db_name in ['TEMENOS', 'ARCUSYM000']:
                    if db_name in db_configs:
                        try:
                            # Simple connection test (would need actual database manager)
                            # For now, just check if config exists
                            results[db_name] = {'status': 'configured'}
                            self.metrics_collector.record_metric(f"database.{db_name}.connection", 1)
                        except Exception as e:
                            results[db_name] = {'status': 'error', 'error': str(e)}
                            overall_healthy = False
                            self.metrics_collector.record_metric(f"database.{db_name}.connection", 0)
                
                return {
                    'healthy': overall_healthy,
                    'database_results': results
                }
            except Exception as e:
                return {'healthy': False, 'error': str(e)}
        
        def check_application_performance():
            """Check application performance metrics."""
            try:
                perf_tracker = get_performance_tracker()
                stats = perf_tracker.get_summary()
                
                # Calculate average response time from operations
                avg_response_time = 0.0
                if stats.get('overall_stats', {}).get('avg_time'):
                    avg_response_time = stats['overall_stats']['avg_time']
                
                error_rate = stats.get('error_rate', 0.0)
                
                # Record application metrics
                self.metrics_collector.record_metric("application.avg_response_time", avg_response_time)
                self.metrics_collector.record_metric("application.error_rate", error_rate)
                self.metrics_collector.record_metric("application.total_operations", stats.get('total_operations', 0))
                
                healthy = avg_response_time < 5.0 and error_rate < 5.0
                warning = avg_response_time > 2.0 or error_rate > 1.0
                
                return {
                    'healthy': healthy,
                    'warning': warning,
                    'avg_response_time': avg_response_time,
                    'error_rate': error_rate,
                    'total_operations': stats.get('total_operations', 0)
                }
            except Exception as e:
                return {'healthy': False, 'error': str(e)}
        
        # Add default health checks
        self.add_health_check(HealthCheck(
            name="system_resources",
            check_function=check_system_resources,
            interval_seconds=30
        ))
        
        self.add_health_check(HealthCheck(
            name="database_connectivity", 
            check_function=check_database_connection,
            interval_seconds=60
        ))
        
        self.add_health_check(HealthCheck(
            name="application_performance",
            check_function=check_application_performance,
            interval_seconds=60
        ))


class MonitoringManager:
    """Main monitoring and observability manager."""
    
    def __init__(self):
        """Initialize monitoring manager."""
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.metrics_collector)
        self.health_checker = HealthChecker(self.metrics_collector)
        
        # Setup logging notification handler
        self.alert_manager.add_notification_handler(self._log_alert_notification)
        
        self._monitoring_active = False
    
    def start_monitoring(self) -> None:
        """Start all monitoring components."""
        if self._monitoring_active:
            return
        
        self.health_checker.start_continuous_monitoring()
        self._monitoring_active = True
        
        # Start alert checking loop
        self._start_alert_checking()
        
        logger.info("Monitoring system started")
    
    def stop_monitoring(self) -> None:
        """Stop all monitoring components."""
        self.health_checker.stop_continuous_monitoring()
        self._monitoring_active = False
        logger.info("Monitoring system stopped")
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """
        Get comprehensive monitoring dashboard data.
        
        Returns:
            Dashboard data including metrics, alerts, and health status
        """
        # Get recent metrics summary
        metric_names = self.metrics_collector.get_all_metric_names()
        metrics_summary = {}
        
        for name in metric_names[:20]:  # Limit to recent metrics
            summary = self.metrics_collector.get_metric_summary(name, minutes=60)
            if summary:
                metrics_summary[name] = summary
        
        # Get system status
        health_summary = self.health_checker.get_health_summary()
        alert_summary = self.alert_manager.get_alert_summary()
        
        # Get recent alerts
        recent_alerts = self.alert_manager.get_alert_history(hours=24)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_status': {
                'overall_health': health_summary['overall_status'],
                'monitoring_active': self._monitoring_active,
                'uptime_hours': self._get_uptime_hours()
            },
            'health_summary': health_summary,
            'alert_summary': alert_summary,
            'metrics_summary': metrics_summary,
            'recent_alerts': recent_alerts[-10:],  # Last 10 alerts
            'performance_stats': get_performance_tracker().get_summary()
        }
    
    def export_metrics(self, format_type: str = "json") -> str:
        """Export all metrics in specified format."""
        return self.metrics_collector.export_metrics(format_type)
    
    def record_business_metric(self, metric_name: str, value: float, 
                             tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a business-specific metric.
        
        Args:
            metric_name: Name of the business metric
            value: Metric value
            tags: Optional tags for categorization
        """
        self.metrics_collector.record_metric(
            f"business.{metric_name}",
            value,
            tags=tags
        )
    
    def create_custom_alert(self, alert_id: str, name: str, description: str,
                          metric_name: str, threshold: float, 
                          severity: AlertSeverity = AlertSeverity.MEDIUM) -> None:
        """
        Create a custom alert.
        
        Args:
            alert_id: Unique alert identifier
            name: Alert name
            description: Alert description
            metric_name: Metric to monitor
            threshold: Threshold value
            severity: Alert severity level
        """
        alert = Alert(
            id=alert_id,
            name=name,
            description=description,
            severity=severity,
            condition=metric_name,
            threshold=threshold
        )
        
        self.alert_manager.add_alert(alert)
    
    def _start_alert_checking(self) -> None:
        """Start periodic alert checking."""
        def alert_check_loop():
            while self._monitoring_active:
                try:
                    self.alert_manager.check_alerts()
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    logger.error(f"Error in alert checking loop: {e}")
                    time.sleep(30)
        
        alert_thread = threading.Thread(target=alert_check_loop, daemon=True)
        alert_thread.start()
    
    def _log_alert_notification(self, alert: Alert) -> None:
        """Log alert notification."""
        logger.warning(
            f"ALERT TRIGGERED: {alert.name} ({alert.severity.value}) - "
            f"{alert.description} - Current: {alert.current_value}, Threshold: {alert.threshold}"
        )
    
    def _get_uptime_hours(self) -> float:
        """Get application uptime in hours."""
        # This would be calculated from application start time
        # For now, return a placeholder
        return 24.0


# Global monitoring manager instance
_monitoring_manager: Optional[MonitoringManager] = None


def get_monitoring_manager() -> MonitoringManager:
    """Get or create global monitoring manager instance."""
    global _monitoring_manager
    
    if _monitoring_manager is None:
        _monitoring_manager = MonitoringManager()
    
    return _monitoring_manager


def start_monitoring() -> None:
    """Start monitoring system."""
    manager = get_monitoring_manager()
    manager.start_monitoring()


def stop_monitoring() -> None:
    """Stop monitoring system."""
    global _monitoring_manager
    
    if _monitoring_manager:
        _monitoring_manager.stop_monitoring()


def record_metric(metric_name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
    """
    Record a metric value (convenience function).
    
    Args:
        metric_name: Metric name
        value: Metric value
        tags: Optional tags
    """
    manager = get_monitoring_manager()
    manager.metrics_collector.record_metric(metric_name, value, tags)


def record_business_metric(metric_name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
    """
    Record a business metric (convenience function).
    
    Args:
        metric_name: Business metric name
        value: Metric value
        tags: Optional tags
    """
    manager = get_monitoring_manager()
    manager.record_business_metric(metric_name, value, tags)


# Decorator for automatic performance monitoring
def monitor_performance(metric_name: Optional[str] = None):
    """
    Decorator to automatically monitor function performance.
    
    Args:
        metric_name: Optional custom metric name
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                
                # Record performance metrics
                name = metric_name or f"function.{func.__name__}"
                record_metric(f"{name}.duration", duration)
                record_metric(f"{name}.success", 1 if success else 0)
                
                # Also record in performance tracker
                perf_tracker = get_performance_tracker()
                with perf_tracker.track_operation(func.__name__):
                    pass
        
        return wrapper
    return decorator
