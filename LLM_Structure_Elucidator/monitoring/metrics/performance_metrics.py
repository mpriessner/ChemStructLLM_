"""
Performance metrics collection and tracking.
"""
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import statistics

@dataclass
class MetricPoint:
    """Single metric measurement point."""
    value: float
    timestamp: datetime
    labels: Dict[str, str]

class PerformanceMetrics:
    """Collect and track performance metrics."""
    
    def __init__(self):
        self.metrics = {
            'response_time': [],
            'token_count': [],
            'api_latency': [],
            'error_rate': [],
            'request_count': 0
        }
        self._start_time: Optional[float] = None
    
    def start_request(self) -> None:
        """Start timing a request."""
        self._start_time = time.time()
    
    def end_request(self, success: bool = True) -> float:
        """End timing a request and record metrics."""
        if not self._start_time:
            raise ValueError("start_request() must be called before end_request()")
        
        duration = time.time() - self._start_time
        self.metrics['response_time'].append(
            MetricPoint(
                value=duration,
                timestamp=datetime.now(),
                labels={'success': str(success)}
            )
        )
        self.metrics['request_count'] += 1
        self._start_time = None
        return duration
    
    def record_token_count(self, count: int, model: str) -> None:
        """Record token usage for a request."""
        self.metrics['token_count'].append(
            MetricPoint(
                value=count,
                timestamp=datetime.now(),
                labels={'model': model}
            )
        )
    
    def record_api_latency(self, latency: float, provider: str) -> None:
        """Record API latency."""
        self.metrics['api_latency'].append(
            MetricPoint(
                value=latency,
                timestamp=datetime.now(),
                labels={'provider': provider}
            )
        )
    
    def record_error(self, error_type: str) -> None:
        """Record an error occurrence."""
        self.metrics['error_rate'].append(
            MetricPoint(
                value=1.0,
                timestamp=datetime.now(),
                labels={'error_type': error_type}
            )
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of collected metrics."""
        summary = {}
        
        # Response time stats
        response_times = [m.value for m in self.metrics['response_time']]
        if response_times:
            summary['response_time'] = {
                'mean': statistics.mean(response_times),
                'median': statistics.median(response_times),
                'p95': statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else None
            }
        
        # Token usage by model
        token_counts = {}
        for metric in self.metrics['token_count']:
            model = metric.labels['model']
            if model not in token_counts:
                token_counts[model] = []
            token_counts[model].append(metric.value)
        
        summary['token_usage'] = {
            model: {
                'total': sum(counts),
                'mean': statistics.mean(counts)
            }
            for model, counts in token_counts.items()
        }
        
        # API latency by provider
        latencies = {}
        for metric in self.metrics['api_latency']:
            provider = metric.labels['provider']
            if provider not in latencies:
                latencies[provider] = []
            latencies[provider].append(metric.value)
        
        summary['api_latency'] = {
            provider: {
                'mean': statistics.mean(lats),
                'p95': statistics.quantiles(lats, n=20)[18] if len(lats) >= 20 else None
            }
            for provider, lats in latencies.items()
        }
        
        # Error rate
        total_requests = self.metrics['request_count']
        total_errors = len(self.metrics['error_rate'])
        summary['error_rate'] = total_errors / total_requests if total_requests > 0 else 0
        
        return summary
