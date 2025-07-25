"""
Alert management and notification system.
"""
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
import json
import os
from pathlib import Path

@dataclass
class Alert:
    """Alert configuration and state."""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    message_template: str
    severity: str
    cooldown: timedelta
    last_triggered: Optional[datetime] = None

@dataclass
class AlertNotification:
    """Alert notification details."""
    alert_name: str
    message: str
    severity: str
    timestamp: datetime
    context: Dict[str, Any]

class AlertManager:
    """Manage system alerts and notifications."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize alert manager with optional config path."""
        self.alerts: Dict[str, Alert] = {}
        self.notifications: List[AlertNotification] = []
        self.config = self._load_config(config_path)
        self._setup_default_alerts()
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load alert configuration from file."""
        if not config_path:
            config_path = os.path.join(os.path.dirname(__file__), 'alert_config.json')
        
        if not os.path.exists(config_path):
            # Create default config if it doesn't exist
            default_config = {
                'email': {
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'username': '',
                    'password': '',
                    'from_address': '',
                    'to_addresses': []
                },
                'thresholds': {
                    'error_rate': 0.1,
                    'response_time': 5.0,
                    'api_latency': 2.0
                }
            }
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            return default_config
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _setup_default_alerts(self) -> None:
        """Set up default system alerts."""
        self.add_alert(
            name='high_error_rate',
            condition=lambda metrics: metrics.get('error_rate', 0) > self.config['thresholds']['error_rate'],
            message_template='High error rate detected: {error_rate:.2%}',
            severity='critical',
            cooldown=timedelta(minutes=30)
        )
        
        self.add_alert(
            name='slow_response_time',
            condition=lambda metrics: metrics.get('response_time', {}).get('p95', 0) > self.config['thresholds']['response_time'],
            message_template='Slow response time detected. P95: {response_time[p95]:.2f}s',
            severity='warning',
            cooldown=timedelta(minutes=15)
        )
        
        self.add_alert(
            name='api_latency',
            condition=lambda metrics: any(
                provider['mean'] > self.config['thresholds']['api_latency']
                for provider in metrics.get('api_latency', {}).values()
            ),
            message_template='High API latency detected for providers: {affected_providers}',
            severity='warning',
            cooldown=timedelta(minutes=15)
        )
    
    def add_alert(self, name: str, condition: Callable[[Dict[str, Any]], bool],
                 message_template: str, severity: str, cooldown: timedelta) -> None:
        """Add a new alert configuration."""
        self.alerts[name] = Alert(
            name=name,
            condition=condition,
            message_template=message_template,
            severity=severity,
            cooldown=cooldown
        )
    
    def check_alerts(self, metrics: Dict[str, Any]) -> List[AlertNotification]:
        """Check all alerts against current metrics."""
        triggered_alerts = []
        current_time = datetime.now()
        
        for alert in self.alerts.values():
            # Skip if alert is in cooldown
            if (alert.last_triggered and 
                current_time - alert.last_triggered < alert.cooldown):
                continue
            
            try:
                if alert.condition(metrics):
                    # Format message with metrics
                    message = alert.message_template.format(**metrics)
                    
                    # Create notification
                    notification = AlertNotification(
                        alert_name=alert.name,
                        message=message,
                        severity=alert.severity,
                        timestamp=current_time,
                        context=metrics
                    )
                    
                    triggered_alerts.append(notification)
                    self.notifications.append(notification)
                    
                    # Update last triggered time
                    alert.last_triggered = current_time
                    
                    # Send notification
                    self._send_notification(notification)
            except Exception as e:
                print(f"Error checking alert {alert.name}: {str(e)}")
        
        return triggered_alerts
    
    def _send_notification(self, notification: AlertNotification) -> None:
        """Send alert notification via configured channels."""
        # Email notification
        if self.config['email']['to_addresses']:
            try:
                msg = MIMEText(
                    f"Alert: {notification.alert_name}\n"
                    f"Severity: {notification.severity}\n"
                    f"Time: {notification.timestamp}\n"
                    f"Message: {notification.message}\n"
                    f"Context: {json.dumps(notification.context, indent=2)}"
                )
                
                msg['Subject'] = f"[{notification.severity.upper()}] LLM Structure Elucidator Alert"
                msg['From'] = self.config['email']['from_address']
                msg['To'] = ', '.join(self.config['email']['to_addresses'])
                
                with smtplib.SMTP(
                    self.config['email']['smtp_server'],
                    self.config['email']['smtp_port']
                ) as server:
                    server.starttls()
                    if self.config['email']['username'] and self.config['email']['password']:
                        server.login(
                            self.config['email']['username'],
                            self.config['email']['password']
                        )
                    server.send_message(msg)
            except Exception as e:
                print(f"Error sending email notification: {str(e)}")
    
    def get_recent_alerts(self, hours: int = 24) -> List[AlertNotification]:
        """Get list of alerts from the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.notifications
            if alert.timestamp >= cutoff_time
        ]
