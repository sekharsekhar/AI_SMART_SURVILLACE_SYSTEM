"""
SMTP Email Alert Engine
Sends email notifications when abnormal activities are detected
Supports HTML emails with image attachments
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from typing import List, Optional, Dict
from datetime import datetime
import time
import os
import cv2
import numpy as np
import threading
from queue import Queue


class EmailAlertEngine:
    """
    SMTP-based email alert system for surveillance notifications
    
    Features:
        - HTML email templates with detection details
        - Frame snapshot attachments
        - Cooldown period to prevent email flooding
        - Async email sending (non-blocking)
        - Alert priority levels
    """
    
    PRIORITY_COLORS = {
        'critical': '#ff0000',
        'high': '#ff6600',
        'medium': '#ffcc00',
        'low': '#00cc00'
    }
    
    def __init__(
        self,
        smtp_host: str = '',
        smtp_port: int = 587,
        smtp_username: str = '',
        smtp_password: str = '',
        from_address: str = '',
        recipients: List[str] = None,
        use_tls: bool = True,
        cooldown_seconds: int = 30,
        enabled: bool = True
    ):
        """
        Initialize email alert engine
        
        Args:
            smtp_host: SMTP server hostname (e.g., smtp.gmail.com)
            smtp_port: SMTP port (587 for TLS, 465 for SSL)
            smtp_username: SMTP authentication username
            smtp_password: SMTP authentication password
            from_address: Sender email address
            recipients: List of recipient email addresses
            use_tls: Whether to use TLS encryption
            cooldown_seconds: Minimum seconds between alerts of same type
            enabled: Whether email alerts are enabled
        """
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_username = smtp_username
        self.smtp_password = smtp_password
        self.from_address = from_address
        self.recipients = recipients or []
        self.use_tls = use_tls
        self.cooldown_seconds = cooldown_seconds
        self.enabled = enabled
        
        # Track last alert times per activity type
        self.last_alert_times: Dict[str, float] = {}
        
        # Async email queue
        self.email_queue = Queue()
        self.worker_thread = None
        self._stop_worker = False
        
        # Start background worker
        self._start_worker()
    
    def _start_worker(self):
        """Start background email sending worker"""
        self._stop_worker = False
        self.worker_thread = threading.Thread(target=self._email_worker, daemon=True)
        self.worker_thread.start()
    
    def _email_worker(self):
        """Background worker that processes email queue"""
        while not self._stop_worker:
            try:
                if not self.email_queue.empty():
                    email_data = self.email_queue.get(timeout=1)
                    self._send_email_sync(email_data)
                else:
                    time.sleep(0.5)
            except Exception as e:
                print(f"Email worker error: {e}")
    
    def stop(self):
        """Stop the email worker thread"""
        self._stop_worker = True
        if self.worker_thread:
            self.worker_thread.join(timeout=2)
    
    def is_configured(self) -> bool:
        """Check if SMTP is properly configured"""
        return bool(
            self.smtp_host and 
            self.smtp_username and 
            self.smtp_password and 
            self.from_address and 
            len(self.recipients) > 0
        )
    
    def can_send_alert(self, activity_type: str) -> bool:
        """
        Check if enough time has passed since last alert
        
        Args:
            activity_type: Type of detected activity
            
        Returns:
            True if cooldown period has passed
        """
        if not self.enabled:
            print(f"[DEBUG] can_send_alert: NOT enabled")
            return False
        
        if not self.is_configured():
            print(f"[DEBUG] can_send_alert: NOT configured")
            return False
        
        current_time = time.time()
        last_time = self.last_alert_times.get(activity_type, 0)
        
        can_send = (current_time - last_time) >= self.cooldown_seconds
        print(f"[DEBUG] can_send_alert: {activity_type}, cooldown_passed={can_send}")
        return can_send
    
    def send_alert(
        self,
        activity_type: str,
        priority: str = 'medium',
        camera_name: str = 'Camera 1',
        zone_name: str = 'Main Area',
        confidence: float = 0.0,
        description: str = '',
        frame: Optional[np.ndarray] = None,
        snapshot_path: Optional[str] = None
    ) -> bool:
        """
        Queue an alert email for sending
        
        Args:
            activity_type: Type of detected activity (violence, intrusion, etc.)
            priority: Alert priority (critical, high, medium, low)
            camera_name: Name of the camera source
            zone_name: Name of the detection zone
            confidence: Detection confidence score (0-1)
            description: Additional description
            frame: Video frame to attach as snapshot
            snapshot_path: Path to save snapshot (if frame provided)
            
        Returns:
            True if alert was queued successfully
        """
        if not self.can_send_alert(activity_type):
            return False
        
        # Update last alert time
        self.last_alert_times[activity_type] = time.time()
        
        # Save snapshot if frame provided
        if frame is not None and snapshot_path:
            os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
            cv2.imwrite(snapshot_path, frame)
        
        # Prepare email data
        email_data = {
            'activity_type': activity_type,
            'priority': priority,
            'camera_name': camera_name,
            'zone_name': zone_name,
            'confidence': confidence,
            'description': description,
            'timestamp': datetime.now(),
            'snapshot_path': snapshot_path if snapshot_path and os.path.exists(snapshot_path) else None
        }
        
        # Queue for async sending
        self.email_queue.put(email_data)
        print(f"[ALERT] {activity_type.upper()} detected! Email queued.")
        
        return True
    
    def _send_email_sync(self, email_data: Dict) -> bool:
        """
        Synchronously send email (called by worker thread)
        
        Args:
            email_data: Dictionary with email details
            
        Returns:
            True if sent successfully
        """
        try:
            # Create message
            msg = MIMEMultipart('related')
            msg['Subject'] = self._generate_subject(email_data)
            msg['From'] = self.from_address
            msg['To'] = ', '.join(self.recipients)
            
            # Create HTML body
            html_body = self._generate_html_body(email_data)
            msg.attach(MIMEText(html_body, 'html'))
            
            # Attach snapshot if available
            if email_data.get('snapshot_path'):
                with open(email_data['snapshot_path'], 'rb') as f:
                    img_data = f.read()
                    img = MIMEImage(img_data)
                    img.add_header('Content-ID', '<snapshot>')
                    img.add_header('Content-Disposition', 'attachment', 
                                   filename='detection_snapshot.jpg')
                    msg.attach(img)
            
            # Send email
            if self.use_tls:
                context = ssl.create_default_context()
                with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                    server.starttls(context=context)
                    server.login(self.smtp_username, self.smtp_password)
                    server.sendmail(self.from_address, self.recipients, msg.as_string())
            else:
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, context=context) as server:
                    server.login(self.smtp_username, self.smtp_password)
                    server.sendmail(self.from_address, self.recipients, msg.as_string())
            
            print(f"[EMAIL SENT] Alert for {email_data['activity_type']} sent to {len(self.recipients)} recipients")
            return True
            
        except Exception as e:
            print(f"[EMAIL ERROR] Failed to send alert: {e}")
            return False
    
    def _generate_subject(self, email_data: Dict) -> str:
        """Generate email subject line"""
        priority = email_data['priority'].upper()
        activity = email_data['activity_type'].replace('_', ' ').title()
        return f"[{priority} ALERT] {activity} Detected - AI Surveillance System"
    
    def _generate_html_body(self, email_data: Dict) -> str:
        """Generate HTML email body"""
        priority_color = self.PRIORITY_COLORS.get(email_data['priority'], '#999999')
        timestamp = email_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        confidence_pct = int(email_data['confidence'] * 100)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; background-color: #f4f4f4; margin: 0; padding: 20px; }}
                .container {{ max-width: 600px; margin: 0 auto; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ background: {priority_color}; color: white; padding: 20px; text-align: center; }}
                .header h1 {{ margin: 0; font-size: 24px; }}
                .header .priority {{ font-size: 14px; opacity: 0.9; }}
                .content {{ padding: 30px; }}
                .alert-icon {{ font-size: 48px; text-align: center; margin-bottom: 20px; }}
                .detail-row {{ display: flex; padding: 10px 0; border-bottom: 1px solid #eee; }}
                .detail-label {{ font-weight: bold; color: #666; width: 140px; }}
                .detail-value {{ color: #333; }}
                .confidence-bar {{ background: #eee; height: 20px; border-radius: 10px; overflow: hidden; margin-top: 5px; }}
                .confidence-fill {{ background: {priority_color}; height: 100%; width: {confidence_pct}%; }}
                .snapshot {{ margin-top: 20px; text-align: center; }}
                .snapshot img {{ max-width: 100%; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.2); }}
                .footer {{ background: #f8f8f8; padding: 15px; text-align: center; color: #888; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>⚠️ SURVEILLANCE ALERT</h1>
                    <div class="priority">{email_data['priority'].upper()} PRIORITY</div>
                </div>
                
                <div class="content">
                    <div class="alert-icon">🚨</div>
                    
                    <h2 style="text-align: center; color: {priority_color};">
                        {email_data['activity_type'].replace('_', ' ').title()} Detected
                    </h2>
                    
                    <div class="detail-row">
                        <span class="detail-label">Timestamp:</span>
                        <span class="detail-value">{timestamp}</span>
                    </div>
                    
                    <div class="detail-row">
                        <span class="detail-label">Camera:</span>
                        <span class="detail-value">{email_data['camera_name']}</span>
                    </div>
                    
                    <div class="detail-row">
                        <span class="detail-label">Location:</span>
                        <span class="detail-value">{email_data['zone_name']}</span>
                    </div>
                    
                    <div class="detail-row">
                        <span class="detail-label">Confidence:</span>
                        <span class="detail-value">
                            {confidence_pct}%
                            <div class="confidence-bar">
                                <div class="confidence-fill"></div>
                            </div>
                        </span>
                    </div>
                    
                    {f'<div class="detail-row"><span class="detail-label">Details:</span><span class="detail-value">{email_data["description"]}</span></div>' if email_data.get('description') else ''}
                    
                    {f'<div class="snapshot"><h3>Detection Snapshot</h3><img src="cid:snapshot" alt="Detection Snapshot"></div>' if email_data.get('snapshot_path') else ''}
                </div>
                
                <div class="footer">
                    AI-Based Intelligent Video Surveillance System<br>
                    This is an automated alert. Please review the footage immediately.
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def test_connection(self) -> Dict:
        """
        Test SMTP connection and authentication
        
        Returns:
            Dictionary with test result and message
        """
        if not self.is_configured():
            return {
                'success': False,
                'message': 'SMTP not configured. Please set host, username, password, and recipients.'
            }
        
        try:
            if self.use_tls:
                context = ssl.create_default_context()
                with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10) as server:
                    server.starttls(context=context)
                    server.login(self.smtp_username, self.smtp_password)
            else:
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, context=context, timeout=10) as server:
                    server.login(self.smtp_username, self.smtp_password)
            
            return {
                'success': True,
                'message': f'Successfully connected to {self.smtp_host}:{self.smtp_port}'
            }
            
        except smtplib.SMTPAuthenticationError:
            return {
                'success': False,
                'message': 'Authentication failed. Check username and password.'
            }
        except smtplib.SMTPConnectError:
            return {
                'success': False,
                'message': f'Could not connect to {self.smtp_host}:{self.smtp_port}'
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Connection error: {str(e)}'
            }
    
    def send_test_email(self) -> Dict:
        """
        Send a test email to verify configuration
        
        Returns:
            Dictionary with result
        """
        if not self.is_configured():
            return {
                'success': False,
                'message': 'SMTP not configured'
            }
        
        # Create simple test message
        msg = MIMEMultipart()
        msg['Subject'] = '[TEST] AI Surveillance System - Email Configuration Test'
        msg['From'] = self.from_address
        msg['To'] = ', '.join(self.recipients)
        
        html = """
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h2 style="color: #28a745;">✅ Email Configuration Successful!</h2>
            <p>This is a test email from your AI-Based Intelligent Video Surveillance System.</p>
            <p>Your SMTP settings are correctly configured and alerts will be delivered to this address.</p>
            <hr>
            <p style="color: #666; font-size: 12px;">AI Surveillance System</p>
        </body>
        </html>
        """
        msg.attach(MIMEText(html, 'html'))
        
        try:
            if self.use_tls:
                context = ssl.create_default_context()
                with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                    server.starttls(context=context)
                    server.login(self.smtp_username, self.smtp_password)
                    server.sendmail(self.from_address, self.recipients, msg.as_string())
            else:
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, context=context) as server:
                    server.login(self.smtp_username, self.smtp_password)
                    server.sendmail(self.from_address, self.recipients, msg.as_string())
            
            return {
                'success': True,
                'message': f'Test email sent to {", ".join(self.recipients)}'
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Failed to send: {str(e)}'
            }


# Test function
if __name__ == '__main__':
    # Example configuration
    alert_engine = EmailAlertEngine(
        smtp_host='smtp.gmail.com',
        smtp_port=587,
        smtp_username='your-email@gmail.com',
        smtp_password='your-app-password',
        from_address='your-email@gmail.com',
        recipients=['recipient@example.com'],
        use_tls=True,
        cooldown_seconds=30,
        enabled=True
    )
    
    # Test connection
    result = alert_engine.test_connection()
    print(f"Connection test: {result}")
    
    # Don't actually send in test mode
    print("Email alert engine initialized successfully")
