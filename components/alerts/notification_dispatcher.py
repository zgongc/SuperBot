"""
Notification Dispatcher - Send alert notifications to various channels
"""

import json
from core.logger_engine import LoggerEngine
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional


class NotificationDispatcher:
    """Dispatch notifications to various channels (WebUI, Telegram, Email)"""

    # Class-level pending WebUI notifications queue
    pending_webui_notifications = []

    def __init__(self, data_manager):
        """
        Initialize dispatcher

        Args:
            data_manager: DataManager instance for database access
        """
        self.data_manager = data_manager
        # LoggerEngine setup
        logger_engine = LoggerEngine()
        self.logger = logger_engine.get_logger(__name__)

        # Notification channels (initialized on demand)
        self.telegram_bot = None
        self.email_client = None

    async def dispatch(self, alert: Dict[str, Any], analysis_result: Dict[str, Any]) -> Optional[int]:
        """
        Send notifications to all enabled channels

        Args:
            alert: Alert object that was triggered
            analysis_result: Analysis result that triggered the alert

        Returns:
            Notification record ID, or None if failed
        """
        try:
            # Create notification record
            notification_id = await self.create_notification_record(alert, analysis_result)

            # Dispatch to enabled channels in parallel
            tasks = []

            if alert.get('notify_webui'):
                tasks.append(self.send_webui(notification_id, alert, analysis_result))

            if alert.get('notify_telegram'):
                tasks.append(self.send_telegram(notification_id, alert, analysis_result))

            if alert.get('notify_email'):
                tasks.append(self.send_email(notification_id, alert, analysis_result))

            # Execute all in parallel
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Log any errors
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        channel = ['webui', 'telegram', 'email'][i]
                        self.logger.error(f"Notification dispatch failed for {channel}: {result}")

            # Update alert statistics
            await self.update_alert_stats(alert['id'])

            self.logger.info(f"Notification {notification_id} dispatched successfully")
            return notification_id

        except Exception as e:
            self.logger.error(f"Notification dispatch failed: {e}")
            return None

    async def create_notification_record(self, alert: Dict[str, Any], result: Dict[str, Any]) -> int:
        """Create a notification record in database"""

        trigger_details = self.build_trigger_details(alert, result)

        query = """
            INSERT INTO alert_notifications (
                alert_id,
                analysis_result_id,
                trigger_type,
                trigger_details,
                triggered_at
            ) VALUES (?, ?, ?, ?, ?)
        """

        notification_id = await self.data_manager.execute(
            query,
            (
                alert['id'],
                result.get('id'),
                alert['alert_type'],
                json.dumps(trigger_details),
                datetime.now().isoformat()
            )
        )

        return notification_id

    async def send_webui(self, notification_id: int, alert: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """
        Send WebUI toast notification

        Adds notification to pending queue for frontend polling
        """
        try:
            # Format message
            message = self.format_webui_message(alert, result)

            # Add notification ID and timestamp
            message['notification_id'] = notification_id
            message['timestamp'] = datetime.now().isoformat()

            # Add to pending queue (class-level shared queue)
            NotificationDispatcher.pending_webui_notifications.append(message)

            # Keep only last 50 notifications in queue
            if len(NotificationDispatcher.pending_webui_notifications) > 50:
                NotificationDispatcher.pending_webui_notifications = \
                    NotificationDispatcher.pending_webui_notifications[-50:]

            self.logger.info(f"WebUI notification queued: {message['title']}")

            # Mark as sent
            await self.mark_channel_sent(notification_id, 'webui', True)

            return True

        except Exception as e:
            self.logger.error(f"WebUI notification failed: {e}")
            return False

    async def send_telegram(self, notification_id: int, alert: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """Send Telegram notification"""
        try:
            # Get Telegram configuration
            telegram_config = await self.get_telegram_config()

            if not telegram_config or not telegram_config.get('enabled'):
                self.logger.debug("Telegram not configured, skipping")
                return False

            # Format message
            message = self.format_telegram_message(alert, result)

            # TODO: Send via Telegram bot API
            # For now, just log it
            self.logger.info(f"Telegram notification: {message[:100]}...")

            # Mark as sent
            await self.mark_channel_sent(notification_id, 'telegram', True)

            return True

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Telegram notification failed: {e}")

            # Mark as failed with error
            await self.mark_channel_sent(notification_id, 'telegram', False, error_msg)

            return False

    async def send_email(self, notification_id: int, alert: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """Send Email notification"""
        try:
            # Get Email configuration
            email_config = await self.get_email_config()

            if not email_config or not email_config.get('enabled'):
                self.logger.debug("Email not configured, skipping")
                return False

            # Format message
            subject, body = self.format_email_message(alert, result)

            # TODO: Send via SMTP
            # For now, just log it
            self.logger.info(f"Email notification: {subject}")

            # Mark as sent
            await self.mark_channel_sent(notification_id, 'email', True)

            return True

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Email notification failed: {e}")

            # Mark as failed with error
            await self.mark_channel_sent(notification_id, 'email', False, error_msg)

            return False

    def format_webui_message(self, alert: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Format message for WebUI toast notification"""

        symbol_name = result.get('symbol', 'Unknown')
        alert_type = alert['alert_type'].replace('_', ' ').title()

        # Build title based on alert type
        if alert['alert_type'] == 'pattern':
            title = f"Pattern Detected! 🎨"
        elif alert['alert_type'] == 'signal':
            title = f"Signal Generated! ⚡"
        elif alert['alert_type'] == 'score':
            title = f"Score Alert! 📊"
        elif alert['alert_type'] == 'trend_change':
            title = f"Trend Changed! 📈"
        else:
            title = f"Alert Triggered! 🔔"

        # Build message
        message_parts = [f"{alert['name']}", f"Symbol: {symbol_name}"]

        # Add specific details
        trigger_details = self.build_trigger_details(alert, result)
        if 'details' in trigger_details:
            message_parts.append(trigger_details['details'])

        message = " • ".join(message_parts)

        return {
            'title': title,
            'message': message,
            'type': 'success',
            'duration': 5000,
            'icon': '🔔',
            'analysis_result_id': result.get('id')
        }

    def format_telegram_message(self, alert: Dict[str, Any], result: Dict[str, Any]) -> str:
        """Format message for Telegram"""

        symbol_name = result.get('symbol', 'Unknown')
        alert_type_display = alert['alert_type'].replace('_', ' ').title()

        # Build message
        lines = [
            "🔔 <b>Analysis Alert Triggered!</b>",
            "",
            f"📊 <b>Alert:</b> {alert['name']}",
            f"📈 <b>Symbol:</b> {symbol_name}",
            f"🎯 <b>Type:</b> {alert_type_display}",
            ""
        ]

        # Add trigger details
        trigger_details = self.build_trigger_details(alert, result)
        if 'details' in trigger_details:
            lines.append(trigger_details['details'])
            lines.append("")

        # Add timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        lines.append(f"🕒 <b>Time:</b> {timestamp}")

        # Add link to analysis
        result_id = result.get('id')
        if result_id:
            # TODO: Get actual WebUI URL from config
            url = f"http://localhost:5001/analysis?id={result_id}"
            lines.append("")
            lines.append(f"<a href=\"{url}\">View Full Analysis</a>")

        return "\n".join(lines)

    def format_email_message(self, alert: Dict[str, Any], result: Dict[str, Any]) -> tuple:
        """Format message for Email"""

        symbol_name = result.get('symbol', 'Unknown')

        # Subject
        subject = f"[SuperBot Alert] {alert['name']} - {symbol_name}"

        # Body (simple text for now, could be HTML)
        trigger_details = self.build_trigger_details(alert, result)

        body = f"""
Analysis Alert Triggered

Alert: {alert['name']}
Symbol: {symbol_name}
Type: {alert['alert_type'].replace('_', ' ').title()}

{trigger_details.get('details', '')}

Triggered: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

View analysis results in SuperBot WebUI.
        """.strip()

        return subject, body

    def build_trigger_details(self, alert: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Build specific trigger details based on alert type"""

        alert_type = alert['alert_type']
        details = {}

        try:
            if alert_type == 'pattern':
                patterns = result.get('patterns', [])
                if isinstance(patterns, str):
                    patterns = json.loads(patterns)

                if patterns:
                    pattern_names = [p.get('name', 'Unknown') for p in patterns]
                    details['details'] = f"🎨 <b>Patterns:</b> {', '.join(pattern_names)}"
                    details['pattern_names'] = pattern_names

            elif alert_type == 'signal':
                signal = result.get('signal_type', 'N/A')
                strength = result.get('signal_strength', 0)
                details['details'] = f"⚡ <b>Signal:</b> {signal} (Strength: {strength:.1%})"
                details['signal_type'] = signal
                details['strength'] = strength

            elif alert_type == 'score':
                score = result.get('overall_score', 0)
                details['details'] = f"📊 <b>Score:</b> {score:.1f}"
                details['score'] = score

            elif alert_type == 'trend_change':
                trend = result.get('trend', 'N/A')
                details['details'] = f"📈 <b>New Trend:</b> {trend}"
                details['trend'] = trend

        except Exception as e:
            self.logger.error(f"Failed to build trigger details: {e}")

        return details

    async def mark_channel_sent(self, notification_id: int, channel: str, success: bool, error: Optional[str] = None):
        """Mark a notification channel as sent (or failed)"""

        timestamp = datetime.now().isoformat()

        if channel == 'webui':
            query = """
                UPDATE alert_notifications
                SET webui_sent = ?, webui_sent_at = ?
                WHERE id = ?
            """
            await self.data_manager.execute(query, (1 if success else 0, timestamp, notification_id))

        elif channel == 'telegram':
            query = """
                UPDATE alert_notifications
                SET telegram_sent = ?, telegram_sent_at = ?, telegram_error = ?
                WHERE id = ?
            """
            await self.data_manager.execute(query, (1 if success else 0, timestamp, error, notification_id))

        elif channel == 'email':
            query = """
                UPDATE alert_notifications
                SET email_sent = ?, email_sent_at = ?, email_error = ?
                WHERE id = ?
            """
            await self.data_manager.execute(query, (1 if success else 0, timestamp, error, notification_id))

    async def update_alert_stats(self, alert_id: int):
        """Update alert trigger statistics"""

        query = """
            UPDATE analysis_alerts
            SET trigger_count = trigger_count + 1,
                last_triggered_at = ?,
                updated_at = ?
            WHERE id = ?
        """

        timestamp = datetime.now().isoformat()
        await self.data_manager.execute(query, (timestamp, timestamp, alert_id))

    async def get_telegram_config(self) -> Optional[Dict[str, Any]]:
        """Get Telegram configuration from settings"""
        # TODO: Implement settings lookup
        # For now, return None
        return None

    async def get_email_config(self) -> Optional[Dict[str, Any]]:
        """Get Email configuration from settings"""
        # TODO: Implement settings lookup
        # For now, return None
        return None
