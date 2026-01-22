"""Settings API endpoints"""
import os
import yaml
from pathlib import Path
from ..helpers.response_helper import success_response, error_response

# Config file paths
CONFIG_FILE = Path('config/main.yaml')
ENV_FILE = Path('config/.env')

def load_settings():
    """Load notification settings from main.yaml"""
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                notifications = config.get('notifications', {})

                return {
                    'telegram': {
                        'enabled': notifications.get('telegram', {}).get('enabled', False),
                        'bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
                        'chat_id': os.getenv('TELEGRAM_CHAT_ID', '')
                    },
                    'email': {
                        'enabled': notifications.get('email', {}).get('enabled', False),
                        'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
                        'smtp_port': os.getenv('SMTP_PORT', '587'),
                        'username': os.getenv('EMAIL_FROM', ''),
                        'password': os.getenv('EMAIL_PASSWORD', ''),
                        'recipient': os.getenv('EMAIL_TO', '')
                    }
                }
        return {}
    except Exception as e:
        print(f"Error loading settings: {e}")
        return {}

def update_env_file(key, value):
    """Update or add a key-value pair in .env file"""
    try:
        ENV_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Read existing .env content
        env_lines = []
        if ENV_FILE.exists():
            with open(ENV_FILE, 'r', encoding='utf-8') as f:
                env_lines = f.readlines()

        # Update or add the key
        key_found = False
        for i, line in enumerate(env_lines):
            if line.strip().startswith(f'{key}='):
                env_lines[i] = f'{key}={value}\n'
                key_found = True
                break

        if not key_found:
            env_lines.append(f'{key}={value}\n')

        # Write back to .env
        with open(ENV_FILE, 'w', encoding='utf-8') as f:
            f.writelines(env_lines)

        # Update os.environ for immediate effect
        os.environ[key] = value
        return True

    except Exception as e:
        print(f"Error updating .env: {e}")
        return False

def update_yaml_enabled(section, enabled):
    """Update enabled flag in main.yaml"""
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if 'notifications' not in config:
            config['notifications'] = {}
        if section not in config['notifications']:
            config['notifications'][section] = {}

        config['notifications'][section]['enabled'] = enabled

        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        return True
    except Exception as e:
        print(f"Error updating main.yaml: {e}")
        return False

def register_routes(bp):
    """Register settings routes"""

    @bp.route('/settings/notifications', methods=['GET'])
    def get_notification_settings():
        """GET /api/settings/notifications - Get notification configuration"""
        try:
            settings = load_settings()
            return success_response(settings)
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/settings/notifications/telegram', methods=['POST'])
    def save_telegram_settings():
        """POST /api/settings/notifications/telegram - Save Telegram configuration"""
        try:
            from flask import request
            data = request.get_json()

            # Save to .env file
            bot_token = data.get('bot_token', '')
            chat_id = data.get('chat_id', '')
            enabled = data.get('enabled', False)

            success = True
            if bot_token:
                success = success and update_env_file('TELEGRAM_BOT_TOKEN', bot_token)
            if chat_id:
                success = success and update_env_file('TELEGRAM_CHAT_ID', chat_id)

            # Update enabled flag in main.yaml
            success = success and update_yaml_enabled('telegram', enabled)

            if success:
                return success_response({'message': 'Telegram configuration saved to config/.env and main.yaml'})
            else:
                return error_response('Failed to save configuration')

        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/settings/notifications/telegram/test', methods=['POST'])
    def test_telegram_connection():
        """POST /api/settings/notifications/telegram/test - Test Telegram connection"""
        try:
            from flask import request
            import requests

            data = request.get_json()
            bot_token = data.get('bot_token')
            chat_id = data.get('chat_id')

            if not bot_token or not chat_id:
                return error_response('Missing bot_token or chat_id')

            # Send test message via Telegram API
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {
                'chat_id': chat_id,
                'text': '🤖 *SuperBot Test Message*\n\nYour Telegram bot is configured correctly!\n\nYou will receive alert notifications here.',
                'parse_mode': 'Markdown'
            }

            response = requests.post(url, json=payload, timeout=10)
            result = response.json()

            if result.get('ok'):
                return success_response({'message': 'Test message sent successfully'})
            else:
                error_msg = result.get('description', 'Unknown error')
                return error_response(f'Telegram API error: {error_msg}')

        except requests.exceptions.Timeout:
            return error_response('Connection timeout - check your internet connection')
        except requests.exceptions.RequestException as e:
            return error_response(f'Network error: {str(e)}')
        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/settings/notifications/email', methods=['POST'])
    def save_email_settings():
        """POST /api/settings/notifications/email - Save Email configuration"""
        try:
            from flask import request
            data = request.get_json()

            # Save to .env file
            username = data.get('username', '')
            password = data.get('password', '')
            recipient = data.get('recipient', '')
            smtp_server = data.get('smtp_server', '')
            smtp_port = data.get('smtp_port', 587)
            enabled = data.get('enabled', False)

            success = True
            if username:
                success = success and update_env_file('EMAIL_FROM', username)
            if password:
                success = success and update_env_file('EMAIL_PASSWORD', password)
            if recipient:
                success = success and update_env_file('EMAIL_TO', recipient)
            if smtp_server:
                success = success and update_env_file('SMTP_SERVER', smtp_server)
            if smtp_port:
                success = success and update_env_file('SMTP_PORT', str(smtp_port))

            # Update enabled flag in main.yaml
            success = success and update_yaml_enabled('email', enabled)

            if success:
                return success_response({'message': 'Email configuration saved to config/.env'})
            else:
                return error_response('Failed to save configuration')

        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/settings/notifications/email/test', methods=['POST'])
    def test_email_connection():
        """POST /api/settings/notifications/email/test - Test Email connection"""
        try:
            from flask import request
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            data = request.get_json()
            smtp_server = data.get('smtp_server')
            smtp_port = data.get('smtp_port', 587)
            username = data.get('username')
            password = data.get('password')
            recipient = data.get('recipient')

            if not all([smtp_server, username, password, recipient]):
                return error_response('Missing required fields')

            # Create test message
            msg = MIMEMultipart()
            msg['From'] = username
            msg['To'] = recipient
            msg['Subject'] = 'SuperBot - Test Email'

            body = """
<html>
<body>
<h2>🤖 SuperBot Test Email</h2>
<p>Your email notification is configured correctly!</p>
<p>You will receive alert notifications at this email address.</p>
<hr>
<p style="color: #666; font-size: 12px;">This is an automated test message from SuperBot.</p>
</body>
</html>
"""
            msg.attach(MIMEText(body, 'html'))

            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
            server.quit()

            return success_response({'message': 'Test email sent successfully'})

        except smtplib.SMTPAuthenticationError:
            return error_response('Authentication failed - check your username and password')
        except smtplib.SMTPException as e:
            return error_response(f'SMTP error: {str(e)}')
        except Exception as e:
            return error_response(str(e), 500)
