"""Settings page routes"""
from flask import render_template

def register_routes(bp):
    """Register settings routes"""

    @bp.route('/settings')
    def settings():
        """Settings page"""
        return render_template('settings.html',
                             page='settings',
                             title='Settings')

    @bp.route('/settings/notifications')
    def notifications_settings():
        """Notification Settings page - Telegram & Email configuration"""
        return render_template('notifications_settings.html',
                             page='settings',
                             title='Notification Settings')

    @bp.route('/exchange-accounts')
    def exchange_accounts():
        """Exchange Accounts page - Multi-exchange management"""
        return render_template('exchange_accounts.html',
                             page='exchange-accounts',
                             title='Exchange Accounts')
