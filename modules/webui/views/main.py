"""Main page routes"""
from flask import render_template, session, flash, redirect, url_for

def register_routes(bp):
    """Register main routes"""

    @bp.route('/')
    def index():
        """Dashboard - Main page"""
        return render_template('dashboard.html',
                             page='dashboard',
                             title='Dashboard')

    @bp.route('/profile')
    def profile():
        """Profile page"""
        return render_template('profile.html',
                             page='profile',
                             title='Profile')

    @bp.route('/symbols')
    def symbols():
        """Symbols page - Manage symbols and categories"""
        return render_template('symbols.html',
                             page='symbols',
                             title='Symbols')

    @bp.route('/data/download')
    def data_download():
        """Data Download page - Download historical data"""
        return render_template('data_download.html',
                             page='data-download',
                             title='Data Download')

    @bp.route('/notifications')
    def notifications():
        """Notifications history page"""
        return render_template('notifications.html',
                             page='notifications',
                             title='Notifications')

    @bp.route('/logout')
    def logout():
        """Logout - clear session and redirect"""
        session.clear()
        flash('Successfully logged out!', 'success')
        return redirect(url_for('views.index'))
