"""Candlestick Pattern Analysis page routes"""
from flask import render_template


def register_routes(bp):
    """Register pattern analysis routes"""

    @bp.route('/analysis/patterns')
    def patterns_analysis():
        """Pattern Analysis - Candlestick Pattern Detection"""
        return render_template('patterns_analysis.html',
                             page='patterns',
                             title='Pattern Analysis')
