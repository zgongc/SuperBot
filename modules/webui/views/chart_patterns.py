"""Trendline Breakout Analysis page routes"""
from flask import render_template


def register_routes(bp):
    """Register trendline breakout analysis routes"""

    @bp.route('/analysis/chart-patterns')
    def chart_patterns():
        """Trendline Breakout Analysis - Support/Resistance breakouts"""
        return render_template('chart_patterns.html',
                             page='chart-patterns',
                             title='Trendline Breakout')
