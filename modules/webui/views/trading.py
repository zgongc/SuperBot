"""Trading page routes"""
from flask import render_template

def register_routes(bp):
    """Register trading routes"""

    @bp.route('/trade')
    def trade():
        """Trade page - Opportunities list"""
        return render_template('trade.html',
                             page='trade',
                             title='Trade')

    @bp.route('/backtest')
    def backtest():
        """Backtest page - Historical results"""
        return render_template('backtest.html',
                             page='backtest',
                             title='Backtest')

    @bp.route('/analysis')
    def analysis():
        """Analysis page - Charts & analysis"""
        return render_template('analysis.html',
                             page='analysis',
                             title='Analysis')
