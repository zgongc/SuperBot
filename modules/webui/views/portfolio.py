"""Portfolio page routes"""
from flask import render_template

def register_routes(bp):
    """Register portfolio routes"""

    @bp.route('/portfolios')
    def portfolios():
        """Portfolios list page - Manage multiple portfolios"""
        return render_template('portfolios.html',
                             page='portfolio',
                             title='Portfolios')

    @bp.route('/portfolio/<int:portfolio_id>')
    def portfolio_detail(portfolio_id):
        """Portfolio detail page - View positions and P&L"""
        return render_template('portfolio_detail.html',
                             page='portfolio',
                             title='Portfolio Detail',
                             portfolio_id=portfolio_id)

    @bp.route('/portfolio')
    def portfolio_legacy():
        """Legacy portfolio page (redirect to portfolios list)"""
        return render_template('portfolio.html',
                             page='portfolio',
                             title='Portfolio')
