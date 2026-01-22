"""SMC (Smart Money Concepts) Analysis page routes"""
from flask import render_template


def register_routes(bp):
    """Register SMC analysis routes"""

    @bp.route('/analysis/smc')
    def smc_analysis():
        """SMC Analysis - Market Structure Analysis"""
        return render_template('smc_analysis.html',
                             page='smc',
                             title='SMC Analysis')
