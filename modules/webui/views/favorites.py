"""Favorites page routes"""
from flask import render_template

def register_routes(bp):
    """Register favorites routes"""

    @bp.route('/favorites')
    def favorites():
        """Favorites page - Manage favorite symbols"""
        return render_template('favorites.html',
                             page='favorites',
                             title='Favorites')
