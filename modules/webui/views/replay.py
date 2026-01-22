"""Replay Trading page routes"""
from flask import render_template


def register_routes(bp):
    """Register replay trading routes"""

    @bp.route('/trading/replay')
    def replay():
        """Replay Trading - Tarihsel veri üzerinde strateji simülasyonu"""
        return render_template('replay.html',
                             page='replay',
                             title='Replay Trading')
