"""Theme API endpoints"""
from flask import session, jsonify

def register_routes(bp):
    """Register theme routes"""

    @bp.route('/theme/toggle', methods=['POST'])
    def toggle_theme():
        """POST /api/theme/toggle - Toggle dark/light theme"""
        current_theme = session.get('theme', 'dark')
        new_theme = 'light' if current_theme == 'dark' else 'dark'
        session['theme'] = new_theme

        return jsonify({
            'status': 'success',
            'data': {
                'theme': new_theme
            }
        })
