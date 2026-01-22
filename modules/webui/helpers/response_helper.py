"""Standard JSON response builders"""
from flask import jsonify

def success_response(data=None, message=None):
    """Build success response"""
    response = {'status': 'success'}
    if data is not None:
        response['data'] = data
    if message:
        response['message'] = message
    return jsonify(response)

def error_response(message, status_code=500):
    """Build error response"""
    return jsonify({
        'status': 'error',
        'message': str(message)
    }), status_code
