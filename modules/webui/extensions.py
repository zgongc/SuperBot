"""Flask extensions initialization"""
from flask_cors import CORS

# Flask extensions
cors = None

def init_extensions(app):
    """Initialize Flask extensions"""
    global cors
    cors = CORS(app)
