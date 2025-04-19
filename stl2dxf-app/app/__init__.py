import os
from flask import Flask
from config import Config
from flask_login import LoginManager
from app.models.user import User

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    """Load user for Flask-Login"""
    return User.get_by_id(user_id)

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Ensure upload directories exist
    os.makedirs(os.path.join(app.root_path, 'uploads/stl'), exist_ok=True)
    os.makedirs(os.path.join(app.root_path, 'uploads/dxf'), exist_ok=True)
    
    # Initialize extensions
    login_manager.init_app(app)
    
    # Register blueprints
    from app.views.auth import auth_bp
    from app.views.dashboard import dashboard_bp
    from app.views.viewer import viewer_bp
    from app.api import api_bp
    
    app.register_blueprint(auth_bp)  # Register at root URL
    app.register_blueprint(dashboard_bp, url_prefix='/dashboard')
    app.register_blueprint(viewer_bp, url_prefix='/viewer')
    app.register_blueprint(api_bp)
    
    @app.after_request
    def add_header(response):
        """Add headers to prevent caching for development"""
        response.headers['Cache-Control'] = 'no-store'
        return response
    
    # Register error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return {'error': 'Not found'}, 404

    @app.errorhandler(500)
    def internal_error(error):
        return {'error': 'Internal server error'}, 500
    
    return app