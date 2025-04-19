import os
from datetime import timedelta

class Config:
    # Application settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-for-demo-only'
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    
    # File upload settings
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max upload size
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app', 'uploads')
    STL_UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER, 'stl')
    DXF_UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER, 'dxf')
    ALLOWED_EXTENSIONS = {'stl'}
    
    # Session settings
    PERMANENT_SESSION_LIFETIME = timedelta(days=1)
    
    # For demo purposes, store user data in memory
    USERS = {}
    USER_FILES = {}