import uuid
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

class User(UserMixin):
    # Dictionary to store users
    _users = {}
    
    def __init__(self, username, email=None, password_hash=None):
        self.id = str(uuid.uuid4())  # Generate UUID automatically
        self.username = username
        self.email = email or f"{username}@example.com"  # Default email if not provided
        self.password_hash = password_hash
        self._files = {}  # Dictionary to store user's files
        # Store the user instance in the class dictionary
        self.__class__._users[self.id] = self
    
    def set_password(self, password):
        """Set password hash"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password against hash"""
        return check_password_hash(self.password_hash, password)
    
    def get_files(self):
        """Get all files as a list"""
        return list(self._files.values())
    
    def add_file(self, file_info):
        """Add a file using its ID as the key"""
        self._files[file_info['id']] = file_info
    
    def to_dict(self):
        """Convert user object to dictionary"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'files': self._files
        }
    
    @classmethod
    def get_by_id(cls, user_id):
        """Get a user by their ID"""
        return cls._users.get(user_id)
    
    @classmethod
    def get_by_username(cls, username):
        """Get a user by their username"""
        for user in cls._users.values():
            if user.username == username:
                return user
        return None
    
    @classmethod
    def get_by_email(cls, email):
        """Get a user by their email"""
        for user in cls._users.values():
            if user.email == email:
                return user
        return None
    
    @classmethod
    def register(cls, username, password, email=None):
        """Register a new user"""
        # Check if username or email already exists
        if cls.get_by_username(username):
            return None
        if email and cls.get_by_email(email):
            return None
        
        # Create new user
        user = cls(username=username, email=email)
        user.set_password(password)
        return user

    @classmethod
    def get_dummy_user(cls):
        """Get or create the dummy user"""
        dummy_user = cls.get_by_username("demo")
        if not dummy_user:
            dummy_user = cls(username="demo")
            dummy_user.set_password("demo123")
        return dummy_user