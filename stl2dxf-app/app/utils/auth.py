from functools import wraps
from flask import redirect, url_for, session

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # For development, we'll assume the user is always logged in
        # In production, you would check session['user_id'] exists
        return f(*args, **kwargs)
    return decorated_function 