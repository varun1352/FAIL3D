from flask import Blueprint
from flask_restx import Api

api_bp = Blueprint('api', __name__, url_prefix='/api')
api = Api(api_bp,
          title='STL to DXF Converter API',
          version='1.0',
          description='API for converting STL files to DXF format',
          doc='/docs')

# Import and register endpoints
from app.api.endpoints import files_ns
api.add_namespace(files_ns) 