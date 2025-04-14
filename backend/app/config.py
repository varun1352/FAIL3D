from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "3D-to-DXF Converter"
    
    # File Upload Settings
    UPLOAD_DIR: Path = Path("uploads")
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: set = {"stl", "obj"}
    
    # Processing Settings
    MAX_TRIANGLES: int = 1000000  # Maximum number of triangles to process
    
    class Config:
        case_sensitive = True

settings = Settings()

# Create upload directory if it doesn't exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True) 