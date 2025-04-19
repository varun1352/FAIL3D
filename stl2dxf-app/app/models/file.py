from datetime import datetime
import uuid

class File:
    """Model class for storing file information"""
    
    _files = {}  # In-memory storage for development
    
    def __init__(self, filename, filepath, upload_date, user_id, status='uploaded'):
        self.id = str(uuid.uuid4())
        self.filename = filename
        self.filepath = filepath
        self.upload_date = upload_date
        self.user_id = user_id
        self.status = status
        self.dxf_path = None
        self.processed = False
    
    def save(self):
        """Save file information to storage"""
        File._files[self.id] = self
        return self
    
    @classmethod
    def get_by_id(cls, file_id):
        """Retrieve file by ID"""
        return cls._files.get(file_id)
    
    @classmethod
    def get_user_files(cls, user_id):
        """Get all files for a specific user"""
        return [f for f in cls._files.values() if f.user_id == user_id]
    
    @classmethod
    def get_all_files(cls):
        """Get all files"""
        return list(cls._files.values())
    
    def update_status(self, status):
        """Update file processing status"""
        self.status = status
        self.save()
    
    def set_dxf_path(self, dxf_path):
        """Set the path to the generated DXF file"""
        self.dxf_path = dxf_path
        self.processed = True
        self.save()
    
    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'filename': self.filename,
            'upload_date': self.upload_date,
            'status': self.status,
            'processed': self.processed,
            'user_id': self.user_id
        } 