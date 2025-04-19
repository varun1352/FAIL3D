from flask import request, send_file
from flask_restx import Resource, fields, reqparse
from werkzeug.datastructures import FileStorage
from app.api import api
from app.models.file import File
from app.utils.converter import convert_stl_to_dxf
import os
from datetime import datetime

# Define namespaces
files_ns = api.namespace('files', description='File operations')

# Define models for documentation
file_model = api.model('File', {
    'id': fields.String(required=True, description='File ID'),
    'filename': fields.String(required=True, description='Original filename'),
    'upload_date': fields.DateTime(required=True, description='Upload date'),
    'status': fields.String(required=True, description='Processing status'),
    'processed': fields.Boolean(required=True, description='Whether the file has been processed'),
})

# File upload parser
upload_parser = reqparse.RequestParser()
upload_parser.add_argument('file',
                         type=FileStorage,
                         location='files',
                         required=True,
                         help='STL file to upload')

@files_ns.route('/')
class FileList(Resource):
    @files_ns.marshal_list_with(file_model)
    @files_ns.doc('list_files',
                 description='Get list of all files',
                 responses={200: 'Success'})
    def get(self):
        """List all files"""
        # For demo, we'll return all files
        return File.get_all_files()

    @files_ns.expect(upload_parser)
    @files_ns.marshal_with(file_model)
    @files_ns.doc('upload_file',
                 description='Upload a new STL file',
                 responses={
                     201: 'File uploaded successfully',
                     400: 'Invalid file type'
                 })
    def post(self):
        """Upload a new file"""
        args = upload_parser.parse_args()
        uploaded_file = args['file']
        
        if not uploaded_file.filename.lower().endswith('.stl'):
            api.abort(400, "Only STL files are allowed")
        
        # Save file
        filename = uploaded_file.filename
        filepath = os.path.join('app/uploads/stl', f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}")
        uploaded_file.save(filepath)
        
        # Create file record
        new_file = File(
            filename=filename,
            filepath=filepath,
            upload_date=datetime.now(),
            user_id='demo',  # For demo purposes
            status='uploaded'
        )
        new_file.save()
        
        return new_file, 201

@files_ns.route('/<string:file_id>')
@files_ns.param('file_id', 'The file identifier')
class FileDetail(Resource):
    @files_ns.marshal_with(file_model)
    @files_ns.doc('get_file',
                 description='Get a file by ID',
                 responses={
                     200: 'Success',
                     404: 'File not found'
                 })
    def get(self, file_id):
        """Get a file by ID"""
        file = File.get_by_id(file_id)
        if not file:
            api.abort(404, "File not found")
        return file

@files_ns.route('/<string:file_id>/convert')
@files_ns.param('file_id', 'The file identifier')
class FileConversion(Resource):
    @files_ns.doc('convert_file',
                 description='Convert STL file to DXF',
                 responses={
                     200: 'Conversion completed',
                     404: 'File not found',
                     409: 'File already processed'
                 })
    def post(self, file_id):
        """Start file conversion"""
        file = File.get_by_id(file_id)
        if not file:
            api.abort(404, "File not found")
        
        if file.processed:
            api.abort(409, "File already processed")
        
        # Create DXF filename
        dxf_filename = os.path.splitext(file.filename)[0] + '.dxf'
        dxf_path = os.path.join('app/uploads/dxf', dxf_filename)
        
        # Start conversion
        file.status = 'processing'
        file.save()
        
        success, message = convert_stl_to_dxf(file.filepath)
        
        if success:
            file.status = 'complete'
            file.processed = True
            file.dxf_path = dxf_path
        else:
            file.status = 'error'
            file.processed = False
        file.save()
        
        return {'message': 'Conversion completed', 'status': file.status}, 200

@files_ns.route('/<string:file_id>/status')
@files_ns.param('file_id', 'The file identifier')
class FileStatus(Resource):
    @files_ns.doc('get_status',
                 description='Get file processing status',
                 responses={
                     200: 'Success',
                     404: 'File not found'
                 })
    def get(self, file_id):
        """Get file processing status"""
        file = File.get_by_id(file_id)
        if not file:
            api.abort(404, "File not found")
        
        return {
            'status': file.status,
            'processed': file.processed
        }

@files_ns.route('/<string:file_id>/download/<string:format>')
@files_ns.param('file_id', 'The file identifier')
@files_ns.param('format', 'File format to download (stl or dxf)')
class FileDownload(Resource):
    @files_ns.doc('download_file',
                 description='Download file in specified format',
                 responses={
                     200: 'Success',
                     404: 'File not found',
                     400: 'Invalid format or file not processed'
                 })
    def get(self, file_id, format):
        """Download file"""
        file = File.get_by_id(file_id)
        if not file:
            api.abort(404, "File not found")
        
        if format.lower() == 'stl':
            if not os.path.exists(file.filepath):
                api.abort(404, "STL file not found")
            return send_file(file.filepath,
                           as_attachment=True,
                           download_name=file.filename)
        
        elif format.lower() == 'dxf':
            if not file.processed or not file.dxf_path or not os.path.exists(file.dxf_path):
                api.abort(400, "DXF file not available")
            download_name = os.path.splitext(file.filename)[0] + '.dxf'
            return send_file(file.dxf_path,
                           as_attachment=True,
                           download_name=download_name)
        
        else:
            api.abort(400, "Invalid format. Use 'stl' or 'dxf'") 