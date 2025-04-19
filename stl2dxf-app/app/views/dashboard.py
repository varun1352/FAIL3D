from flask import Blueprint, render_template, request, flash, redirect, url_for, current_app, send_file, jsonify
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from app.models.user import User
from app.models.file import File
from app.utils.auth import login_required
from flask_login import login_required, current_user
from app.utils.converter import convert_stl_to_dxf

dashboard_bp = Blueprint('dashboard', __name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'stl'}

@dashboard_bp.route('/')
@login_required
def index():
    # Get user's files
    files = current_user.get_files()
    return render_template('dashboard/index.html', files=files)

@dashboard_bp.route('/settings')
@login_required
def settings():
    return render_template('dashboard/settings.html')

@dashboard_bp.route('/about')
@login_required
def about():
    return render_template('dashboard/about.html')

@dashboard_bp.route('/help')
@login_required
def help():
    return render_template('dashboard/help.html')

@dashboard_bp.route('/upload')
@login_required
def upload():
    return render_template('dashboard/upload.html')

@dashboard_bp.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('dashboard.upload'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('dashboard.upload'))
    
    if not file.filename.lower().endswith('.stl'):
        flash('Only STL files are allowed', 'error')
        return redirect(url_for('dashboard.upload'))
    
    try:
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{secure_filename(file.filename)}"
        filepath = os.path.join(current_app.root_path, 'uploads/stl', filename)
        
        # Save the file
        file.save(filepath)
        
        # Add file info to user's files
        file_info = {
            'id': timestamp,
            'filename': file.filename,  # Original filename for display
            'stored_filename': filename,  # Actual filename on disk
            'upload_date': datetime.now(),
            'status': 'uploaded',
            'type': 'stl',
            'processed': False,
            'filepath': filepath
        }
        current_user.add_file(file_info)
        
        flash('File uploaded successfully', 'success')
        return redirect(url_for('dashboard.view_file', file_id=timestamp))
        
    except Exception as e:
        flash(f'Error uploading file: {str(e)}', 'error')
        return redirect(url_for('dashboard.upload'))

@dashboard_bp.route('/view/<file_id>')
@login_required
def view_file(file_id):
    # Check user's files
    if file_id in current_user._files:
        return render_template('dashboard/view.html', file=current_user._files[file_id])
    
    flash('File not found', 'error')
    return redirect(url_for('dashboard.index'))

@dashboard_bp.route('/download/stl/<file_id>')
@login_required
def download_stl(file_id):
    if file_id not in current_user._files:
        flash('File not found', 'error')
        return redirect(url_for('dashboard.index'))
    
    file = current_user._files[file_id]
    return send_file(
        file['filepath'],
        as_attachment=False,  # Stream the file instead of downloading
        mimetype='model/stl'
    )

@dashboard_bp.route('/process/<file_id>', methods=['POST'])
@login_required
def process_file(file_id):
    if file_id not in current_user._files:
        return jsonify({'status': 'error', 'message': 'File not found'}), 404
    
    file = current_user._files[file_id]
    if file['processed']:
        return jsonify({'status': 'complete', 'message': 'File already processed'})
    
    try:
        # Start the conversion
        success, result = convert_stl_to_dxf(file['filepath'])
        
        if success:
            # Update file info
            file['processed'] = True
            file['status'] = 'complete'
            file['dxf_path'] = result
            return jsonify({'status': 'complete'})
        else:
            return jsonify({'status': 'error', 'message': result})
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@dashboard_bp.route('/download/dxf/<file_id>')
@login_required
def download_dxf(file_id):
    if file_id not in current_user._files:
        flash('File not found', 'error')
        return redirect(url_for('dashboard.index'))
    
    file = current_user._files[file_id]
    if not file['processed']:
        flash('DXF file not ready yet', 'error')
        return redirect(url_for('dashboard.index'))
    
    return send_file(
        file['dxf_path'],
        as_attachment=True,
        download_name=f"{file['filename'].replace('.stl', '.dxf')}"
    )

@dashboard_bp.route('/view/dxf/<file_id>')
@login_required
def view_dxf(file_id):
    if file_id not in current_user._files:
        flash('File not found', 'error')
        return redirect(url_for('dashboard.index'))
    
    file = current_user._files[file_id]
    if not file['processed']:
        flash('DXF file not ready yet', 'error')
        return redirect(url_for('dashboard.view_file', file_id=file_id))
    
    return render_template('viewer/dxf_viewer.html', file=file)