import os
import subprocess
import threading
import time
from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app, jsonify, send_from_directory, send_file, abort, Response, make_response
from flask_login import login_required, current_user
from app.models.file import File
from datetime import datetime
from app import create_app
import base64
import cv2
import numpy as np
from app.utils.converter import convert_stl_to_dxf

viewer_bp = Blueprint('viewer', __name__, url_prefix='/viewer')

@viewer_bp.route('/stl/<file_id>')
@login_required
def view_stl(file_id):
    if file_id not in current_user._files:
        abort(404)
    return render_template('viewer/stl_viewer.html', file=current_user._files[file_id])

@viewer_bp.route('/dxf/<file_id>')
@login_required
def view_dxf(file_id):
    if file_id not in current_user._files:
        abort(404)
    
    file = current_user._files[file_id]
    if not file['processed'] or not file['dxf_path']:
        flash('DXF file not ready yet', 'error')
        return redirect(url_for('dashboard.view_file', file_id=file_id))
    
    return render_template('viewer/dxf_viewer.html', 
                         file=file,
                         file_id=file_id,
                         filename=file['filename'])

@viewer_bp.route('/api/dxf/<file_id>')
@login_required
def get_dxf_content(file_id):
    if file_id not in current_user._files:
        abort(404)
    
    file = current_user._files[file_id]
    if not file['processed'] or not file['dxf_path']:
        abort(404)
    
    try:
        with open(file['dxf_path'], 'r') as f:
            content = f.read()
        return Response(content, mimetype='text/plain')
    except Exception as e:
        print(f"Error reading DXF file: {str(e)}")
        abort(500)

@viewer_bp.route('/api/dxf/<file_id>/download')
@login_required
def download_dxf(file_id):
    if file_id not in current_user._files:
        abort(404)
    
    file = current_user._files[file_id]
    if not file['processed'] or not file['dxf_path']:
        abort(404)
    
    return send_file(
        file['dxf_path'],
        as_attachment=True,
        download_name=os.path.basename(file['dxf_path'])
    )

@viewer_bp.route('/process/<file_id>', methods=['POST'])
@login_required
def process(file_id):
    try:
        print(f"Starting processing for file_id: {file_id}")
        
        if file_id not in current_user._files:
            print(f"File not found with id: {file_id}")
            return jsonify({'status': 'error', 'message': 'File not found'})
        
        file = current_user._files[file_id]
        if file['processed']:
            print(f"File already processed: {file_id}")
            return jsonify({'status': 'complete'})
        
        # Store necessary paths before starting the thread
        app_root = current_app.root_path
        input_path = file['filepath']
        
        print(f"Input path: {input_path}")
        if not os.path.exists(input_path):
            print(f"Input file does not exist at path: {input_path}")
            return jsonify({'status': 'error', 'message': 'Input file not found'})
        
        # Create DXF filename with timestamp to avoid conflicts
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dxf_filename = f"input_{timestamp}.dxf"
        dxf_path = os.path.join(app_root, 'uploads', 'dxf', dxf_filename)
        
        # Ensure the dxf directory exists
        dxf_dir = os.path.dirname(dxf_path)
        if not os.path.exists(dxf_dir):
            print(f"Creating DXF directory: {dxf_dir}")
            os.makedirs(dxf_dir, exist_ok=True)
        
        # Start conversion in background
        file['status'] = 'processing'
        
        def process_in_thread():
            try:
                print(f"Converting file: {input_path}")
                print(f"Output path: {dxf_path}")
                
                # Get the script path
                script_path = os.path.join(app_root, '..', 'stl_to_dxf_works.py')
                print(f"Script path: {script_path}")
                
                if not os.path.exists(script_path):
                    print(f"Conversion script not found at: {script_path}")
                    file['status'] = 'error'
                    return
                
                # Build the command
                cmd = [
                    'python',
                    script_path,
                    input_path,
                    '-o', dxf_path,
                    '-t', '2000',  # Tolerance value
                    '-v',  # Verbose output
                    '--no-sections'  # Don't create section views
                ]
                
                # Run the conversion script
                print(f"Running command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("Command output: ")
                    print(result.stdout)
                    print("Conversion completed successfully")
                    file['status'] = 'complete'
                    file['processed'] = True
                    file['dxf_path'] = dxf_path
                else:
                    print("Error output:")
                    print(result.stderr)
                    file['status'] = 'error'
                    
            except Exception as e:
                print(f"Error during conversion: {str(e)}")
                file['status'] = 'error'
        
        # Start processing in a separate thread
        thread = threading.Thread(target=process_in_thread)
        thread.start()
        print(f"Started processing thread for file: {file_id}")
        
        return jsonify({'status': 'processing'})
        
    except Exception as e:
        print(f"Error in process route: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@viewer_bp.route('/check-status/<file_id>')
@login_required
def check_status(file_id):
    if file_id not in current_user._files:
        abort(404)
    file = current_user._files[file_id]
    return jsonify({
        'status': file['status'],
        'processed': file['processed']
    })

@viewer_bp.route('/download/<file_id>/<format>')
@login_required
def download_file(file_id, format):
    """Download STL or DXF file"""
    if file_id not in current_user._files:
        abort(404)
    file = current_user._files[file_id]
    
    if format.lower() == 'stl':
        if not os.path.exists(file['filepath']):
            abort(404)
        directory = os.path.dirname(file['filepath'])
        filename = os.path.basename(file['filepath'])
        return send_from_directory(directory, filename, as_attachment=True)
    
    elif format.lower() == 'dxf':
        if not file['processed'] or not file['dxf_path'] or not os.path.exists(file['dxf_path']):
            abort(404)
        directory = os.path.dirname(file['dxf_path'])
        filename = os.path.basename(file['dxf_path'])
        return send_from_directory(directory, filename, as_attachment=True)
    
    abort(400)

@viewer_bp.route('/stl-file/<filename>')
@login_required
def serve_stl_file(filename):
    """Serve an STL file for the viewer"""
    return send_from_directory(current_app.config['STL_UPLOAD_FOLDER'], filename)

@viewer_bp.route('/dxf-file/<file_id>')
def serve_dxf_file_new(file_id):
    try:
        file = File.get_by_id(file_id)
        if not file:
            print(f"File not found: {file_id}")
            abort(404)
            
        if not file.processed or not file.dxf_path:
            print(f"File not processed or no DXF path: {file_id}")
            print(f"Processed: {file.processed}, DXF Path: {file.dxf_path}")
            abort(404)
            
        if not os.path.exists(file.dxf_path):
            print(f"DXF file not found at path: {file.dxf_path}")
            abort(404)
            
        try:
            with open(file.dxf_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"DXF file length: {len(content)}")
                
            response = make_response(content)
            response.headers['Content-Type'] = 'text/plain'
            response.headers['Content-Disposition'] = f'inline; filename={os.path.basename(file.dxf_path)}'
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            return response
            
        except Exception as e:
            print(f"Error reading DXF file: {str(e)}")
            abort(500)
            
    except Exception as e:
        print(f"Error serving DXF file: {str(e)}")
        abort(500)

@viewer_bp.route('/add-dimensions/<file_id>', methods=['POST'])
@login_required
def add_dimensions(file_id):
    try:
        # Get the uploaded file info
        file_info = get_file_info(file_id)
        if not file_info:
            return jsonify({'error': 'File not found'}), 404

        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        options = data.get('options', {})
        view_image = data.get('viewImage', '')

        if not view_image:
            return jsonify({'error': 'No view image provided'}), 400

        # Save the view image temporarily
        view_image = view_image.split(',')[1]  # Remove data URL prefix
        view_image_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f'view_{file_id}.jpg')
        with open(view_image_path, 'wb') as f:
            f.write(base64.b64decode(view_image))

        # Get paths
        dxf_path = os.path.join(current_app.config['UPLOAD_FOLDER'], file_info['dxf_filename'])
        
        # Create a background task for processing
        thread = threading.Thread(target=process_dimensions, args=(
            dxf_path,
            view_image_path,
            options,
            file_info
        ))
        thread.start()

        return jsonify({'message': 'Dimensions are being added'}), 200

    except Exception as e:
        current_app.logger.error(f'Error adding dimensions: {str(e)}')
        return jsonify({'error': str(e)}), 500

def process_dimensions(dxf_path, view_image_path, options, file_info):
    try:
        with current_app.app_context():
            # Load the DXF file
            with open(dxf_path, 'r') as f:
                dxf_content = f.read()

            # Load and analyze the view image
            image = cv2.imread(view_image_path)
            if image is None:
                raise Exception('Failed to load view image')

            # Process the image to detect features
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Detect lines using HoughLinesP
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=10)
            
            # Extract dimensions based on options
            dimensions = []
            
            if options.get('overall'):
                # Add overall dimensions
                if lines is not None:
                    # Find bounding box
                    points = np.array([[x1, y1] for line in lines for x1, y1, x2, y2 in line])
                    x_min, y_min = points.min(axis=0)
                    x_max, y_max = points.max(axis=0)
                    
                    # Add width and height dimensions
                    dimensions.append({
                        'type': 'LINEAR',
                        'start': (x_min, y_min),
                        'end': (x_max, y_min),
                        'text_pos': ((x_min + x_max) / 2, y_min - 20),
                        'measurement': abs(x_max - x_min)
                    })
                    dimensions.append({
                        'type': 'LINEAR',
                        'start': (x_min, y_min),
                        'end': (x_min, y_max),
                        'text_pos': (x_min - 20, (y_min + y_max) / 2),
                        'measurement': abs(y_max - y_min)
                    })

            if options.get('features'):
                # Add feature dimensions
                if lines is not None:
                    # Group parallel lines
                    parallel_groups = []
                    for i, line1 in enumerate(lines):
                        x1, y1, x2, y2 = line1[0]
                        angle1 = np.arctan2(y2 - y1, x2 - x1)
                        
                        found_group = False
                        for group in parallel_groups:
                            x3, y3, x4, y4 = lines[group[0]][0]
                            angle2 = np.arctan2(y4 - y3, x4 - x3)
                            
                            if abs(angle1 - angle2) < 0.1 or abs(abs(angle1 - angle2) - np.pi) < 0.1:
                                group.append(i)
                                found_group = True
                                break
                        
                        if not found_group:
                            parallel_groups.append([i])
                    
                    # Add dimensions for parallel features
                    for group in parallel_groups:
                        if len(group) > 1:
                            line1 = lines[group[0]][0]
                            line2 = lines[group[1]][0]
                            
                            # Calculate distance between parallel lines
                            distance = np.abs(
                                (line2[3] - line2[1]) * line1[0] - 
                                (line2[2] - line2[0]) * line1[1] + 
                                line2[2] * line2[1] - line2[3] * line2[0]
                            ) / np.sqrt(
                                (line2[3] - line2[1])**2 + 
                                (line2[2] - line2[0])**2
                            )
                            
                            # Add dimension
                            mid_point1 = ((line1[0] + line1[2])/2, (line1[1] + line1[3])/2)
                            mid_point2 = ((line2[0] + line2[2])/2, (line2[1] + line2[3])/2)
                            
                            dimensions.append({
                                'type': 'LINEAR',
                                'start': mid_point1,
                                'end': mid_point2,
                                'text_pos': ((mid_point1[0] + mid_point2[0])/2, 
                                           (mid_point1[1] + mid_point2[1])/2),
                                'measurement': distance
                            })

            if options.get('holes'):
                # Detect circles (holes)
                circles = cv2.HoughCircles(
                    gray, 
                    cv2.HOUGH_GRADIENT, 
                    dp=1, 
                    minDist=50,
                    param1=50,
                    param2=30,
                    minRadius=10,
                    maxRadius=100
                )
                
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for circle in circles[0, :]:
                        x, y, r = circle
                        
                        # Add diameter dimension
                        dimensions.append({
                            'type': 'DIAMETER',
                            'center': (x, y),
                            'radius': r,
                            'text_pos': (x + r + 10, y),
                            'measurement': 2 * r
                        })

            # Add dimensions to DXF
            for dim in dimensions:
                if dim['type'] == 'LINEAR':
                    dxf_content += f'''
0
DIMENSION
8
Dimensions
2
*D1
70
1
10
{dim['start'][0]}
20
{dim['start'][1]}
11
{dim['end'][0]}
21
{dim['end'][1]}
12
{dim['text_pos'][0]}
22
{dim['text_pos'][1]}
42
{dim['measurement']}
1
{dim['measurement']:.2f}
'''
                elif dim['type'] == 'DIAMETER':
                    dxf_content += f'''
0
DIMENSION
8
Dimensions
2
*D1
70
3
10
{dim['center'][0]}
20
{dim['center'][1]}
40
{dim['radius']}
12
{dim['text_pos'][0]}
22
{dim['text_pos'][1]}
42
{dim['measurement']}
1
Ø{dim['measurement']:.2f}
'''

            # Save the updated DXF file
            with open(dxf_path, 'w') as f:
                f.write(dxf_content)

            # Clean up
            os.remove(view_image_path)

            current_app.logger.info(f'Successfully added dimensions to {file_info["dxf_filename"]}')

    except Exception as e:
        current_app.logger.error(f'Error processing dimensions: {str(e)}')
        raise