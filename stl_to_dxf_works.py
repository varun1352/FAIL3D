#!/usr/bin/env python3
# stl_to_dxf_optimized.py - Convert STL files to DXF with optimizations for complex models
# run using 
# python stl_to_dxf_works.py input.stl -t 2000 -v --no-sections 
import numpy as np
import trimesh
import ezdxf
import os
import sys
import argparse
import time
import logging
from typing import Tuple, List, Dict, Optional, Set, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_stl(file_path: str, simplify: bool = True, target_faces: int = 5000) -> trimesh.Trimesh:
    """
    Load an STL file and return a trimesh mesh, with optional simplification.
    
    Args:
        file_path: Path to the STL file
        simplify: Whether to simplify the mesh
        target_faces: Target number of faces for simplification
        
    Returns:
        Loaded mesh object
    """
    try:
        start_time = time.time()
        original_mesh = trimesh.load(file_path)
        load_time = time.time() - start_time
        
        logger.info(f"\nSTL Loading Details:")
        logger.info(f"- File: {file_path}")
        logger.info(f"- Faces: {len(original_mesh.faces):,}")
        logger.info(f"- Vertices: {len(original_mesh.vertices):,}")
        logger.info(f"- Loading time: {load_time:.2f} seconds")
        logger.info(f"- Volume: {original_mesh.volume:.2f} cubic units")
        logger.info(f"- Surface area: {original_mesh.area:.2f} square units")
        logger.info(f"- Is watertight: {original_mesh.is_watertight}")
        
        # Simplify the mesh if it's complex
        if simplify and len(original_mesh.faces) > target_faces:
            logger.info(f"Simplifying mesh (target: {target_faces:,} faces)...")
            simplify_start = time.time()
            try:
                mesh = original_mesh.simplify_quadratic_decimation(target_faces)
                simplify_time = time.time() - simplify_start
                logger.info(f"- Simplified from {len(original_mesh.faces):,} to {len(mesh.faces):,} faces")
                logger.info(f"- Simplification time: {simplify_time:.2f} seconds")
            except Exception as e:
                logger.warning(f"Mesh simplification failed: {e}")
                logger.warning("Using original mesh instead")
                mesh = original_mesh
        else:
            mesh = original_mesh
        
        # Center the mesh on the origin for better processing
        mesh.apply_translation(-mesh.centroid)
        
        return mesh
    except Exception as e:
        logger.error(f"Error loading STL file: {e}")
        sys.exit(1)

def extract_silhouette_edges(mesh: trimesh.Trimesh, direction: np.ndarray, max_time: float = 5.0) -> np.ndarray:
    """
    Extract edges that form the silhouette when viewed from a particular direction.
    Includes a timeout to handle complex meshes.
    
    Args:
        mesh: Input trimesh mesh
        direction: View direction as unit vector
        max_time: Maximum processing time in seconds
        
    Returns:
        Array of edges forming the silhouette
    """
    start_time = time.time()
    
    # Normalize direction vector
    direction = np.array(direction) / np.linalg.norm(direction)
    
    # Calculate dot product of face normals with view direction
    dots = np.dot(mesh.face_normals, direction)
    
    # Find pairs of faces where one is front-facing and one is back-facing
    silhouette_edges = []
    edges_seen = set()
    
    # For very large meshes, use a simplified approach
    if len(mesh.faces) > 20000:
        logger.info("Large mesh detected, using simplified edge extraction")
        # Just get edges where adjacent faces have opposite orientations
        for i, edge in enumerate(mesh.edges_unique):
            # Get the faces that share this edge
            face_indices = mesh.edges_unique_faces[i]
            if len(face_indices) == 2:  # Only consider edges with exactly 2 adjacent faces
                f1, f2 = face_indices
                # If one face faces toward and one faces away, this is a silhouette edge
                if dots[f1] * dots[f2] <= 0:
                    v1, v2 = mesh.vertices[edge[0]], mesh.vertices[edge[1]]
                    silhouette_edges.append([v1, v2])
                    
                    # Check timeout
                    if time.time() - start_time > max_time:
                        logger.warning(f"Silhouette extraction taking too long, returning partial results")
                        break
        
        processing_time = time.time() - start_time
        logger.info(f"Extracted {len(silhouette_edges)} silhouette edges in {processing_time:.2f} seconds (simplified method)")
        return np.array(silhouette_edges)
    
    # For smaller meshes, use the more accurate face adjacency method
    # Check mesh has necessary attributes
    has_face_adjacency = hasattr(mesh, 'face_adjacency') and hasattr(mesh, 'face_adjacency_edges')
    
    if not has_face_adjacency:
        logger.warning("Mesh does not have face adjacency data. Computing it now...")
        try:
            # Compute face adjacency data
            mesh.face_adjacency  # This will compute it if not already computed
            mesh.face_adjacency_edges
            has_face_adjacency = True
        except Exception as e:
            logger.error(f"Unable to compute face adjacency: {e}")
            has_face_adjacency = False
    
    if has_face_adjacency:
        # Use face adjacency data to find silhouette edges
        for i, (face_idx, adj_idx) in enumerate(mesh.face_adjacency):
            # If one face faces toward and one faces away, this is a silhouette edge
            if dots[face_idx] * dots[adj_idx] <= 0:
                # Get the edge vertices
                edge = mesh.face_adjacency_edges[i]
                edge_key = tuple(sorted(edge))
                
                if edge_key not in edges_seen:
                    edges_seen.add(edge_key)
                    # Create an edge from the vertices
                    v1, v2 = mesh.vertices[edge[0]], mesh.vertices[edge[1]]
                    silhouette_edges.append([v1, v2])
            
            # Check timeout
            if time.time() - start_time > max_time:
                logger.warning(f"Silhouette extraction taking too long, returning partial results")
                break
    else:
        # Fallback method: use edge connectivity
        logger.warning("Using fallback method for silhouette edge detection")
        # Get all edges
        edges = mesh.edges_unique
        edges_faces = mesh.edges_unique_faces
        
        for i, edge in enumerate(edges):
            # Get the faces that share this edge
            face_indices = edges_faces[i]
            if len(face_indices) == 2:  # Only consider edges with exactly 2 adjacent faces
                f1, f2 = face_indices
                # If one face faces toward and one faces away, this is a silhouette edge
                if dots[f1] * dots[f2] <= 0:
                    v1, v2 = mesh.vertices[edge[0]], mesh.vertices[edge[1]]
                    silhouette_edges.append([v1, v2])
            
            # Check timeout
            if time.time() - start_time > max_time:
                logger.warning(f"Silhouette extraction taking too long, returning partial results")
                break
    
    processing_time = time.time() - start_time
    logger.info(f"Extracted {len(silhouette_edges)} silhouette edges in {processing_time:.2f} seconds")
    
    return np.array(silhouette_edges)

def extract_section_edges(mesh: trimesh.Trimesh, origin: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """
    Extract edges from a cross-section of the mesh.
    
    Args:
        mesh: Input trimesh mesh
        origin: Origin point of the section plane
        normal: Normal vector of the section plane
        
    Returns:
        Array of edges in the section
    """
    try:
        # Get the cross-section
        section = mesh.section(plane_origin=origin, plane_normal=normal)
        if section is None or not hasattr(section, 'entities') or len(section.entities) == 0:
            return np.array([])
        
        # Extract edges from the section
        edges = []
        for entity in section.entities:
            if hasattr(entity, 'points'):
                points = entity.points
                # Create edges between consecutive points
                for i in range(len(points) - 1):
                    edges.append([points[i], points[i+1]])
                # Close the loop if it's a closed path
                if hasattr(entity, 'closed') and entity.closed:
                    edges.append([points[-1], points[0]])
        
        return np.array(edges)
    except Exception as e:
        logger.warning(f"Failed to extract section: {e}")
        return np.array([])

def project_edges_to_2d(edges: np.ndarray, view_matrix: np.ndarray) -> np.ndarray:
    """
    Project 3D edges onto a 2D plane using a view transformation matrix.
    
    Args:
        edges: 3D edges to project
        view_matrix: 4x4 transformation matrix
        
    Returns:
        2D projected edges
    """
    if len(edges) == 0:
        return np.array([])
    
    # Process each edge
    projected_edges = []
    for edge in edges:
        # Convert to homogeneous coordinates (add w=1)
        p1_h = np.append(edge[0], 1)
        p2_h = np.append(edge[1], 1)
        
        # Apply transformation
        p1_transformed = np.dot(view_matrix, p1_h)
        p2_transformed = np.dot(view_matrix, p2_h)
        
        # Convert to 2D (drop z)
        p1_2d = p1_transformed[:2] / p1_transformed[3] if p1_transformed[3] != 0 else p1_transformed[:2]
        p2_2d = p2_transformed[:2] / p2_transformed[3] if p2_transformed[3] != 0 else p2_transformed[:2]
        
        # Add to projected edges
        projected_edges.append([p1_2d, p2_2d])
    
    return np.array(projected_edges)

def clean_edges(edges: np.ndarray, tolerance: float = 1e-6, min_length_factor: float = 0.01) -> np.ndarray:
    """
    Clean up edges by removing duplicates, merging colinear segments, and removing
    very short edges that are likely numerical artifacts.
    
    Args:
        edges: Input edges
        tolerance: Numerical tolerance for comparisons
        min_length_factor: Factor of max length to use as minimum length threshold
        
    Returns:
        Cleaned edges
    """
    if len(edges) == 0:
        return edges
    
    # Convert edges to a unique representation
    unique_edges = set()
    for edge in edges:
        # Skip edges with NaN or inf
        if (np.isnan(edge[0]).any() or np.isnan(edge[1]).any() or 
            np.isinf(edge[0]).any() or np.isinf(edge[1]).any()):
            continue
        
        # Sort the vertices to handle reversed edges
        p1 = tuple(np.round(edge[0], decimals=6))
        p2 = tuple(np.round(edge[1], decimals=6))
        if p1 != p2:  # Skip zero-length edges
            if p1 < p2:
                unique_edges.add((p1, p2))
            else:
                unique_edges.add((p2, p1))
    
    # Convert back to numpy array
    edges = np.array([[np.array(e[0]), np.array(e[1])] for e in unique_edges])
    
    if len(edges) == 0:
        return edges
    
    # Remove very short edges
    edge_lengths = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1)
    max_length = np.max(edge_lengths)
    min_length_threshold = max_length * min_length_factor
    edges = edges[edge_lengths > min_length_threshold]
    
    if len(edges) == 0:
        return edges
    
    # Merge colinear segments (with a timeout)
    start_time = time.time()
    merged_edges = []
    used = set()
    
    for i, edge1 in enumerate(edges):
        if i in used:
            continue
            
        current_edge = edge1.copy()
        used.add(i)
        
        # Timeout for merging colinear segments (3 seconds)
        if time.time() - start_time > 3.0:
            # If timeout, just add the remaining edges without merging
            for j, edge in enumerate(edges):
                if j not in used:
                    merged_edges.append(edge.copy())
                    used.add(j)
            break
        
        while True:
            found_continuation = False
            for j, edge2 in enumerate(edges):
                if j in used:
                    continue
                    
                # Check if edges share an endpoint (within tolerance)
                if np.allclose(current_edge[1], edge2[0], atol=tolerance):
                    # Check if they're colinear
                    v1 = current_edge[1] - current_edge[0]
                    v2 = edge2[1] - edge2[0]
                    v1_norm = np.linalg.norm(v1)
                    v2_norm = np.linalg.norm(v2)
                    
                    if v1_norm > 0 and v2_norm > 0:
                        v1_normalized = v1 / v1_norm
                        v2_normalized = v2 / v2_norm
                        
                        if np.allclose(v1_normalized, v2_normalized, atol=0.01):
                            current_edge[1] = edge2[1]
                            used.add(j)
                            found_continuation = True
                            break
                        
            if not found_continuation or (time.time() - start_time > 3.0):
                break
                
        merged_edges.append(current_edge)
    
    return np.array(merged_edges)

def get_orthographic_projections(mesh: trimesh.Trimesh, use_sections: bool = True) -> Dict[str, np.ndarray]:
    """
    Generate orthographic projections (top, front, right) from the mesh.
    
    Args:
        mesh: Input trimesh mesh
        use_sections: Whether to include cross-sections in the projections
        
    Returns:
        Dictionary with projections for each view
    """
    start_time = time.time()
    logger.info("\nGenerating Orthographic Projections:")
    
    # Define view directions and transformation matrices
    views = {
        'top': {
            'direction': np.array([0, 0, 1]),
            'matrix': np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 1]
            ]),
            'section_origin': np.array([0, 0, 0]),
            'section_normal': np.array([0, 0, 1])
        },
        'front': {
            'direction': np.array([0, -1, 0]),
            'matrix': np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 1]
            ]),
            'section_origin': np.array([0, 0, 0]),
            'section_normal': np.array([0, 1, 0])
        },
        'right': {
            'direction': np.array([1, 0, 0]),
            'matrix': np.array([
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 1]
            ]),
            'section_origin': np.array([0, 0, 0]),
            'section_normal': np.array([1, 0, 0])
        }
    }
    
    # Get edges for each view
    projections = {}
    
    for view_name, view_data in views.items():
        view_start = time.time()
        
        # Extract silhouette edges for this view (with timeout)
        silhouette_edges = extract_silhouette_edges(
            mesh, 
            view_data['direction'],
            max_time=5.0  # 5 second timeout per view
        )
        logger.info(f"View {view_name}: Extracted {len(silhouette_edges)} silhouette edges")
        
        # Add section edges if requested
        section_edges = np.array([])
        if use_sections:
            # Add multiple sections through the model at different levels
            bounds = mesh.bounds
            extents = bounds[1] - bounds[0]
            sections = []
            
            # Get the relevant axis for sections based on view
            if view_name == 'top':
                axis = 2  # Z axis for top view
                levels = np.linspace(bounds[0][axis], bounds[1][axis], 5)
            elif view_name == 'front':
                axis = 1  # Y axis for front view
                levels = np.linspace(bounds[0][axis], bounds[1][axis], 5)
            else:  # right view
                axis = 0  # X axis for right view
                levels = np.linspace(bounds[0][axis], bounds[1][axis], 5)
            
            # Create sections at each level
            for level in levels:
                origin = np.zeros(3)
                origin[axis] = level
                section = extract_section_edges(mesh, origin, view_data['section_normal'])
                if len(section) > 0:
                    sections.append(section)
            
            # Combine all sections
            if sections:
                section_edges = np.vstack(sections)
                logger.info(f"View {view_name}: Added {len(section_edges)} section edges")
        
        # Combine silhouette and section edges
        combined_edges = np.vstack([silhouette_edges, section_edges]) if len(section_edges) > 0 else silhouette_edges
        
        # Project to 2D
        projected_edges = project_edges_to_2d(combined_edges, view_data['matrix'])
        logger.info(f"View {view_name}: Projected edges to 2D")
        
        # Clean up the edges
        clean_projected = clean_edges(projected_edges, min_length_factor=0.01)
        logger.info(f"View {view_name}: Cleaned to {len(clean_projected)} edges")
        
        # Store the result
        projections[view_name] = clean_projected
        
        view_time = time.time() - view_start
        logger.info(f"View {view_name} completed in {view_time:.2f} seconds")
    
    total_time = time.time() - start_time
    logger.info(f"Total projection time: {total_time:.2f} seconds")
    
    return projections

def normalize_projection(edges: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Normalize a projection to fit in a standard box while preserving aspect ratio.
    Returns the normalized edges and the transformation parameters.
    
    Args:
        edges: Input 2D edges
        
    Returns:
        Tuple of (normalized_edges, transform_params)
    """
    if len(edges) == 0:
        return edges, {'scale': 1.0, 'offset_x': 0.0, 'offset_y': 0.0, 'min_x': 0.0, 'min_y': 0.0, 'width': 0.0, 'height': 0.0}
    
    # Find bounds
    all_points = np.vstack([edges[:, 0], edges[:, 1]])
    min_x = np.min(all_points[:, 0])
    max_x = np.max(all_points[:, 0])
    min_y = np.min(all_points[:, 1])
    max_y = np.max(all_points[:, 1])
    
    # Calculate dimensions
    width = max_x - min_x
    height = max_y - min_y
    
    # Determine scale factor
    target_size = 100.0  # Target size for the larger dimension
    scale = target_size / max(width, height) if max(width, height) > 0 else 1.0
    
    # Normalize edges
    normalized = []
    for edge in edges:
        p1 = [(edge[0][0] - min_x) * scale, (edge[0][1] - min_y) * scale]
        p2 = [(edge[1][0] - min_x) * scale, (edge[1][1] - min_y) * scale]
        normalized.append([p1, p2])
    
    transform_params = {
        'scale': scale,
        'offset_x': -min_x,
        'offset_y': -min_y,
        'min_x': min_x,
        'min_y': min_y,
        'width': width,
        'height': height
    }
    
    return np.array(normalized), transform_params

def create_dxf(projections: Dict[str, np.ndarray], output_path: str, mesh: trimesh.Trimesh) -> None:
    """
    Create a DXF file with the orthographic projections.
    
    Args:
        projections: Dict of orthographic projections
        output_path: Path to save the DXF file
        mesh: Original mesh for dimensions
    """
    start_time = time.time()
    logger.info("\nCreating DXF File:")
    
    # Get model dimensions
    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]
    
    logger.info(f"Model dimensions:")
    logger.info(f"- Width (X): {extents[0]:.2f}")
    logger.info(f"- Depth (Y): {extents[1]:.2f}")
    logger.info(f"- Height (Z): {extents[2]:.2f}")
    
    # Normalize projections
    normalized_projections = {}
    transform_params = {}
    
    for view_name, edges in projections.items():
        normalized, params = normalize_projection(edges)
        normalized_projections[view_name] = normalized
        transform_params[view_name] = params
        
        logger.info(f"Normalized {view_name} view:")
        logger.info(f"- Original size: {params['width']:.2f} x {params['height']:.2f}")
        logger.info(f"- Scale factor: {params['scale']:.4f}")
    
    # Start with a clean DXF R12 drawing (for maximum compatibility)
    doc = ezdxf.new('R2000')
    
    # Create layers
    layers = {
        'VISIBLE': {'color': 7, 'desc': 'White for visible lines'},
        'HIDDEN': {'color': 1, 'linetype': 'DASHED', 'desc': 'Red, dashed for hidden lines'},
        'DIMENSIONS': {'color': 3, 'desc': 'Green for dimensions'},
        'TEXT': {'color': 5, 'desc': 'Blue for text'}
    }
    
    logger.info("\nDXF Layers:")
    for name, attrs in layers.items():
        dxf_attrs = {k: v for k, v in attrs.items() if k != 'desc'}
        # Make sure DASHED linetype exists
        if name == 'HIDDEN':
            if 'DASHED' not in doc.linetypes:
                doc.linetypes.add('DASHED', pattern=[0.5, -0.5])
            dxf_attrs['linetype'] = 'DASHED'
        
        doc.layers.new(name=name, dxfattribs=dxf_attrs)
        logger.info(f"- {name}: {attrs['desc']}")
    
    # Access the modelspace
    msp = doc.modelspace()
    
    # Calculate spacing between views
    spacing = 150.0  # Fixed spacing of 150 units
    
    # Set up view positions
    view_positions = {
        'top': (spacing, spacing * 2, 'TOP VIEW'),
        'front': (spacing, 0, 'FRONT VIEW'),
        'right': (spacing * 2.5, 0, 'RIGHT VIEW')
    }
    
    # Draw each view
    logger.info("\nDrawing views:")
    for view_name, (x, y, title) in view_positions.items():
        view_start = time.time()
        draw_view(msp, normalized_projections[view_name], x, y, title)
        view_time = time.time() - view_start
        logger.info(f"- {title}:")
        logger.info(f"  - Position: ({x:.2f}, {y:.2f})")
        logger.info(f"  - Drawing time: {view_time:.2f} seconds")
    
    # Add dimensions
    dim_start = time.time()
    add_dimensions(msp, mesh, spacing)
    dim_time = time.time() - dim_start
    logger.info(f"\nDimensions added in {dim_time:.2f} seconds")
    
    # Add title block
    add_title_block(msp, os.path.basename(output_path), mesh)
    
    # Save the DXF file
    try:
        save_start = time.time()
        doc.saveas(output_path)
        save_time = time.time() - save_start
        total_time = time.time() - start_time
        
        logger.info(f"\nDXF File Summary:")
        logger.info(f"- Output file: {output_path}")
        logger.info(f"- File size: {os.path.getsize(output_path):,} bytes")
        logger.info(f"- Save time: {save_time:.2f} seconds")
        logger.info(f"- Total processing time: {total_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error saving DXF file: {e}")

def draw_view(msp, edges, x_offset, y_offset, title):
    """
    Draw a single view in the DXF modelspace.
    
    Args:
        msp: DXF modelspace
        edges: 2D edges to draw
        x_offset: X position offset
        y_offset: Y position offset
        title: Title of the view
    """
    # Draw title
    text_height = 3.0
    title_y_offset = -10.0  # Place title below the view
    
    msp.add_text(
        title,
        dxfattribs={
            'height': text_height,
            'layer': 'TEXT',
            'style': 'STANDARD',
            'insert': (x_offset + 50, y_offset + title_y_offset)  # Center the title
        }
    )
    
    # Draw each edge in the view
    for edge in edges:
        # Convert coordinates and add offset
        start = (float(edge[0][0]) + x_offset, float(edge[0][1]) + y_offset)
        end = (float(edge[1][0]) + x_offset, float(edge[1][1]) + y_offset)
        
        # Skip edges with NaN or infinity
        if (np.isnan(start[0]) or np.isnan(start[1]) or 
            np.isnan(end[0]) or np.isnan(end[1]) or
            np.isinf(start[0]) or np.isinf(start[1]) or
            np.isinf(end[0]) or np.isinf(end[1])):
            continue
            
        # Skip edges that are too short (likely noise)
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        if dx*dx + dy*dy < 0.01:  # Increased threshold for short edges
            continue
        
        # Add to DXF as LINE entity
        msp.add_line(start, end, dxfattribs={'layer': 'VISIBLE'})

def add_dimensions(msp, mesh, spacing):
    """
    Add basic dimensions to the DXF drawing.
    
    Args:
        msp: DXF modelspace
        mesh: Original mesh for dimensions
        spacing: Spacing between views
    """
    # Get model dimensions
    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]
    
    # Fixed positions for dimensions
    width = 100.0  # All views normalized to ~100 units
    height = 100.0
    depth = 100.0
    
    # Scale factor to show real dimensions
    text_height = 2.5
    dim_offset = 15.0  # Offset from the drawing
    
    # Add width dimension (x-axis) to front view
    msp.add_linear_dim(
        base=(spacing, -dim_offset),
        p1=(spacing, 0),
        p2=(spacing + width, 0),
        angle=0,
        text=f"{extents[0]:.2f}",
        dimstyle="STANDARD",
        override={'dimtxsty': 'STANDARD', 'dimtxt': text_height},
        dxfattribs={'layer': 'DIMENSIONS'}
    )
    
    # Add height dimension (z-axis) to front view
    msp.add_linear_dim(
        base=(spacing - dim_offset, 0),
        p1=(spacing, 0),
        p2=(spacing, height),
        angle=90,
        text=f"{extents[2]:.2f}",
        dimstyle="STANDARD",
        override={'dimtxsty': 'STANDARD', 'dimtxt': text_height},
        dxfattribs={'layer': 'DIMENSIONS'}
    )
    
    # Add depth dimension (y-axis) to right view
    msp.add_linear_dim(
        base=(spacing * 2.5 - dim_offset, 0),
        p1=(spacing * 2.5, 0),
        p2=(spacing * 2.5, depth),
        angle=90,
        text=f"{extents[1]:.2f}",
        dimstyle="STANDARD",
        override={'dimtxsty': 'STANDARD', 'dimtxt': text_height},
        dxfattribs={'layer': 'DIMENSIONS'}
    )

def add_title_block(msp, filename, mesh):
    """
    Add a simple title block to the drawing.
    
    Args:
        msp: DXF modelspace
        filename: Name of the DXF file
        mesh: Mesh object for additional info
    """
    # Title block position and size
    x, y = 0, 0
    width, height = 400, 20
    
    # Draw rectangle
    msp.add_lwpolyline(
        [(x, y), (x + width, y), (x + width, y + height), (x, y + height), (x, y)],
        dxfattribs={'layer': 'TEXT', 'closed': True}
    )
    
    # Add text
    text_height = 5.0
    msp.add_text(
        f"File: {filename}",
        dxfattribs={
            'height': text_height,
            'layer': 'TEXT',
            'insert': (x + 10, y + height/2)
        }
    )
    
    # Add some model stats
    stats_text = f"Vertices: {len(mesh.vertices):,}  Faces: {len(mesh.faces):,}  Volume: {mesh.volume:.2f}"
    msp.add_text(
        stats_text,
        dxfattribs={
            'height': text_height,
            'layer': 'TEXT',
            'insert': (x + width - 200, y + height/2)
        }
    )

def main():
    """Main function to handle command-line usage."""
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Convert STL to DXF with optimizations for complex models')
    parser.add_argument('input', help='Input STL file path')
    parser.add_argument('--output', '-o', help='Output DXF file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--simplify', '-s', action='store_true', help='Simplify mesh (default: True)')
    parser.add_argument('--target-faces', '-t', type=int, default=5000, help='Target number of faces for simplification')
    parser.add_argument('--no-sections', action='store_true', help='Skip cross-sections')
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Set default output path if not specified
    if not args.output:
        base_name = os.path.splitext(args.input)[0]
        args.output = f"{base_name}.dxf"
    
    logger.info("\nSTL to DXF Converter (Optimized)")
    logger.info("==============================")
    
    # Process the file
    mesh = load_stl(args.input, simplify=args.simplify, target_faces=args.target_faces)
    projections = get_orthographic_projections(mesh, use_sections=not args.no_sections)
    create_dxf(projections, args.output, mesh)
    
    total_time = time.time() - start_time
    logger.info(f"\nTotal conversion time: {total_time:.2f} seconds")
    logger.info("==============================")

if __name__ == "__main__":
    main()