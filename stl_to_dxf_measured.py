#!/usr/bin/env python3
# stl_to_dxf_measured_final.py - Convert STL files to DXF with accurate measurements and professional drawing standards
# Uses the edge processing from the working version with improved measurements
import numpy as np
import trimesh
import ezdxf
import os
import sys
import argparse
import time
import logging
from typing import Tuple, List, Dict, Optional, Set, Any
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_stl(file_path: str, simplify: bool = True, target_faces: int = 2000) -> trimesh.Trimesh:
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

        # Record precise measurements
        bounds = original_mesh.bounds
        extents = bounds[1] - bounds[0]
        logger.info(f"\nPrecise Measurements:")
        logger.info(f"- X dimension (width): {extents[0]:.3f} units")
        logger.info(f"- Y dimension (depth): {extents[1]:.3f} units")
        logger.info(f"- Z dimension (height): {extents[2]:.3f} units")

        # Simplify the mesh if it's complex
        if simplify and len(original_mesh.faces) > target_faces:
            logger.info(f"Simplifying mesh (target: {target_faces:,} faces)...")
            try:
                simplify_start = time.time()
                if hasattr(original_mesh, 'simplify_quadratic_decimation'):
                    mesh = original_mesh.simplify_quadratic_decimation(target_faces)
                    simplify_time = time.time() - simplify_start
                    logger.info(f"- Simplified from {len(original_mesh.faces):,} to {len(mesh.faces):,} faces")
                    logger.info(f"- Simplification time: {simplify_time:.2f} seconds")
                else:
                    logger.warning("Mesh simplification not available, using original mesh")
                    mesh = original_mesh
            except Exception as e:
                logger.warning(f"Mesh simplification failed: {e}")
                mesh = original_mesh
        else:
            mesh = original_mesh

        # Center the mesh on the origin
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
    direction = np.array(direction) / np.linalg.norm(direction)
    dots = np.dot(mesh.face_normals, direction)
    silhouette_edges = []
    edges_seen: Set[Any] = set()

    if len(mesh.faces) > 20000:
        logger.info("Large mesh detected, using simplified edge extraction")
        for i, edge in enumerate(mesh.edges_unique):
            face_indices = mesh.edges_unique_faces[i]
            if len(face_indices) == 2:
                f1, f2 = face_indices
                if dots[f1] * dots[f2] <= 0:
                    v1, v2 = mesh.vertices[edge[0]], mesh.vertices[edge[1]]
                    silhouette_edges.append([v1, v2])
            if time.time() - start_time > max_time:
                logger.warning("Silhouette extraction timeout, returning partial results")
                break
        logger.info(f"Extracted {len(silhouette_edges)} silhouette edges in {time.time()-start_time:.2f} seconds (simplified)")
        return np.array(silhouette_edges)

    has_face_adjacency = hasattr(mesh, 'face_adjacency') and hasattr(mesh, 'face_adjacency_edges')
    if not has_face_adjacency:
        logger.warning("Computing face adjacency data...")
        try:
            _ = mesh.face_adjacency
            _ = mesh.face_adjacency_edges
            has_face_adjacency = True
        except Exception as e:
            logger.error(f"Failed adjacency compute: {e}")
            has_face_adjacency = False

    if has_face_adjacency:
        for i, (f1, f2) in enumerate(mesh.face_adjacency):
            if dots[f1] * dots[f2] <= 0:
                edge = mesh.face_adjacency_edges[i]
                key = tuple(sorted(edge))
                if key not in edges_seen:
                    edges_seen.add(key)
                    v1, v2 = mesh.vertices[edge[0]], mesh.vertices[edge[1]]
                    silhouette_edges.append([v1, v2])
            if time.time() - start_time > max_time:
                logger.warning("Silhouette extraction timeout, returning partial results")
                break
    else:
        logger.warning("Fallback edge extraction")
        for i, edge in enumerate(mesh.edges_unique):
            face_indices = mesh.edges_unique_faces[i]
            if len(face_indices) == 2:
                f1, f2 = face_indices
                if dots[f1] * dots[f2] <= 0:
                    v1, v2 = mesh.vertices[edge[0]], mesh.vertices[edge[1]]
                    silhouette_edges.append([v1, v2])
            if time.time() - start_time > max_time:
                logger.warning("Silhouette extraction timeout, returning partial results")
                break

    logger.info(f"Extracted {len(silhouette_edges)} silhouette edges in {time.time()-start_time:.2f} seconds")
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
        section = mesh.section(plane_origin=origin, plane_normal=normal)
        if section is None or not hasattr(section, 'entities') or len(section.entities) == 0:
            return np.array([])
        edges = []
        for entity in section.entities:
            if hasattr(entity, 'points'):
                pts = entity.points
                for i in range(len(pts)-1):
                    edges.append([pts[i], pts[i+1]])
                if getattr(entity, 'closed', False):
                    edges.append([pts[-1], pts[0]])
        return np.array(edges)
    except Exception as e:
        logger.warning(f"Section extraction failed: {e}")
        return np.array([])


def project_edges_to_2d(edges: np.ndarray, view_matrix: np.ndarray) -> np.ndarray:
    """
    Project 3D edges onto a 2D plane using a view transformation matrix.
    """
    if len(edges) == 0:
        return np.array([])
    proj = []
    for edge in edges:
        p1_h = np.append(edge[0], 1)
        p2_h = np.append(edge[1], 1)
        p1_t = view_matrix.dot(p1_h)
        p2_t = view_matrix.dot(p2_h)
        p1 = p1_t[:2]/p1_t[3] if p1_t[3]!=0 else p1_t[:2]
        p2 = p2_t[:2]/p2_t[3] if p2_t[3]!=0 else p2_t[:2]
        proj.append([p1, p2])
    return np.array(proj)


def clean_edges(edges: np.ndarray, tolerance: float = 1e-6, min_length_factor: float = 0.01) -> np.ndarray:
    """
    Clean up edges by removing duplicates, merging colinear, and filtering short edges.
    """
    if len(edges) == 0:
        return edges
    unique = set()
    for e in edges:
        if np.isnan(e[0]).any() or np.isnan(e[1]).any() or np.isinf(e[0]).any() or np.isinf(e[1]).any():
            continue
        p1 = tuple(np.round(e[0],6)); p2 = tuple(np.round(e[1],6))
        if p1!=p2:
            unique.add((p1,p2) if p1<p2 else (p2,p1))
    arr = np.array([[np.array(u[0]),np.array(u[1])] for u in unique])
    if len(arr)==0:
        return arr
    lengths = np.linalg.norm(arr[:,1]-arr[:,0],axis=1)
    maxl = lengths.max()
    arr = arr[lengths> maxl*min_length_factor]
    if len(arr)==0:
        return arr
    merged=[]; used=set(); start=time.time()
    for i,e1 in enumerate(arr):
        if i in used: continue
        cur=e1.copy(); used.add(i)
        while True:
            found=False
            for j,e2 in enumerate(arr):
                if j in used: continue
                if np.allclose(cur[1], e2[0], atol=tolerance):
                    v1=cur[1]-cur[0]; v2=e2[1]-e2[0]
                    if np.linalg.norm(v1)>0 and np.linalg.norm(v2)>0:
                        if np.allclose(v1/np.linalg.norm(v1),v2/np.linalg.norm(v2), atol=0.01):
                            cur[1]=e2[1]; used.add(j); found=True; break
            if not found or time.time()-start>3: break
        merged.append(cur)
    return np.array(merged)


def normalize_projection(edges: np.ndarray, dimensions: Dict[str, float], scale: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Normalize 2D edges to a standard box using a global scale factor.
    """
    if len(edges)==0:
        return edges, {'scale':scale,'offset_x':0,'offset_y':0,'min_x':0,'min_y':0,'width':dimensions['width'],'height':dimensions['height'],'real_width':dimensions['width'],'real_height':dimensions['height']}
    pts = np.vstack([edges[:,0],edges[:,1]])
    min_x, max_x = pts[:,0].min(), pts[:,0].max()
    min_y, max_y = pts[:,1].min(), pts[:,1].max()
    w, h = max_x-min_x, max_y-min_y
    real_w, real_h = dimensions['width'], dimensions['height']
    norm=[]
    for e in edges:
        p1=[(e[0][0]-min_x)*scale, (e[0][1]-min_y)*scale]
        p2=[(e[1][0]-min_x)*scale, (e[1][1]-min_y)*scale]
        norm.append([p1,p2])
    params={'scale':scale,'offset_x':-min_x,'offset_y':-min_y,'min_x':min_x,'min_y':min_y,'width':w,'height':h,'real_width':real_w,'real_height':real_h}
    return np.array(norm), params


def get_orthographic_projections(mesh: trimesh.Trimesh, use_sections: bool = True, scale: float = 1.0) -> Dict[str, Dict]:
    """
    Generate orthographic projections (front/top/right) with consistent scaling.
    """
    start = time.time(); logger.info("\nGenerating Orthographic Projections:")
    bounds = mesh.bounds; extents = bounds[1]-bounds[0]
    views = {
        'front':{'direction':np.array([0,-1,0]),'matrix':np.array([[1,0,0,0],[0,0,1,0],[0,0,0,0],[0,0,0,1]]),'dimensions':{'width':extents[0],'height':extents[2]}},
        'top':  {'direction':np.array([0,0,1]), 'matrix':np.array([[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,1]]),'dimensions':{'width':extents[0],'height':extents[1]}},
        'right':{'direction':np.array([1,0,0]), 'matrix':np.array([[0,1,0,0],[0,0,1,0],[0,0,0,0],[0,0,0,1]]),'dimensions':{'width':extents[1],'height':extents[2]}}
    }
    projections={}
    for name,data in views.items():
        sil=extract_silhouette_edges(mesh,data['direction'])
        sil2d=project_edges_to_2d(sil,data['matrix'])
        if use_sections:
            # sections can be added here if needed
            pass
        clean=clean_edges(sil2d)
        projections[name]={'edges':clean,'dimensions':data['dimensions']}
        logger.info(f"View {name}: {len(clean)} edges after clean")
    logger.info(f"Total projection time: {time.time()-start:.2f}s")
    return projections


def create_dxf(projections: Dict[str, Dict], output_path: str, mesh: trimesh.Trimesh, drawing_number: str, scale: float) -> None:
    """
    Create a DXF drawing with views, dims, projection lines, and title block.
    """
    start = time.time(); logger.info("\nCreating DXF File:")
    bounds = mesh.bounds; extents = bounds[1]-bounds[0]
    logger.info(f"Model dims - X:{extents[0]:.2f}, Y:{extents[1]:.2f}, Z:{extents[2]:.2f}")
    doc = ezdxf.new('R2000')
    msp = doc.modelspace()
    # layers
    layers = {'VISIBLE':{'color':7}, 'HIDDEN':{'color':1,'linetype':'DASHED'}, 'DIMENSIONS':{'color':3}, 'CENTER':{'color':2,'linetype':'CENTER'}, 'TEXT':{'color':5}}
    for name,attrs in layers.items():
        if name in ['HIDDEN','CENTER']:
            ltype=attrs['linetype']
            if ltype not in doc.linetypes:
                pattern=[0.5,-0.5] if name=='HIDDEN' else [2.0,-0.25,0.25,-0.25]
                doc.linetypes.add(ltype, pattern=pattern)
        doc.layers.new(name=name, dxfattribs={k:v for k,v in attrs.items() if k!='desc'})
    # third-angle view placement
    margin=20.0; front_w=front_h=100.0
    view_positions={'front':(0,0,'FRONT VIEW'),'top':(0,front_h+margin,'TOP VIEW'),'right':(front_w+margin,0,'RIGHT VIEW')}
    # projection lines
    dash={'layer':'HIDDEN'}
    msp.add_line((0,front_h),(0,front_h+margin), dxfattribs=dash)
    msp.add_line((front_w,0),(front_w+margin,0), dxfattribs=dash)
    # draw each view
    for name,(x,y,title) in view_positions.items():
        draw_view_with_measurements(msp, projections[name]['edges'], projections[name]['dimensions'], x, y, title)
    draw_centerlines(msp, front_w+margin)
    add_title_block(msp, os.path.basename(output_path), mesh, drawing_number, scale)
    try:
        doc.saveas(output_path)
        logger.info(f"DXF saved to {output_path} ({os.path.getsize(output_path):,} bytes) in {time.time()-start:.2f}s")
    except Exception as e:
        logger.error(f"Failed to save DXF: {e}")


def draw_view_with_measurements(msp, edges: np.ndarray, transform_params: Dict[str, Any], x_offset: float, y_offset: float, title: str):
    text_h=3.0
    msp.add_text(title, dxfattribs={'height':text_h,'layer':'TEXT','insert':(x_offset+30,y_offset-10)})
    for e in edges:
        s=(float(e[0][0])+x_offset, float(e[0][1])+y_offset)
        t=(float(e[1][0])+x_offset, float(e[1][1])+y_offset)
        if np.isnan(s[0]) or np.isnan(t[0]) or np.isinf(s[0]) or np.isinf(t[0]): continue
        if (t[0]-s[0])**2+(t[1]-s[1])**2<0.01: continue
        msp.add_line(s,t,dxfattribs={'layer':'VISIBLE'})
    add_view_dimensions(msp, transform_params, x_offset, y_offset)


def add_view_dimensions(msp, params: Dict[str, Any], x_offset: float, y_offset: float):
    real_w=params['real_width']; real_h=params['real_height']
    width=100.0; height=100.0; dim_off=15.0; txt_h=2.5
    # horizontal
    msp.add_linear_dim(base=(x_offset,y_offset-dim_off), p1=(x_offset,y_offset), p2=(x_offset+width,y_offset), angle=0,
                       text=f"{real_w:.2f}", dimstyle="STANDARD", override={'dimtxt':txt_h}, dxfattribs={'layer':'DIMENSIONS'})
    # vertical
    msp.add_linear_dim(base=(x_offset-dim_off,y_offset), p1=(x_offset,y_offset), p2=(x_offset,y_offset+height), angle=90,
                       text=f"{real_h:.2f}", dimstyle="STANDARD", override={'dimtxt':txt_h}, dxfattribs={'layer':'DIMENSIONS'})


def draw_centerlines(msp, spacing: float):
    msp.add_line((50,spacing),(spacing-50,spacing),dxfattribs={'layer':'CENTER'})
    msp.add_line((spacing/2,spacing+50),(spacing/2,spacing*2-50),dxfattribs={'layer':'CENTER'})


def add_title_block(msp, filename: str, mesh: trimesh.Trimesh, drawing_number: str, scale: float):
    x,y=0,0; w,h=400,25; txt_h=5.0
    msp.add_lwpolyline([(x,y),(x+w,y),(x+w,y+h),(x,y+h),(x,y)], dxfattribs={'layer':'TEXT','closed':True})
    msp.add_line((x+w*0.7,y),(x+w*0.7,y+h),dxfattribs={'layer':'TEXT'})
    msp.add_line((x+w*0.85,y),(x+w*0.85,y+h),dxfattribs={'layer':'TEXT'})
    today=datetime.now().strftime("%Y-%m-%d")
    msp.add_text(f"Part: {filename}", dxfattribs={'height':txt_h,'layer':'TEXT','insert':(x+10,y+h/2+8)})
    msp.add_text(f"Drawing No: {drawing_number}", dxfattribs={'height':txt_h,'layer':'TEXT','insert':(x+10,y+h/2)})
    msp.add_text(f"Date: {today}", dxfattribs={'height':txt_h,'layer':'TEXT','insert':(x+w*0.5,y+h/2)})
    scale_txt=f"Scale: 1:{1/scale:.1f}" if scale!=0 else "Scale: 1:1"
    msp.add_text(scale_txt, dxfattribs={'height':txt_h,'layer':'TEXT','insert':(x+w*0.75,y+h/2)})
    # third-angle symbol
    cx,cy=x+w*0.9,y+h*0.3
    msp.add_circle((cx,cy),radius=5.0,dxfattribs={'layer':'TEXT'})
    msp.add_circle((cx+6,cy),radius=1.5,dxfattribs={'layer':'TEXT'})
    msp.add_lwpolyline([(cx,cy+5),(cx,cy-5),(cx+12,cy)],dxfattribs={'layer':'TEXT'})


def main():
    start=time.time()
    parser=argparse.ArgumentParser(description='Convert STL to DXF with accurate measurements')
    parser.add_argument('input', help='Input STL file path')
    parser.add_argument('--output','-o', help='Output DXF file path')
    parser.add_argument('--verbose','-v',action='store_true',help='Enable verbose output')
    parser.add_argument('--simplify','-s',action='store_true',help='Simplify mesh')
    parser.add_argument('--target-faces','-t',type=int,default=2000,help='Target number of faces')
    parser.add_argument('--no-sections',action='store_true',help='Skip cross-sections')
    parser.add_argument('--drawing-number','-d',default='0001',help='Drawing number')
    args=parser.parse_args()
    if args.verbose: logger.setLevel(logging.DEBUG)
    if not args.output:
        base=os.path.splitext(args.input)[0]
        args.output=f"{base}_measured.dxf"

    logger.info("\nSTL to DXF Converter with Enhanced Measurements and Standards")
    mesh=load_stl(args.input, simplify=args.simplify, target_faces=args.target_faces)
    extents=mesh.bounds[1]-mesh.bounds[0]
    largest=max(extents)
    drawing_target=100.0
    global_scale=drawing_target/largest if largest>0 else 1.0
    projections=get_orthographic_projections(mesh, use_sections=not args.no_sections, scale=global_scale)
    create_dxf(projections, args.output, mesh, args.drawing_number, global_scale)
    logger.info(f"\nTotal conversion time: {time.time()-start:.2f} seconds")

if __name__ == "__main__":
    main()
