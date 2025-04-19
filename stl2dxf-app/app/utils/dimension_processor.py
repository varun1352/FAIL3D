import cv2
import numpy as np
import ezdxf
from ezdxf.math import Vec3
import logging
import base64
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)

class DimensionProcessor:
    def __init__(self, dxf_path, view_image_data, options=None):
        self.dxf_path = dxf_path
        self.view_image_data = view_image_data
        self.options = options or {
            'overall_dimensions': True,
            'feature_dimensions': True,
            'holes': True
        }
        self.doc = None
        self.msp = None
        self.image = None
        
    def load_dxf(self):
        try:
            self.doc = ezdxf.readfile(self.dxf_path)
            self.msp = self.doc.modelspace()
            return True
        except Exception as e:
            logger.error(f"Failed to load DXF file: {str(e)}")
            return False
            
    def load_image(self):
        try:
            # Remove data URL prefix if present
            if ',' in self.view_image_data:
                self.view_image_data = self.view_image_data.split(',')[1]
            
            # Decode base64 image
            img_data = base64.b64decode(self.view_image_data)
            img = Image.open(BytesIO(img_data))
            self.image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            return True
        except Exception as e:
            logger.error(f"Failed to load view image: {str(e)}")
            return False
            
    def detect_features(self):
        features = {
            'lines': [],
            'circles': [],
            'contours': []
        }
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Line detection
            if self.options.get('overall_dimensions') or self.options.get('feature_dimensions'):
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=10)
                if lines is not None:
                    features['lines'] = lines.reshape(-1, 4)
            
            # Circle detection for holes
            if self.options.get('holes'):
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                         param1=50, param2=30, minRadius=10, maxRadius=100)
                if circles is not None:
                    features['circles'] = circles[0]
            
            # Contour detection
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            features['contours'] = contours
            
            return features
        except Exception as e:
            logger.error(f"Failed to detect features: {str(e)}")
            return None
            
    def add_dimensions_to_dxf(self, features):
        try:
            # Create a new layer for dimensions if it doesn't exist
            if 'DIMENSIONS' not in self.doc.layers:
                self.doc.layers.new('DIMENSIONS', dxfattribs={'color': 1})
                
            # Add overall dimensions
            if self.options.get('overall_dimensions') and features['lines'] is not None:
                self._add_overall_dimensions(features['lines'])
                
            # Add feature dimensions
            if self.options.get('feature_dimensions') and features['lines'] is not None:
                self._add_feature_dimensions(features['lines'])
                
            # Add hole dimensions
            if self.options.get('holes') and features['circles'] is not None:
                self._add_hole_dimensions(features['circles'])
                
            return True
        except Exception as e:
            logger.error(f"Failed to add dimensions to DXF: {str(e)}")
            return False
            
    def _add_overall_dimensions(self, lines):
        # Find bounding box
        x_coords = []
        y_coords = []
        for x1, y1, x2, y2 in lines:
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
            
        if x_coords and y_coords:
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            # Add horizontal dimension
            self.msp.add_linear_dim(
                base=(min_x, min_y - 20),
                p1=(min_x, min_y),
                p2=(max_x, min_y),
                dxfattribs={'layer': 'DIMENSIONS'}
            )
            
            # Add vertical dimension
            self.msp.add_linear_dim(
                base=(max_x + 20, min_y),
                p1=(max_x, min_y),
                p2=(max_x, max_y),
                angle=90,
                dxfattribs={'layer': 'DIMENSIONS'}
            )
            
    def _add_feature_dimensions(self, lines):
        # Add dimensions for significant features
        for x1, y1, x2, y2 in lines:
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length > 50:  # Only dimension significant features
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                self.msp.add_linear_dim(
                    base=(x1, y1 - 10),
                    p1=(x1, y1),
                    p2=(x2, y2),
                    angle=angle,
                    dxfattribs={'layer': 'DIMENSIONS'}
                )
                
    def _add_hole_dimensions(self, circles):
        # Add diameter dimensions for holes
        for x, y, r in circles:
            # Add circle entity
            self.msp.add_circle((x, y), r, dxfattribs={'layer': 'DIMENSIONS'})
            
            # Add diameter dimension
            self.msp.add_diameter_dim(
                center=(x, y),
                radius=r,
                angle=45,
                dxfattribs={'layer': 'DIMENSIONS'}
            )
            
    def save_dxf(self):
        try:
            self.doc.save()
            return True
        except Exception as e:
            logger.error(f"Failed to save DXF file: {str(e)}")
            return False
            
    def process(self):
        """Main processing function that orchestrates the dimension detection and addition."""
        if not self.load_dxf():
            return False
            
        if not self.load_image():
            return False
            
        features = self.detect_features()
        if not features:
            return False
            
        if not self.add_dimensions_to_dxf(features):
            return False
            
        return self.save_dxf() 