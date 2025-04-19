# DimensionForge - Product Requirements Document

## Overview
DimensionForge is a web-based tool for converting STL files to DXF format with interactive 3D and 2D viewing capabilities. The application provides a streamlined workflow for engineers and designers to process their 3D models and obtain 2D technical drawings.

## Core Features

### 1. User Authentication
- Simple dummy login system
- No database integration required
- Session management for user state
- Logout functionality

### 2. Dashboard
#### Requirements
- Clean, technical interface inspired by CAD software
- List of processed files with details
- Upload new file button
- Navigation to viewer pages
- File management interface

#### Display Information
- File name
- Processing status
- Creation date
- File size
- Preview thumbnail
- Actions (View/Download)

### 3. File Upload
#### Requirements
- Drag-and-drop interface
- File type validation (STL only)
- Progress indicator
- Error handling
- Size limit warnings
- Automatic redirect to STL viewer

### 4. STL Viewer
#### Requirements
- Interactive 3D viewer using Three.js
- Standard CAD-like controls:
  - Orbit rotation
  - Pan
  - Zoom
  - Reset view
- Processing trigger button
- Loading indicator during processing
- Error handling and user feedback

#### Viewer Controls
- Mouse controls:
  - Left click + drag: Rotate
  - Right click + drag: Pan
  - Scroll: Zoom
- View buttons:
  - Top view
  - Front view
  - Side view
  - Isometric view

### 5. Processing System
#### Requirements
- Background processing of STL to DXF conversion
- Progress indication
- Error handling
- Automatic cleanup of temporary files
- Success/failure notifications

### 6. DXF Viewer
#### Requirements
- 2D technical drawing display
- Zoom and pan controls
- Download button
- Return to dashboard option
- Print functionality

#### Viewer Features
- Zoom controls
- Pan navigation
- Fit to screen option
- Layer visibility toggle
- Download in original quality

## Technical Requirements

### Performance
- Page load time < 2 seconds
- File upload handling up to 50MB
- Processing time feedback
- Responsive UI across desktop browsers

### Browser Support
- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

### File Management
- Automatic cleanup of uploaded files
- Organized storage structure:
  - /uploads for STL files
  - /output for DXF files
- Unique file naming convention

## User Interface

### Design Philosophy
- Clean, professional appearance
- CAD-like interface familiarity
- High contrast for technical details
- Clear visual hierarchy
- Responsive layout
- Consistent color scheme:
  - Primary: #2C3E50 (Dark Blue)
  - Secondary: #3498DB (Light Blue)
  - Accent: #E74C3C (Red)
  - Background: #ECF0F1 (Light Gray)

### Navigation
- Clear breadcrumb trail
- Persistent navigation menu
- Context-aware actions
- Keyboard shortcuts for common actions

## Error Handling
- Clear error messages
- User-friendly error descriptions
- Recovery suggestions
- Automatic error reporting
- Graceful fallbacks

## Success Metrics
- Successful file conversions
- User session duration
- Error rate
- Processing time
- User interaction flow completion

## Future Considerations
- Database integration
- User accounts and authentication
- File version history
- Batch processing
- Advanced measurement tools
- Collaboration features 