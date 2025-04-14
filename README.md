# 3D-to-DXF Converter

A web-based tool that converts 3D models (STL/OBJ) into CAD-compatible 2D drawings (DXF) for mechanical engineers and designers.

## Features

- Upload and process 3D models in STL/OBJ format
- Convert 3D models to 2D DXF drawings
- Support for multiple view projections (front, top, side)
- Automatic dimensioning and annotation
- Modern, responsive web interface

## Tech Stack

- **Backend**: Python/FastAPI
- **Frontend**: React/TypeScript
- **3D Processing**: NumPy, SciPy
- **DXF Generation**: ezdxf

## Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

## Project Structure

```
.
├── backend/           # FastAPI backend
├── frontend/          # React frontend
├── docs/             # Documentation
└── tests/            # Test suite
```

## Getting Started

### Backend Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. Run the development server:
   ```bash
   uvicorn main:app --reload
   ```

### Frontend Setup

1. Install dependencies:
   ```bash
   cd frontend
   npm install
   ```

2. Start the development server:
   ```bash
   npm start
   ```

## Development

- Backend API will be available at `http://localhost:8000`
- Frontend development server will run at `http://localhost:3000`
- API documentation will be available at `http://localhost:8000/docs`

## License

MIT License

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests. 