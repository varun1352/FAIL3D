/**
 * STL Viewer for the STL to DXF Converter application
 */

// Global variables
let scene, camera, renderer, controls, mesh;
let isProcessing = false;

// Initialize Three.js viewer
function initViewer() {
    // Create scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);
    
    // Setup camera
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 5;
    
    // Add lights
    const ambientLight = new THREE.AmbientLight(0x404040, 1.5);
    scene.add(ambientLight);
    
    const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight1.position.set(1, 1, 1);
    scene.add(directionalLight1);
    
    const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight2.position.set(-1, -1, -1);
    scene.add(directionalLight2);
    
    // Setup renderer
    const container = document.getElementById('stlViewer');
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);
    
    // Add orbit controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.25;
    
    // Add grid helper
    const gridHelper = new THREE.GridHelper(10, 10);
    scene.add(gridHelper);
    
    // Add axes helper
    const axesHelper = new THREE.AxesHelper(5);
    scene.add(axesHelper);
    
    // Handle window resize
    window.addEventListener('resize', onWindowResize);
    
    // Initial render
    animate();
    
    // Load the STL model
    loadSTLModel(fileInfo.stlPath);
}

// Load STL model
function loadSTLModel(stlPath) {
    const loader = new THREE.STLLoader();
    
    // Show loading spinner
    const container = document.getElementById('stlViewer');
    const loadingSpinner = document.createElement('div');
    loadingSpinner.className = 'loading-spinner';
    loadingSpinner.innerHTML = '<div class="spinner"></div>';
    container.appendChild(loadingSpinner);
    
    loader.load(
        stlPath,
        function(geometry) {
            // Remove loading spinner
            container.removeChild(loadingSpinner);
            
            // Create mesh
            const material = new THREE.MeshPhongMaterial({
                color: 0x3f51b5,
                specular: 0x111111,
                shininess: 100
            });
            
            mesh = new THREE.Mesh(geometry, material);
            
            // Center and normalize model
            geometry.computeBoundingBox();
            const boundingBox = geometry.boundingBox;
            const center = new THREE.Vector3();
            boundingBox.getCenter(center);
            mesh.position.set(-center.x, -center.y, -center.z);
            
            // Scale model
            const size = boundingBox.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            const scale = 5 / maxDim;
            mesh.scale.set(scale, scale, scale);
            
            // Add to scene
            scene.add(mesh);
            
            // Adjust camera
            controls.reset();
            controls.update();
        },
        function(xhr) {
            // Progress
            const percent = Math.round((xhr.loaded / xhr.total) * 100);
            console.log(percent + '% loaded');
        },
        function(error) {
            // Error
            console.error('Error loading STL:', error);
            
            // Remove loading spinner
            container.removeChild(loadingSpinner);
            
            // Show error message
            const errorMessage = document.createElement('div');
            errorMessage.className = 'error-message';
            errorMessage.innerHTML = 'Error loading STL file. Please try again.';
            container.appendChild(errorMessage);
        }
    );
}

// Handle window resize
function onWindowResize() {
    const container = document.getElementById('stlViewer');
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

// Process STL to DXF
function processSTLToDXF() {
    if (isProcessing) return;
    
    isProcessing = true;
    
    // Update UI
    const processButton = document.getElementById('processButton');
    const processingStatus = document.getElementById('processingStatus');
    const statusBadge = document.getElementById('statusBadge');
    
    if (processButton) {
        processButton.disabled = true;
        processButton.innerHTML = '<i class="fas fa-cog fa-spin"></i> Processing...';
    }
    
    if (processingStatus) {
        processingStatus.style.display = 'block';
    }
    
    if (statusBadge) {
        statusBadge.className = 'status-badge processing';
        statusBadge.textContent = 'Processing';
    }
    
    // Send processing request
    fetch(`/viewer/process/${fileInfo.id}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        console.log('Processing started:', data);
        
        // Start polling for status
        pollProcessingStatus();
    })
    .catch(error => {
        console.error('Error starting processing:', error);
        isProcessing = false;
        
        if (processButton) {
            processButton.disabled = false;
            processButton.innerHTML = 'Process to DXF';
        }
        
        if (processingStatus) {
            processingStatus.style.display = 'none';
        }
        
        if (statusBadge) {
            statusBadge.className = 'status-badge error';
            statusBadge.textContent = 'Error';
        }
        
        alert('Error starting processing. Please try again.');
    });
}

// Poll for processing status
function pollProcessingStatus() {
    if (!isProcessing) return;
    
    fetch(`/viewer/check-status/${fileInfo.id}`)
        .then(response => response.json())
        .then(data => {
            console.log('Processing status:', data);
            
            if (data.status === 'complete') {
                // Processing complete
                isProcessing = false;
                
                // Update UI
                const statusBadge = document.getElementById('statusBadge');
                
                if (statusBadge) {
                    statusBadge.className = 'status-badge complete';
                    statusBadge.textContent = 'Complete';
                }
                
                // Redirect to DXF viewer
                window.location.href = `/viewer/dxf/${fileInfo.id}`;
            } else if (data.status === 'error') {
                // Processing error
                isProcessing = false;
                
                // Update UI
                const processButton = document.getElementById('processButton');
                const processingStatus = document.getElementById('processingStatus');
                const statusBadge = document.getElementById('statusBadge');
                
                if (processButton) {
                    processButton.disabled = false;
                    processButton.innerHTML = 'Process to DXF';
                }
                
                if (processingStatus) {
                    processingStatus.style.display = 'none';
                }
                
                if (statusBadge) {
                    statusBadge.className = 'status-badge error';
                    statusBadge.textContent = 'Error';
                }
                
                // Show error message
                alert(`Error processing file: ${data.error_message || 'Unknown error'}`);
            } else {
                // Still processing, poll again in 2 seconds
                setTimeout(pollProcessingStatus, 2000);
            }
        })
        .catch(error => {
            console.error('Error checking status:', error);
            
            // Try again in 5 seconds
            setTimeout(pollProcessingStatus, 5000);
        });
}

// Initialize viewer when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize 3D viewer
    initViewer();
    
    // Setup process button
    const processButton = document.getElementById('processButton');
    if (processButton) {
        processButton.addEventListener('click', processSTLToDXF);
    }
    
    // Check if already processing
    if (fileInfo.status === 'processing') {
        isProcessing = true;
        pollProcessingStatus();
    }
});