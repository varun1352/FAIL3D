/**
 * Dashboard functionality for STL to DXF Converter
 */

document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const fileNameDisplay = document.getElementById('fileNameDisplay');
    const uploadButton = document.getElementById('uploadButton');
    const uploadForm = document.getElementById('uploadForm');

    if (uploadArea && fileInput) {
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
            document.body.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        // Handle drag and drop visual feedback
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, unhighlight, false);
        });

        function highlight(e) {
            uploadArea.classList.add('dragover');
        }

        function unhighlight(e) {
            uploadArea.classList.remove('dragover');
        }

        // Handle dropped files
        uploadArea.addEventListener('drop', handleDrop, false);

        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            handleFiles(files);
        }

        // Handle file selection via input
        fileInput.addEventListener('change', function() {
            handleFiles(this.files);
        });

        // Handle the actual file processing
        function handleFiles(files) {
            if (files.length > 0) {
                const file = files[0];
                if (file.name.toLowerCase().endsWith('.stl')) {
                    fileNameDisplay.innerHTML = `<div class="alert alert-success">Selected: ${file.name}</div>`;
                    uploadButton.disabled = false;
                } else {
                    fileNameDisplay.innerHTML = `<div class="alert alert-danger">Please select an STL file</div>`;
                    uploadButton.disabled = true;
                }
            }
        }

        // Handle form submission
        uploadForm.addEventListener('submit', function(e) {
            if (!fileInput.files || fileInput.files.length === 0) {
                e.preventDefault();
                fileNameDisplay.innerHTML = `<div class="alert alert-danger">Please select a file first</div>`;
            }
        });

        // Make the entire upload area clickable
        uploadArea.addEventListener('click', function(e) {
            if (e.target !== fileInput) {
                fileInput.click();
            }
        });
    }

    // Status badge styling
    const statusBadges = document.querySelectorAll('.status-badge');
    statusBadges.forEach(badge => {
        const status = badge.textContent.trim().toLowerCase();
        badge.classList.add(status);
    });

    // Handle file processing status updates
    function updateFileStatus() {
        const statusElements = document.querySelectorAll('.status-badge');
        
        statusElements.forEach(element => {
            if (element.classList.contains('processing')) {
                const fileId = element.closest('.card').dataset.fileId;
                
                fetch(`/viewer/check-status/${fileId}`)
                    .then(response => response.json())
                    .then(data => {
                        if (data.status === 'complete') {
                            location.reload();
                        } else if (data.status === 'error') {
                            element.textContent = 'Error';
                            element.className = 'status-badge error';
                        }
                    })
                    .catch(error => console.error('Error checking status:', error));
            }
        });
    }

    // Check processing status every 5 seconds
    setInterval(updateFileStatus, 5000);
});