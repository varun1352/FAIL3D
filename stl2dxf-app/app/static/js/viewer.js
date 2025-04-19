// Function to process the file
async function processFile(fileId) {
    try {
        const formData = new FormData();
        formData.append('file_id', fileId);

        const response = await fetch('/process', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Processing failed');
        }

        // Update UI based on response
        updateStatusBadge(data.status);
        if (data.status === 'processed') {
            showDownloadButton();
        }

        return data;
    } catch (error) {
        console.error('Error processing file:', error);
        updateStatusBadge('failed');
        showErrorMessage(error.message);
    }
}

// Function to update the status badge
function updateStatusBadge(status) {
    const statusBadge = document.querySelector('.status-badge');
    if (!statusBadge) return;

    statusBadge.textContent = status.charAt(0).toUpperCase() + status.slice(1);
    statusBadge.className = 'status-badge'; // Reset classes
    
    // Add appropriate color class based on status
    switch (status) {
        case 'processed':
            statusBadge.classList.add('bg-success');
            break;
        case 'failed':
            statusBadge.classList.add('bg-danger');
            break;
        default:
            statusBadge.classList.add('bg-warning');
    }
}

// Function to show the download button
function showDownloadButton() {
    const downloadBtn = document.querySelector('.download-btn');
    if (downloadBtn) {
        downloadBtn.style.display = 'block';
    }
}

// Function to show error message
function showErrorMessage(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'alert alert-danger mt-3';
    errorDiv.textContent = message;
    
    const container = document.querySelector('.container');
    if (container) {
        container.insertBefore(errorDiv, container.firstChild);
        
        // Remove the error message after 5 seconds
        setTimeout(() => {
            errorDiv.remove();
        }, 5000);
    }
}

// Initialize processing when the page loads
document.addEventListener('DOMContentLoaded', () => {
    const fileIdElement = document.getElementById('file-id');
    if (fileIdElement) {
        const fileId = fileIdElement.value;
        if (fileId) {
            processFile(fileId);
        }
    }
}); 