let videoStream = null;
let statsInterval = null;
let retryCount = 0;
const MAX_RETRIES = 3;
let isManuallyStopped = false;


// DOM Elements
const videoElement = document.getElementById('video-stream');
const errorElement = document.getElementById('video-error');
const loadingElement = document.getElementById('loading-indicator');
const startButton = document.querySelector('.start-camera');
const stopButton = document.querySelector('.stop-camera');
const statusDot = document.querySelector('.status-dot');

function showError() {
    if (errorElement) errorElement.style.display = 'flex';
    if (loadingElement) loadingElement.style.display = 'none';
}

function showLoading() {
    if (errorElement) errorElement.style.display = 'none';
    if (loadingElement) loadingElement.style.display = 'flex';
}

function hideLoadingAndError() {
    if (errorElement) errorElement.style.display = 'none';
    if (loadingElement) loadingElement.style.display = 'none';
}

function handleVideoError(event) {

    console.error('Video stream error:', event);
    if (isManuallyStopped) {
        console.log('Video error ignored because camera was manually stopped.');
        return;
    }
    if (retryCount < MAX_RETRIES) {
        retryCount++;
        console.log(`Retrying video stream (${retryCount}/${MAX_RETRIES})...`);
        setTimeout(startCamera, 1000);
    } else {
        showError();
        stopCamera();
    }
}

function handleVideoLoad() {
    console.log('Video stream loaded successfully');
    hideLoadingAndError();
    retryCount = 0;
}

function startCamera() {
    if (!videoStream) {
        showLoading();
        isManuallyStopped = false; // reset flag on start   
        // Add timestamp to prevent caching
        const timestamp = new Date().getTime();
        if (videoElement) {
            videoElement.src = `/speed_estimation/video_feed/?t=${timestamp}`;
            videoStream = true;
        }

        // Update UI
        if (startButton) startButton.disabled = true;
        if (stopButton) stopButton.disabled = false;
        if (statusDot) statusDot.style.backgroundColor = 'var(--success-color)';

        // Start stats updates
        if (!statsInterval) {
            statsInterval = setInterval(updateStats, 2000);
        }
    }
}

function stopCamera() {
    if (videoStream) {
        isManuallyStopped = true; // indicate manual stop
        if (videoElement) {
            videoElement.src = '';
            videoElement.removeAttribute('src');
            videoElement.load(); // Optional cleanup
        }
        videoStream = false;

        // Update UI
        if (startButton) startButton.disabled = false;
        if (stopButton) stopButton.disabled = true;
        if (statusDot) statusDot.style.backgroundColor = 'var(--danger-color)';

        // Reset stats and clear interval
        const vehicleCount = document.getElementById('vehicle-count');
        const currentSpeed = document.getElementById('current-speed');
        if (vehicleCount) vehicleCount.textContent = '0';
        if (currentSpeed) currentSpeed.textContent = '0';

        if (statsInterval) {
            clearInterval(statsInterval);
            statsInterval = null;
        }

        hideLoadingAndError();
        retryCount = 0;
    }
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', function () {
    console.log('Initializing video stream components...');

    // Initialize button states
    if (stopButton) stopButton.disabled = true;
    if (statusDot) statusDot.style.backgroundColor = 'var(--danger-color)';
    hideLoadingAndError();

    // Add event listeners
    if (videoElement) {
        videoElement.addEventListener('error', handleVideoError);
        videoElement.addEventListener('load', handleVideoLoad);
    }

    if (startButton) {
        startButton.addEventListener('click', startCamera);
    }

    if (stopButton) {
        stopButton.addEventListener('click', stopCamera);
    }
});

// Get CSRF token
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

// Update stats periodically
async function updateStats() {
    if (!videoStream) return;

    try {
        const response = await fetch('/speed_estimation/get_stats/', {
            method: 'POST',
            headers: {
                'X-CSRFToken': getCookie('csrftoken'),
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({})
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }

        const vehicleCount = document.getElementById('vehicle-count');
        const currentSpeed = document.getElementById('current-speed');
        if (vehicleCount) vehicleCount.textContent = data.vehicle_count;
        if (currentSpeed) currentSpeed.textContent = data.current_speed;
    } catch (error) {
        console.error('Error updating stats:', error);
        if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
            stopCamera();
        }
    }
}
