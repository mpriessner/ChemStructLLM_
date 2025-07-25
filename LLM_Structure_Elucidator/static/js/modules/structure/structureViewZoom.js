export function initializeZoom(container) {
    const image = container.querySelector('.zoomable-molecule');
    const zoomIndicator = container.querySelector('.zoom-indicator');

    let scale = 1;
    const MIN_SCALE = 0.5;
    const MAX_SCALE = 5;
    const ZOOM_SPEED = 0.1;

    // Variables for panning
    let isPanning = false;
    let startX;
    let startY;
    let translateX = 0;
    let translateY = 0;

    function updateTransform() {
        image.style.transform = `translate(${translateX}px, ${translateY}px) scale(${scale})`;
    }

    function updateZoom(event) {
        event.preventDefault();
        const rect = container.getBoundingClientRect();
        const mouseX = event.clientX - rect.left;
        const mouseY = event.clientY - rect.top;

        // Get current image position relative to container
        const imageRect = image.getBoundingClientRect();
        const imageCenterX = imageRect.left + imageRect.width / 2 - rect.left;
        const imageCenterY = imageRect.top + imageRect.height / 2 - rect.top;

        // Calculate position relative to image center
        const relativeX = (mouseX - imageCenterX) / scale;
        const relativeY = (mouseY - imageCenterY) / scale;

        // Calculate new scale
        const oldScale = scale;
        if (event.deltaY < 0) {
            scale = Math.min(scale * (1 + ZOOM_SPEED), MAX_SCALE);
        } else {
            scale = Math.max(scale * (1 - ZOOM_SPEED), MIN_SCALE);
        }

        // Update zoom indicator
        zoomIndicator.textContent = `Zoom: ${Math.round(scale * 100)}%`;

        // Calculate new translation to keep mouse point fixed
        const newX = mouseX - relativeX * scale;
        const newY = mouseY - relativeY * scale;
        translateX += newX - imageCenterX;
        translateY += newY - imageCenterY;

        updateTransform();
    }

    // Mouse down event - start panning
    container.addEventListener('mousedown', (event) => {
        isPanning = true;
        container.classList.add('grabbing');
        startX = event.clientX - translateX;
        startY = event.clientY - translateY;
    });

    // Mouse move event - update position while panning
    window.addEventListener('mousemove', (event) => {
        if (!isPanning) return;
        
        translateX = event.clientX - startX;
        translateY = event.clientY - startY;
        updateTransform();
    });

    // Mouse up event - stop panning
    window.addEventListener('mouseup', () => {
        isPanning = false;
        container.classList.remove('grabbing');
    });

    // Mouse leave event - stop panning
    container.addEventListener('mouseleave', () => {
        isPanning = false;
        container.classList.remove('grabbing');
    });

    // Prevent default drag behavior
    container.addEventListener('dragstart', (e) => {
        e.preventDefault();
    });

    // Add wheel event listener for zooming
    container.addEventListener('wheel', updateZoom, { passive: false });

    // Center image initially
    function centerImage() {
        const rect = container.getBoundingClientRect();
        const imageRect = image.getBoundingClientRect();
        translateX = (rect.width - imageRect.width) / 2;
        translateY = (rect.height - imageRect.height) / 2;
        updateTransform();
    }

    // Center the image when it loads
    image.addEventListener('load', centerImage);
    centerImage();
}
