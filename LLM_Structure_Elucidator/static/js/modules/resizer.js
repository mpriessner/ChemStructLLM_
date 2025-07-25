export function initializeResizer() {
    const container = document.querySelector('.container');
    const chatPanel = document.querySelector('.chat-panel');
    const handle = document.querySelector('.resize-handle');
    const visualizationGrid = document.querySelector('.visualization-grid');
    let isResizing = false;

    // Initialize handle position
    updateHandlePosition();

    handle.addEventListener('mousedown', startResizing);

    function startResizing(e) {
        isResizing = true;
        handle.classList.add('active');
        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('mouseup', stopResizing);
        document.body.style.userSelect = 'none';
        e.preventDefault(); // Prevent text selection
    }

    function handleMouseMove(e) {
        if (!isResizing) return;

        const containerRect = container.getBoundingClientRect();
        let newWidth = (e.clientX / containerRect.width) * 100;

        // Constrain width between 20% and 80%
        newWidth = Math.min(Math.max(newWidth, 20), 80);
        
        // Update elements
        chatPanel.style.width = `${newWidth}%`;
        handle.style.left = `${newWidth}%`;
        visualizationGrid.style.width = `${100 - newWidth}%`;

        // Trigger window resize event for any plots
        window.dispatchEvent(new Event('resize'));
    }

    function stopResizing() {
        if (!isResizing) return;
        
        isResizing = false;
        handle.classList.remove('active');
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', stopResizing);
        document.body.style.userSelect = '';
    }

    function updateHandlePosition() {
        const width = chatPanel.getBoundingClientRect().width;
        const containerWidth = container.getBoundingClientRect().width;
        const widthPercent = (width / containerWidth) * 100;
        handle.style.left = `${widthPercent}%`;
        visualizationGrid.style.width = `${100 - widthPercent}%`;
    }

    // Update on window resize
    window.addEventListener('resize', updateHandlePosition);
}
