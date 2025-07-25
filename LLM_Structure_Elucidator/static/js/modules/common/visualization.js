// Common visualization utilities

// Function to clear placeholder content
export function clearPlaceholder(containerId) {
    console.log('[Visualization] Clearing placeholder for container:', containerId);
    const container = typeof containerId === 'string' ? document.getElementById(containerId) : containerId;
    if (!container) {
        console.error('[Visualization] Container not found:', containerId);
        return;
    }
    const placeholder = container.querySelector('.content-placeholder');
    if (placeholder) {
        placeholder.remove();
        console.log('[Visualization] Removed placeholder');
    }
}

// Function to show loading state
export function showLoadingState(containerId, message = 'Loading...') {
    console.log('[Visualization] Showing loading state for container:', containerId);
    const container = typeof containerId === 'string' ? document.getElementById(containerId) : containerId;
    if (!container) {
        console.error('[Visualization] Container not found:', containerId);
        return;
    }
    
    // Remove any existing loading state
    hideLoadingState(container);
    
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading-state';
    loadingDiv.innerHTML = `
        <div class="spinner"></div>
        <p>${message}</p>
    `;
    container.appendChild(loadingDiv);
    console.log('[Visualization] Added loading state');
}

// Function to hide loading state
export function hideLoadingState(containerId) {
    console.log('[Visualization] Hiding loading state for container:', containerId);
    const container = typeof containerId === 'string' ? document.getElementById(containerId) : containerId;
    if (!container) {
        console.error('[Visualization] Container not found:', containerId);
        return;
    }
    const loadingState = container.querySelector('.loading-state');
    if (loadingState) {
        loadingState.remove();
        console.log('[Visualization] Removed loading state');
    }
}

// Function to create a visualization panel
export function createVisualizationPanel(containerId) {
    console.log('[Visualization] Creating/verifying panel for container:', containerId);
    const container = typeof containerId === 'string' ? document.getElementById(containerId) : containerId;
    if (!container) {
        console.error('[Visualization] Container not found:', containerId);
        return null;
    }

    // Clear any existing content
    container.innerHTML = '';
    console.log('[Visualization] Cleared container content');

    // Add visualization panel class if not present
    if (!container.classList.contains('visualization-panel')) {
        container.classList.add('visualization-panel');
    }

    return container;
}

// Function to show error message in a container
export function showError(containerId, message) {
    console.log('[Visualization] Showing error in container:', containerId);
    const container = typeof containerId === 'string' ? document.getElementById(containerId) : containerId;
    if (!container) {
        console.error('[Visualization] Container not found:', containerId);
        return;
    }

    // Clear existing content
    container.innerHTML = '';

    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.innerHTML = `
        <div class="error-icon">⚠️</div>
        <p>${message}</p>
    `;
    container.appendChild(errorDiv);
    console.log('[Visualization] Added error message:', message);
}

// Function to clear a container
export function clearContainer(containerId) {
    console.log('[Visualization] Clearing container:', containerId);
    const container = typeof containerId === 'string' ? document.getElementById(containerId) : containerId;
    if (!container) {
        console.error('[Visualization] Container not found:', containerId);
        return;
    }
    container.innerHTML = '';
    console.log('[Visualization] Container cleared');
}