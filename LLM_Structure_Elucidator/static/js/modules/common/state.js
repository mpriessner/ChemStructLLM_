// Common state management

// Application state
const state = {
    currentTab: 'structure',  // Current active tab
    isLoading: false,        // Global loading state
    lastError: null,         // Last error message
    activeVisualizations: new Set()  // Track active visualization panels
};

// State getters
export function getCurrentTab() {
    return state.currentTab;
}

export function isLoading() {
    return state.isLoading;
}

export function getLastError() {
    return state.lastError;
}

export function getActiveVisualizations() {
    return Array.from(state.activeVisualizations);
}

// State setters
export function setCurrentTab(tab) {
    state.currentTab = tab;
    notifyStateChange('tab', tab);
}

export function setLoading(loading) {
    state.isLoading = loading;
    notifyStateChange('loading', loading);
}

export function setError(error) {
    state.lastError = error;
    notifyStateChange('error', error);
}

export function addActiveVisualization(panelId) {
    state.activeVisualizations.add(panelId);
    notifyStateChange('visualizations', getActiveVisualizations());
}

export function removeActiveVisualization(panelId) {
    state.activeVisualizations.delete(panelId);
    notifyStateChange('visualizations', getActiveVisualizations());
}

// State change listeners
const listeners = new Map();

export function subscribe(key, callback) {
    if (!listeners.has(key)) {
        listeners.set(key, new Set());
    }
    listeners.get(key).add(callback);
    
    // Return unsubscribe function
    return () => {
        const callbacks = listeners.get(key);
        if (callbacks) {
            callbacks.delete(callback);
        }
    };
}

// Notify listeners of state changes
function notifyStateChange(key, value) {
    const callbacks = listeners.get(key);
    if (callbacks) {
        callbacks.forEach(callback => callback(value));
    }
}

// Reset state
export function resetState() {
    state.currentTab = 'structure';
    state.isLoading = false;
    state.lastError = null;
    state.activeVisualizations.clear();
    
    // Notify all listeners
    ['tab', 'loading', 'error', 'visualizations'].forEach(key => {
        notifyStateChange(key, state[key]);
    });
}
