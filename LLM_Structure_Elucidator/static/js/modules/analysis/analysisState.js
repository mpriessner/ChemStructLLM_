// Analysis-specific state management
import { subscribe as globalSubscribe, setError } from '../common/state.js';

// Analysis state
const analysisState = {
    plotData: null,
    plotType: null,
    plotOptions: {},
    moleculeData: null,
    moleculeType: null,
    moleculeOptions: {},
    isAnalyzing: false
};

// State getters
export function getPlotData() {
    return analysisState.plotData;
}

export function getPlotType() {
    return analysisState.plotType;
}

export function getPlotOptions() {
    return { ...analysisState.plotOptions };
}

export function getMoleculeData() {
    return analysisState.moleculeData;
}

export function getMoleculeType() {
    return analysisState.moleculeType;
}

export function getMoleculeOptions() {
    return { ...analysisState.moleculeOptions };
}

export function isAnalyzing() {
    return analysisState.isAnalyzing;
}

// State setters
export function setPlotData(data) {
    console.log('[Analysis State] Setting plot data:', data);
    analysisState.plotData = data;
    notifyStateChange('plotData', data);
}

export function setPlotType(type) {
    console.log('[Analysis State] Setting plot type:', type);
    analysisState.plotType = type;
    notifyStateChange('plotType', type);
}

export function setPlotOptions(options) {
    console.log('[Analysis State] Setting plot options:', options);
    analysisState.plotOptions = { ...options };
    notifyStateChange('plotOptions', analysisState.plotOptions);
}

export function setMoleculeData(data) {
    console.log('[Analysis State] Setting molecule data:', data);
    analysisState.moleculeData = data;
    notifyStateChange('moleculeData', data);
}

export function setMoleculeType(type) {
    console.log('[Analysis State] Setting molecule type:', type);
    analysisState.moleculeType = type;
    notifyStateChange('moleculeType', type);
}

export function setMoleculeOptions(options) {
    console.log('[Analysis State] Setting molecule options:', options);
    analysisState.moleculeOptions = { ...options };
    notifyStateChange('moleculeOptions', analysisState.moleculeOptions);
}

export function setAnalyzing(analyzing) {
    console.log('[Analysis State] Setting analyzing state:', analyzing);
    analysisState.isAnalyzing = analyzing;
    notifyStateChange('analyzing', analyzing);
}

// State change listeners
const listeners = new Map();

// Subscribe to state changes
export function subscribe(key, callback) {
    console.log('[Analysis State] Adding listener for:', key);
    if (!listeners.has(key)) {
        listeners.set(key, new Set());
    }
    listeners.get(key).add(callback);
    
    // Return unsubscribe function
    return () => {
        console.log('[Analysis State] Removing listener for:', key);
        const keyListeners = listeners.get(key);
        if (keyListeners) {
            keyListeners.delete(callback);
        }
    };
}

// Notify listeners of state changes
function notifyStateChange(key, value) {
    console.log('[Analysis State] Notifying state change:', key, value);
    const keyListeners = listeners.get(key);
    if (keyListeners) {
        keyListeners.forEach(callback => {
            try {
                callback(value);
            } catch (error) {
                console.error(`[Analysis State] Error in listener for ${key}:`, error);
                setError(`Error in state listener: ${error.message}`);
            }
        });
    }
}

// Reset analysis state
export function resetAnalysisState() {
    console.log('[Analysis State] Resetting state');
    analysisState.plotData = null;
    analysisState.plotType = null;
    analysisState.plotOptions = {};
    analysisState.moleculeData = null;
    analysisState.moleculeType = null;
    analysisState.moleculeOptions = {};
    analysisState.isAnalyzing = false;
    
    // Notify all listeners
    Object.keys(analysisState).forEach(key => {
        notifyStateChange(key, analysisState[key]);
    });
    console.log('[Analysis State] State reset complete');
}
