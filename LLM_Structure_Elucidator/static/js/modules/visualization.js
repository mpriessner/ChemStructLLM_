// Import structure module
import { renderMolecule as renderMoleculeStructure, clearMoleculeVisualizations } from './structure/index.js';

// Visualization state
let lastMoleculeData = null;
let isLoading = false;
let lastError = null;

export function setLastMoleculeData(data) {
    lastMoleculeData = data;
}

export function getLastMoleculeData() {
    return lastMoleculeData;
}

export async function renderMolecule(data) {
    console.log('[Visualization] Starting molecule render:', data);
    isLoading = true;
    
    try {
        if (!data || !data.smiles || !data.image) {
            throw new Error('Invalid molecule data: missing required fields');
        }
        
        // Store the data
        lastMoleculeData = data;
        
        // Render the molecule structure
        console.log('[Visualization] Rendering molecule structure...');
        await renderMoleculeStructure(data);
        console.log('[Visualization] Molecule structure rendered successfully');
        
    } catch (error) {
        console.error('[Visualization] Error rendering molecule:', error);
        lastError = error.message;
        throw error;
    } finally {
        isLoading = false;
    }
}

export function clearAllVisualizations() {
    console.log('[Visualization] Clearing all visualizations');
    isLoading = true;
    
    try {
        // Clear molecule visualizations
        clearMoleculeVisualizations();
        lastMoleculeData = null;

        // Clear NMR plot
        clearNMRPlot();
    } catch (error) {
        console.error('[Visualization] Error clearing visualizations:', error);
        lastError = error.message;
    } finally {
        isLoading = false;
    }
}

// Function to clear only NMR plot
export function clearNMRPlot() {
    console.log('[Visualization] Clearing NMR plot');
    try {
        const plotContainer = document.getElementById('plotly-div');
        if (plotContainer) {
            Plotly.purge(plotContainer);
        }
    } catch (error) {
        console.error('[Visualization] Error clearing NMR plot:', error);
    }
}

// Function to render NMR plot
export function renderNMRPlot(plotData) {
    console.log('[Visualization] Rendering NMR plot:', plotData);
    try {
        const plotContainer = document.getElementById('plotly-div');
        if (!plotContainer) {
            console.error('[Visualization] Plot container not found');
            return;
        }

        if (!plotData.parameters || !plotData.parameters.nmr_data) {
            console.error('[Visualization] Invalid plot data:', plotData);
            return;
        }

        const { x, y } = plotData.parameters.nmr_data;
        const { title, x_label, y_label } = plotData.parameters;

        const trace = {
            x: x,
            y: y,
            type: 'scatter',
            mode: 'lines',
            line: {
                color: '#1f77b4',
                width: 2
            }
        };

        const layout = {
            title: title,
            xaxis: {
                title: x_label,
                autorange: 'reversed'  // Reverse x-axis for NMR convention
            },
            yaxis: {
                title: y_label
            },
            showlegend: false,
            plot_bgcolor: '#ffffff',
            paper_bgcolor: '#ffffff'
        };

        const config = {
            responsive: true,
            displayModeBar: true,
            scrollZoom: true
        };

        Plotly.newPlot(plotContainer, [trace], layout, config);
        console.log('[Visualization] NMR plot rendered successfully');
    } catch (error) {
        console.error('[Visualization] Error rendering NMR plot:', error);
        throw error;
    }
}