// Plot visualization module
import { clearPlaceholder, showLoadingState, hideLoadingState, createVisualizationPanel, showError } from '../common/visualization.js';
import { setPlotData, setPlotType, setPlotOptions, setAnalyzing, getPlotData } from './analysisState.js';

// Initialize Plotly with default configuration
const defaultPlotConfig = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['sendDataToCloud', 'lasso2d', 'select2d'],
    useResizeHandler: true
};

/**
 * Parse plot data safely
 * @param {Object|string} data - Raw plot data
 * @returns {Object} Parsed plot data
 */
function parsePlotData(data) {
    console.log('[Plot Visualization] Parsing plot data:', data);
    
    if (!data) {
        console.error('[Plot Visualization] No plot data provided');
        throw new Error('No plot data provided');
    }

    let plotData;
    try {
        if (typeof data === 'string') {
            console.log('[Plot Visualization] Parsing string data');
            plotData = JSON.parse(data);
        } else if (data.data && Array.isArray(data.data)) {
            console.log('[Plot Visualization] Using direct plot data');
            plotData = data;
        } else if (data.plot) {
            console.log('[Plot Visualization] Extracting plot from data');
            plotData = typeof data.plot === 'string' ? JSON.parse(data.plot) : data.plot;
        } else {
            console.error('[Plot Visualization] Invalid data format:', data);
            throw new Error('Invalid plot data format');
        }

        if (!plotData || !plotData.data || !Array.isArray(plotData.data)) {
            console.error('[Plot Visualization] Invalid plot structure:', plotData);
            throw new Error('Invalid plot data structure');
        }

        // Add layout if not present
        if (!plotData.layout) {
            console.log('[Plot Visualization] Adding default layout');
            plotData.layout = {};
        }

        // Add default config
        plotData.config = { ...defaultPlotConfig, ...(plotData.config || {}) };
        console.log('[Plot Visualization] Final plot data:', plotData);
        return plotData;
    } catch (error) {
        console.error('[Plot Visualization] Error parsing plot data:', error);
        throw error;
    }
}

/**
 * Render a plot in the specified container
 * @param {Object} data - Plot data object containing plot property with Plotly data
 * @param {string} containerId - Target container ID
 * @param {Object} options - Additional plot options
 */
export async function renderPlot(data, containerId = 'vis-content-4', options = {}) {
    console.log('[Plot Visualization] Starting plot render:', { data, containerId, options });
    
    try {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`[Plot Visualization] Container ${containerId} not found`);
            throw new Error(`Plot container ${containerId} not found`);
        }

        showLoadingState(containerId);
        console.log('[Plot Visualization] Showing loading state');

        // Parse and validate plot data
        const plotData = parsePlotData(data);
        console.log('[Plot Visualization] Parsed plot data:', plotData);

        // Create visualization panel if needed
        createVisualizationPanel(containerId);
        console.log('[Plot Visualization] Created/verified visualization panel');

        // Clear any existing plot
        clearPlot(containerId);
        console.log('[Plot Visualization] Cleared existing plot');

        // Add default layout properties
        const layout = {
            autosize: true,
            margin: { l: 50, r: 50, b: 50, t: 50, pad: 4 },
            ...plotData.layout
        };
        console.log('[Plot Visualization] Final layout:', layout);

        // Create plot
        await Plotly.newPlot(container, plotData.data, layout, plotData.config);
        console.log('[Plot Visualization] Plot created');

        // Update state
        setPlotData(plotData);
        setPlotType(plotData.type || 'unknown');
        setPlotOptions(options);
        setAnalyzing(false);
        
        hideLoadingState(containerId);
        console.log('[Plot Visualization] Plot rendering complete');
        
    } catch (error) {
        console.error('[Plot Visualization] Error rendering plot:', error);
        showError(containerId, `Failed to render plot: ${error.message}`);
        hideLoadingState(containerId);
        throw error;
    }
}

/**
 * Update an existing plot
 * @param {Object} data - New plot data
 * @param {string} containerId - Target container ID
 */
export function updatePlot(data, containerId = 'vis-content-4') {
    console.log('[Plot Visualization] Updating plot:', { data, containerId });
    
    try {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`[Plot Visualization] Container ${containerId} not found`);
            throw new Error(`Plot container ${containerId} not found`);
        }

        const plotData = parsePlotData(data);
        Plotly.update(container, plotData.data, plotData.layout);
        console.log('[Plot Visualization] Plot updated successfully');
        
        // Update state
        setPlotData(plotData);
        
    } catch (error) {
        console.error('[Plot Visualization] Error updating plot:', error);
        showError(containerId, `Failed to update plot: ${error.message}`);
        throw error;
    }
}

/**
 * Clear the plot visualization
 * @param {string} containerId - Target container ID
 */
export function clearPlot(containerId = 'vis-content-4') {
    console.log('[Plot Visualization] Clearing plot');
    try {
        const container = document.getElementById(containerId);
        if (container) {
            Plotly.purge(container);
            console.log('[Plot Visualization] Cleared plot container');
        }
    } catch (error) {
        console.error('[Plot Visualization] Error clearing plot:', error);
    }
}