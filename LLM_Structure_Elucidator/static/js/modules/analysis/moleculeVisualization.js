// Molecule visualization module
import { clearPlaceholder, showLoadingState, hideLoadingState, createVisualizationPanel, showError } from '../common/visualization.js';
import { setMoleculeData, setMoleculeType, setMoleculeOptions, setAnalyzing } from './analysisState.js';

/**
 * Initialize a 3Dmol viewer in a container
 * @param {string} containerId - Container ID
 * @param {Object} options - Viewer options
 * @returns {Object} 3Dmol viewer instance
 */
function initialize3DmolViewer(containerId, options = {}) {
    console.log('[Molecule Visualization] Initializing 3Dmol viewer:', containerId);
    
    const container = document.getElementById(containerId);
    if (!container) {
        throw new Error(`Container ${containerId} not found`);
    }
    
    // Create a div for the 3Dmol viewer
    const viewerDiv = document.createElement('div');
    viewerDiv.style.width = '100%';
    viewerDiv.style.height = '100%';
    viewerDiv.style.position = 'relative';
    container.appendChild(viewerDiv);
    
    // Initialize the viewer
    const config = {
        backgroundColor: 'white',
        ...options
    };
    
    const viewer = $3Dmol.createViewer(viewerDiv, config);
    console.log('[Molecule Visualization] 3Dmol viewer created');
    return viewer;
}

/**
 * Render a molecule visualization in the specified container
 * @param {Object} data - Molecule data containing SMILES and visualization info
 * @param {Object} options - Additional visualization options
 */
export async function renderMolecule(data, options = {}) {
    console.log('[Molecule Visualization] Starting molecule render:', { data, options });
    
    try {
        // Use specified container or default to molecule tab
        const containerId = data.container || 'vis-content-1';
        console.log('[Molecule Visualization] Using container:', containerId);
        
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`[Molecule Visualization] Container ${containerId} not found`);
            throw new Error(`Molecule container ${containerId} not found`);
        }

        showLoadingState(containerId);
        console.log('[Molecule Visualization] Showing loading state');

        // Validate molecule data
        if (!data || !data.smiles) {
            console.error('[Molecule Visualization] Invalid molecule data:', data);
            throw new Error('Invalid molecule data structure');
        }

        // Create visualization panel if needed
        createVisualizationPanel(containerId);
        console.log('[Molecule Visualization] Created/verified visualization panel');

        // Clear existing content
        clearMolecule(containerId);
        console.log('[Molecule Visualization] Cleared existing content');

        if (data.type === '3d') {
            console.log('[Molecule Visualization] Rendering 3D molecule');
            
            // Use vis-content-3 for 3D visualization
            const container = document.getElementById('vis-content-3');
            if (!container) {
                console.error('[Molecule Visualization] Container vis-content-3 not found');
                throw new Error('3D molecule container not found');
            }

            try {
                // Initialize 3Dmol viewer
                const viewerDiv = document.createElement('div');
                viewerDiv.style.width = '100%';
                viewerDiv.style.height = '400px';
                viewerDiv.style.position = 'relative';
                container.appendChild(viewerDiv);
                
                const viewer = $3Dmol.createViewer(
                    $(viewerDiv),
                    {
                        backgroundColor: 'white',
                        defaultcolors: $3Dmol.rasmolElementColors
                    }
                );
                
                if (!viewer) {
                    throw new Error('Failed to create 3Dmol viewer');
                }
                
                // Convert SMILES to PDB using NCI service
                console.log('[Molecule Visualization] Converting SMILES to PDB');
                const url = `https://cactus.nci.nih.gov/chemical/structure/${encodeURIComponent(data.smiles)}/file?format=pdb`;
                
                const response = await fetch(url);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const pdbData = await response.text();
                if (!pdbData || (!pdbData.includes('ATOM') && !pdbData.includes('HETATM'))) {
                    throw new Error('Invalid PDB data received');
                }
                
                // Add the molecule model and set visualization style
                viewer.addModel(pdbData, "pdb");
                viewer.setStyle({}, {stick:{}});
                viewer.zoomTo();
                viewer.render();
                viewer.animate({loop: "forward", reps: 0});
                
                console.log('[Molecule Visualization] 3D molecule rendered');
                
                // Add molecule information div below 3D viewer
                const infoDiv = document.createElement('div');
                infoDiv.className = 'molecule-info';
                infoDiv.style.marginTop = '10px';
                infoDiv.style.textAlign = 'left';
                infoDiv.style.padding = '10px';
                infoDiv.style.backgroundColor = '#f5f5f5';
                infoDiv.style.borderRadius = '5px';
                
                // Add SMILES information
                const smilesDiv = document.createElement('div');
                smilesDiv.style.marginBottom = '5px';
                smilesDiv.innerHTML = `<strong>SMILES:</strong> <span style="font-family: monospace;">${data.smiles}</span>`;
                infoDiv.appendChild(smilesDiv);
                
                // Add molecular weight information
                if (data.molecular_weight) {
                    const mwDiv = document.createElement('div');
                    mwDiv.innerHTML = `<strong>Molecular Weight:</strong> ${data.molecular_weight}`;
                    infoDiv.appendChild(mwDiv);
                }
                
                container.appendChild(infoDiv);
                
            } catch (error) {
                console.error('[Molecule Visualization] Error in 3D rendering:', error);
                const errorDiv = document.createElement('div');
                errorDiv.className = 'error-message';
                errorDiv.style.color = 'red';
                errorDiv.style.padding = '10px';
                errorDiv.innerHTML = `Error loading 3D structure: ${error.message}`;
                container.appendChild(errorDiv);
                throw error;
            }
            
        } else {
            console.log('[Molecule Visualization] Rendering 2D molecule');
            // Create image element for 2D view
            const img = document.createElement('img');
            img.src = `data:${data.format || 'image/png'};${data.encoding || 'base64'},${data.image}`;
            img.style.maxWidth = '100%';
            img.style.height = 'auto';
            img.alt = 'Molecule Structure';
            img.dataset.smiles = data.smiles;
            
            // Add image to container
            container.appendChild(img);
            
            // Add molecule information div
            const infoDiv = document.createElement('div');
            infoDiv.className = 'molecule-info';
            infoDiv.style.marginTop = '10px';
            infoDiv.style.textAlign = 'left';
            infoDiv.style.padding = '10px';
            infoDiv.style.backgroundColor = '#f5f5f5';
            infoDiv.style.borderRadius = '5px';
            
            // Add SMILES information
            const smilesDiv = document.createElement('div');
            smilesDiv.style.marginBottom = '5px';
            smilesDiv.innerHTML = `<strong>SMILES:</strong> <span style="font-family: monospace;">${data.smiles}</span>`;
            infoDiv.appendChild(smilesDiv);
            
            // Add molecular weight information
            if (data.molecular_weight) {
                const mwDiv = document.createElement('div');
                mwDiv.innerHTML = `<strong>Molecular Weight:</strong> ${data.molecular_weight}`;
                infoDiv.appendChild(mwDiv);
            }
            
            container.appendChild(infoDiv);
            console.log('[Molecule Visualization] 2D molecule rendered');
        }

        // Update state
        setMoleculeData(data);
        setMoleculeType(data.type || 'unknown');
        setMoleculeOptions(options);
        setAnalyzing(false);
        
        hideLoadingState(containerId);
        console.log('[Molecule Visualization] Molecule rendering complete');
        
    } catch (error) {
        console.error('[Molecule Visualization] Error rendering molecule:', error);
        const containerId = data?.container || 'vis-content-1';
        showError(containerId, `Failed to render molecule: ${error.message}`);
        hideLoadingState(containerId);
        throw error;
    }
}

/**
 * Clear the molecule visualization
 * @param {string} containerId - Target container ID
 */
export function clearMolecule(containerId = 'vis-content-1') {
    console.log('[Molecule Visualization] Clearing molecule visualization');
    try {
        const container = document.getElementById(containerId);
        if (container) {
            // Remove any existing 3Dmol viewers
            const viewers = container.querySelectorAll('div');
            viewers.forEach(viewer => {
                if (viewer._viewer) {
                    viewer._viewer.clear();
                }
            });
            
            container.innerHTML = '';
            console.log('[Molecule Visualization] Cleared molecule container');
        }
    } catch (error) {
        console.error('[Molecule Visualization] Error clearing molecule:', error);
    }
}
