// Structure module entry point
import { renderMolecule2D } from './molecule2D.js';

// Module state
let lastMoleculeData = null;

export function setMoleculeData(data) {
    lastMoleculeData = data;
}

export function getMoleculeData() {
    return lastMoleculeData;
}

/**
 * Render a molecule in both 2D and 3D views
 * @param {Object} data Molecule data object containing SMILES and image
 */
export async function renderMolecule(data) {
    console.log('Structure module: Rendering molecule with data:', data);
    
    try {
        if (!data || !data.smiles) {
            throw new Error('Invalid molecule data');
        }
        
        // Store the data
        setMoleculeData(data);
        
        // Render 2D molecule
        await renderMolecule2D(data);
        
        // Initialize 3D viewer container
        initializeMolecule3DContainer();
        
        // Render 3D molecule
        await renderMolecule3D(data.smiles);
        
    } catch (error) {
        console.error('Error in structure module:', error);
        throw error;
    }
}

/**
 * Initialize the 3D molecule container
 */
function initializeMolecule3DContainer() {
    // Create container for box 3 if it doesn't exist
    let container = document.getElementById('vis-container-3');
    if (!container) {
        container = document.createElement('div');
        container.id = 'vis-container-3';
        container.className = 'visualization-container';
        const grid = document.querySelector('.visualization-grid');
        if (!grid) {
            throw new Error('Visualization grid not found');
        }
        grid.appendChild(container);
    }

    // Create content div if it doesn't exist
    let content = document.getElementById('vis-content-3');
    if (!content) {
        content = document.createElement('div');
        content.id = 'vis-content-3';
        content.className = 'visualization-content';
        container.appendChild(content);
    }

    // Clear existing content
    content.innerHTML = '';

    // Create 3D molecule panel
    let molecule3DPanel = document.createElement('div');
    molecule3DPanel.id = 'molecule-3d-panel';
    molecule3DPanel.className = 'visualization-panel active';
    content.appendChild(molecule3DPanel);

    // Create 3D container
    let container3D = document.createElement('div');
    container3D.id = 'container-3d';
    container3D.style.width = '100%';
    container3D.style.height = '100%';
    container3D.style.position = 'relative';
    molecule3DPanel.appendChild(container3D);
}

/**
 * Render a 3D molecule view
 */
async function renderMolecule3D(smiles) {
    console.log('Creating 3Dmol viewer');
    
    const container3D = document.getElementById('container-3d');
    if (!container3D) {
        throw new Error('3D container not found');
    }
    
    try {
        let viewer = $3Dmol.createViewer(
            $('#container-3d'),
            {
                backgroundColor: 'white',
                defaultcolors: $3Dmol.rasmolElementColors
            }
        );
        
        if (!viewer) {
            throw new Error('Failed to create 3Dmol viewer');
        }
        
        console.log('Fetching PDB data for SMILES:', smiles);
        
        let url = `https://cactus.nci.nih.gov/chemical/structure/${encodeURIComponent(smiles)}/file?format=pdb`;
        
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const pdbData = await response.text();
        if (!pdbData || (!pdbData.includes('ATOM') && !pdbData.includes('HETATM'))) {
            throw new Error('Invalid PDB data received');
        }
        
        viewer.addModel(pdbData, "pdb");
        viewer.setStyle({}, {stick:{}});
        viewer.zoomTo();
        viewer.render();
        viewer.animate({loop: "forward",reps: 0});
        
    } catch (error) {
        console.error('Error in 3D rendering:', error);
        const molecule3DPanel = document.getElementById('molecule-3d-panel');
        if (molecule3DPanel) {
            molecule3DPanel.innerHTML = '<p>Error loading 3D structure: ' + error.message + '</p>';
        }
        throw error;
    }
}

/**
 * Clear all molecule visualizations
 */
export function clearMoleculeVisualizations() {
    // Clear 2D visualization
    const content1 = document.getElementById('vis-content-1');
    if (content1) {
        content1.innerHTML = '';
    }
    
    // Clear 3D visualization
    const content3 = document.getElementById('vis-content-3');
    if (content3) {
        content3.innerHTML = '';
    }
    
    // Reset state
    lastMoleculeData = null;
}
