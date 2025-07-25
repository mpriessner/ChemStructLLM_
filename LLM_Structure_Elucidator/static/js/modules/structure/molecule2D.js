// 2D Molecule visualization module

/**
 * Render a 2D molecule visualization
 * @param {Object} data Molecule data object containing image and metadata
 */
export function renderMolecule2D(data) {
    console.log('[Molecule2D] Starting 2D molecule render:', data);
    const container = document.getElementById('vis-content-1');
    if (!container) {
        console.error('[Molecule2D] Container not found: vis-content-1');
        throw new Error('2D visualization container not found');
    }

    try {
        // Create or get molecule panel
        let moleculePanel = document.getElementById('molecule-panel-1');
        if (!moleculePanel) {
            console.log('[Molecule2D] Creating new molecule panel');
            moleculePanel = document.createElement('div');
            moleculePanel.id = 'molecule-panel-1';
            moleculePanel.className = 'visualization-panel active';
            container.appendChild(moleculePanel);
        }

        // Clear existing content
        console.log('[Molecule2D] Clearing existing content');
        moleculePanel.innerHTML = '';
        
        // Clear placeholder if it exists
        const placeholder = container.querySelector('.content-placeholder');
        if (placeholder) {
            console.log('[Molecule2D] Removing placeholder');
            placeholder.remove();
        }

        if (!data || !data.image) {
            console.error('[Molecule2D] Invalid data:', data);
            throw new Error('No molecule image data provided');
        }

        // Create molecule container with flex layout
        console.log('[Molecule2D] Creating molecule container');
        const molContainer = document.createElement('div');
        molContainer.className = 'molecule-container';
        molContainer.style.cssText = `
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
            height: 100%;
        `;

        // Create image container
        console.log('[Molecule2D] Creating image container');
        const imgContainer = document.createElement('div');
        imgContainer.className = 'molecule-image-container';
        imgContainer.style.cssText = `
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 20px;
            flex: 1;
            width: 100%;
        `;

        // Create and set up the image
        console.log('[Molecule2D] Creating molecule image');
        const img = document.createElement('img');
        img.src = 'data:image/png;base64,' + data.image;
        img.alt = 'Molecule structure';
        img.style.cssText = `
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        `;
        imgContainer.appendChild(img);

        // Create info container
        console.log('[Molecule2D] Creating info container');
        const infoContainer = document.createElement('div');
        infoContainer.className = 'molecule-info';
        infoContainer.style.cssText = `
            text-align: center;
            width: 100%;
            padding: 10px;
            background: rgba(0, 0, 0, 0.05);
            border-radius: 5px;
        `;

        // Add molecule information
        if (data.smiles) {
            console.log('[Molecule2D] Adding SMILES information');
            const smilesInfo = document.createElement('p');
            smilesInfo.textContent = 'SMILES: ' + data.smiles;
            infoContainer.appendChild(smilesInfo);
        }

        if (data.molecular_weight) {
            console.log('[Molecule2D] Adding molecular weight information');
            const weightInfo = document.createElement('p');
            weightInfo.textContent = 'Molecular Weight: ' + data.molecular_weight;
            infoContainer.appendChild(weightInfo);
        }

        // Assemble the components
        console.log('[Molecule2D] Assembling components');
        molContainer.appendChild(imgContainer);
        molContainer.appendChild(infoContainer);
        moleculePanel.appendChild(molContainer);

    } catch (error) {
        console.error('[Molecule2D] Error in renderMolecule2D:', error);
        throw error;
    }
}
