// Structure view 2D molecule visualization module

/**
 * Render a 2D molecule visualization in the structure view tab
 * @param {Object} data Molecule data object containing image and metadata
 */
export function renderStructureView2D(data) {
    console.log('[StructureView2D] Starting 2D molecule render:', data);
    
    // First, check if we're in the structure tab
    const structureTab = document.querySelector('.tab-content[data-tab="structure"]');
    if (!structureTab || !structureTab.classList.contains('active')) {
        console.error('[StructureView2D] Structure tab is not active or not found');
        return;
    }
    
    const container = document.getElementById('vis-content-structure');
    console.log('[StructureView2D] Looking for container:', {
        containerFound: !!container,
        containerClasses: container ? container.className : 'not found',
        containerChildren: container ? container.children.length : 0
    });
    
    if (!container) {
        console.error('[StructureView2D] Container not found: vis-content-structure');
        throw new Error('2D visualization container not found');
    }

    try {
        // Get or create the visualization panel
        let visPanel = container.querySelector('.visualization-panel');
        if (!visPanel) {
            console.log('[StructureView2D] Creating visualization panel');
            visPanel = document.createElement('div');
            visPanel.className = 'visualization-panel active';
            container.appendChild(visPanel);
        }

        // Clear existing content
        console.log('[StructureView2D] Clearing existing content');
        visPanel.innerHTML = '';

        if (!data || !data.image) {
            console.error('[StructureView2D] Invalid data:', data);
            throw new Error('No molecule image data provided');
        }

        // Create molecule container with flex layout
        console.log('[StructureView2D] Creating molecule container');
        const molContainer = document.createElement('div');
        molContainer.className = 'molecule-container';

        // Add zoom functionality
        let scale = 1;
        const MIN_SCALE = 0.5;
        const MAX_SCALE = 5;
        const ZOOM_SPEED = 0.1;

        // Function to update zoom level
        function updateZoom(deltaY) {
            const zoomIn = deltaY < 0;
            const oldScale = scale;
            
            // Calculate new scale
            if (zoomIn) {
                scale = Math.min(scale * (1 + ZOOM_SPEED), MAX_SCALE);
            } else {
                scale = Math.max(scale * (1 - ZOOM_SPEED), MIN_SCALE);
            }
            
            // Update zoom indicator
            const zoomIndicator = molContainer.querySelector('.zoom-indicator');
            if (zoomIndicator) {
                zoomIndicator.textContent = `Zoom: ${Math.round(scale * 100)}%`;
            }
            
            // Update cursor style based on zoom limits
            const imgContainer = molContainer.querySelector('.molecule-image-container');
            if (imgContainer) {
                if (scale === MAX_SCALE) {
                    imgContainer.style.cursor = 'zoom-out';
                } else if (scale === MIN_SCALE) {
                    imgContainer.style.cursor = 'zoom-in';
                } else {
                    imgContainer.style.cursor = 'zoom-in';
                }
            }
            
            // Update image transform
            const img = molContainer.querySelector('.zoomable-molecule');
            if (img) {
                img.style.transform = `scale(${scale})`;
            }
        }

        // Add wheel event listener for zooming
        molContainer.addEventListener('wheel', (event) => {
            event.preventDefault();
            updateZoom(event.deltaY);
        }, { passive: false });

        // Create image container with improved structure
        console.log('[StructureView2D] Creating image container');
        const imgContainer = document.createElement('div');
        imgContainer.className = 'molecule-image-container';
        imgContainer.style.cssText = `
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 20px;
            flex: 1;
            width: 100%;
            overflow: hidden;
            position: relative;
            cursor: zoom-in;
            touch-action: none; /* Prevent touch scrolling while zooming */
            background: rgba(255, 255, 255, 0.8);
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        `;

        // Create and set up the image with improved transform handling
        console.log('[StructureView2D] Creating molecule image');
        const img = document.createElement('img');
        img.src = 'data:image/png;base64,' + data.image;
        img.alt = 'Molecule structure';
        img.className = 'zoomable-molecule';
        img.style.cssText = `
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            transform-origin: center center;
            transition: transform 0.1s ease-out;
            will-change: transform;
            pointer-events: none; /* Prevent image from capturing events */
        `;

        // Create zoom level indicator with improved visibility
        const zoomInfo = document.createElement('div');
        zoomInfo.className = 'zoom-indicator';
        zoomInfo.style.cssText = `
            position: absolute;
            bottom: 10px;
            right: 10px;
            background: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            z-index: 1000;
            user-select: none;
        `;
        zoomInfo.textContent = 'Zoom: 100%';

        // Add zoom functionality to the image container
        let imageScale = 1;
        const MIN_IMAGE_SCALE = 0.5;
        const MAX_IMAGE_SCALE = 5;
        const IMAGE_ZOOM_SPEED = 0.1;

        // Function to update image zoom level
        function updateImageZoom(deltaY) {
            console.log('[StructureView2D] Zoom event detected:', {
                deltaY,
                currentScale: imageScale,
                container: 'molecule-image-container'
            });
            
            const zoomIn = deltaY < 0;
            const oldScale = imageScale;
            
            // Calculate new scale with improved precision
            const zoomFactor = 1 + (zoomIn ? IMAGE_ZOOM_SPEED : -IMAGE_ZOOM_SPEED);
            imageScale = Math.round((imageScale * zoomFactor) * 100) / 100; // Round to 2 decimal places
            imageScale = Math.min(Math.max(imageScale, MIN_IMAGE_SCALE), MAX_IMAGE_SCALE);
            
            console.log('[StructureView2D] Scale calculation:', {
                oldScale,
                newScale: imageScale,
                zoomFactor,
                zoomIn
            });
            
            if (imageScale !== oldScale) {
                // Update image transform
                const transform = `scale(${imageScale})`;
                img.style.transform = transform;
                console.log('[StructureView2D] Applied transform:', transform);
                
                // Update zoom indicator
                const zoomPercentage = Math.round(imageScale * 100);
                zoomInfo.textContent = `Zoom: ${zoomPercentage}%`;
                
                // Update cursor based on zoom level
                imgContainer.style.cursor = imageScale >= MAX_IMAGE_SCALE ? 'zoom-out' : 
                                         imageScale <= MIN_IMAGE_SCALE ? 'zoom-in' : 
                                         'zoom-in';
                
                console.log('[StructureView2D] Zoom updated:', {
                    scale: imageScale,
                    percentage: zoomPercentage,
                    cursor: imgContainer.style.cursor
                });
            } else {
                console.log('[StructureView2D] Scale unchanged - at limit:', imageScale);
            }
        }

        // Add wheel event listener with improved handling
        imgContainer.addEventListener('wheel', (e) => {
            e.preventDefault();
            e.stopPropagation();
            
            // Ensure we're in the structure tab
            const structureTab = document.querySelector('.tab-content[data-tab="structure"]');
            if (!structureTab || !structureTab.classList.contains('active')) {
                console.log('[StructureView2D] Ignoring wheel event - not in structure tab');
                return;
            }
            
            console.log('[StructureView2D] Wheel event:', {
                deltaY: e.deltaY,
                deltaMode: e.deltaMode,
                ctrlKey: e.ctrlKey
            });
            
            // Use deltaY for zoom calculation
            updateImageZoom(e.deltaY);
        }, { passive: false });

        // Create info container
        console.log('[StructureView2D] Creating info container');
        const infoContainer = document.createElement('div');
        infoContainer.className = 'molecule-info';
        infoContainer.style.cssText = `
            text-align: center;
            width: 100%;
            padding: 10px;
            background: rgba(0, 0, 0, 0.05);
            border-radius: 5px;
            margin-top: 10px;
            user-select: none;
        `;

        // Add molecule information
        if (data.smiles) {
            console.log('[StructureView2D] Adding SMILES information');
            const smilesInfo = document.createElement('p');
            smilesInfo.style.margin = '5px 0';
            smilesInfo.textContent = 'SMILES: ' + data.smiles;
            infoContainer.appendChild(smilesInfo);
        }

        if (data.molecular_weight) {
            console.log('[StructureView2D] Adding molecular weight information');
            const weightInfo = document.createElement('p');
            weightInfo.style.margin = '5px 0';
            weightInfo.textContent = 'Molecular Weight: ' + data.molecular_weight;
            infoContainer.appendChild(weightInfo);
        }

        // Add zoom instructions
        const instructions = document.createElement('p');
        instructions.style.cssText = `
            margin: 5px 0;
            font-style: italic;
            color: #666;
            font-size: 0.9em;
        `;
        instructions.textContent = 'Use mouse wheel to zoom in/out';
        infoContainer.appendChild(instructions);

        // Assemble the components
        console.log('[StructureView2D] Assembling components');
        imgContainer.appendChild(img);
        imgContainer.appendChild(zoomInfo);
        molContainer.appendChild(imgContainer);
        molContainer.appendChild(infoContainer);
        visPanel.appendChild(molContainer);

        console.log('[StructureView2D] Render complete - zoom functionality ready');

    } catch (error) {
        console.error('[StructureView2D] Error in renderStructureView2D:', error);
        throw error;
    }
}
