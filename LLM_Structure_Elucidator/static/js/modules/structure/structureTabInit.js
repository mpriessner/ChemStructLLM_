import { initializeZoom } from './structureViewZoom.js';

export function initializeStructureTab() {
    const structureContainer = document.getElementById('structure-container-1');
    if (structureContainer) {
        initializeZoom(structureContainer);
    }
}

// Initialize when the structure tab is shown
document.addEventListener('DOMContentLoaded', () => {
    const structureTabButton = document.querySelector('button[data-tab="structure"]');
    if (structureTabButton) {
        structureTabButton.addEventListener('click', () => {
            // Small delay to ensure container is visible
            setTimeout(initializeStructureTab, 100);
        });
    }
    
    // Also initialize if structure tab is active by default
    if (document.querySelector('.tab-content[data-tab="structure"].active')) {
        initializeStructureTab();
    }
});
