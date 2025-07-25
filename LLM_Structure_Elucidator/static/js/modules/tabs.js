import { initializeStructureTab } from './structure/structureTabInit.js';

export function initializeTabs() {
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tab = button.dataset.tab;

            // Remove active class from all buttons and contents
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));

            // Add active class to clicked button and corresponding content
            button.classList.add('active');
            document.querySelector(`.tab-content[data-tab="${tab}"]`).classList.add('active');

            // Initialize structure tab if it's being shown
            if (tab === 'structure') {
                setTimeout(initializeStructureTab, 100);
            }
        });
    });
}
