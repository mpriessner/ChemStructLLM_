import { initializeAudioHandlers, initializeSocket, addSpeakButton } from './modules/audio.js';
import { 
    renderMolecule,
    setLastMoleculeData,
    clearAllVisualizations 
} from './modules/visualization.js';
import { renderPlot, clearPlot } from './modules/analysis/plotVisualization.js'; // check later if needed?
import { setPlotData } from './modules/analysis/analysisState.js';
import { initializeChat, setupChatHandlers, appendMessage } from './modules/chat.js';
import { initializeResizer } from './modules/resizer.js';
import { initializeTabs } from './modules/tabs.js';
import { initializeStructureTab } from './modules/structure/structureTabInit.js';

const socket = io();
const messageInput = document.getElementById('message-input');
const modelSelect = document.getElementById('model-select');
const chatMessages = document.getElementById('chat-messages');
const sendButton = document.getElementById('send-button');
const clearChatButton = document.getElementById('clear-chat');

// Initialize modules
initializeSocket(socket);
initializeAudioHandlers();
initializeChat(socket, messageInput, chatMessages, addSpeakButton);
setupChatHandlers(sendButton, clearChatButton, modelSelect);
initializeResizer();
initializeTabs();
initializeStructureTab(); // Initialize structure view

// Initialize test button
document.addEventListener('DOMContentLoaded', () => {
    console.log('Setting up test button handler');
    const testButton = document.getElementById('test-functionality');
    if (testButton) {
        console.log('Test button found');
        testButton.addEventListener('click', () => {
            console.log('Test button clicked');
            runTests();
        });
    } else {
        console.error('Test button not found in DOM');
    }
});

// Initialize structure elucidation button
document.addEventListener('DOMContentLoaded', () => {
    console.log('Setting up structure elucidation button handler');
    const elucidationButton = document.getElementById('run-elucidation');
    if (elucidationButton) {
        console.log('Structure elucidation button found');
        elucidationButton.addEventListener('click', () => {
            const messageInput = document.getElementById('message-input');
            const command = 'perform structure elucidation on all molecules';
            
            // Get the selected model
            const modelSelect = document.getElementById('model-select');
            const selectedModel = modelSelect.value;
            
            // Simulate sending the message
            appendMessage('user', command);
            
            // Send the request to the server with batch processing mode
            fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    user_input: command,
                    model_choice: selectedModel,
                    processing_mode: 'batch'  // Force batch processing mode
                })
            })
            .then(response => response.json())
            .catch(error => {
                console.error('Error in structure elucidation:', error);
                appendMessage('bot', '❌ Structure elucidation failed: ' + error.message);
            });
        });
    } else {
        console.error('Structure elucidation button not found in DOM');
    }
});

async function runTests() {
    console.log('Starting tests');
    const testButton = document.getElementById('test-functionality');
    testButton.disabled = true;
    let allTestsPassed = true;  // Track overall test success
    
    try {
        console.log('Fetching test file');
        appendMessage('bot', '🧪 Starting upload test...');
        
        // Create a File object from the test CSV
        const response = await fetch('/test_data/test_smiles_with_nmr.csv');
        const blob = await response.blob();
        const testFile = new File([blob], 'test_smiles_with_nmr.csv', { type: 'text/csv' });
        
        console.log('Creating form data');
        // Create FormData and append file
        const formData = new FormData();
        formData.append('file', testFile);
        
        console.log('Sending upload request');
        // Perform the upload
        const uploadResponse = await fetch('/upload_file', {
            method: 'POST',
            body: formData
        });
        
        const result = await uploadResponse.json();
        console.log('Upload result:', result);
        
        if (result.error) {
            appendMessage('bot', '❌ Upload test failed: ' + result.error);
            allTestsPassed = false;
        } else {
            appendMessage('bot', '✅ Upload test successful: ' + result.message);
            
            // After successful upload, test show molecule
            console.log('Starting molecule test');
            appendMessage('bot', '🧪 Testing show molecule command...');
            // Get the selected model from the dropdown
            const modelSelect = document.getElementById('model-select');
            const selectedModel = modelSelect.value;
            
            try {
                console.log('Starting tests');
                appendMessage('bot', '🧪 Starting molecule data test...');
                
                // Get first molecule from JSON storage
                const firstMoleculeResponse = await fetch('/get_first_molecule_json');
                const firstMoleculeData = await firstMoleculeResponse.json();
                
                console.log('========== First Molecule Data ==========');
                console.log('Full data:', JSON.stringify(firstMoleculeData, null, 2));
                console.log('SMILES:', firstMoleculeData.smiles);
                console.log('Sample ID:', firstMoleculeData.sample_id);
                console.log('NMR Data Available:', {
                    'proton': !!firstMoleculeData.nmr_data?.proton,
                    'carbon': !!firstMoleculeData.nmr_data?.carbon,
                    'hsqc': !!firstMoleculeData.nmr_data?.hsqc,
                    'cosy': !!firstMoleculeData.nmr_data?.cosy
                });
                console.log('=======================================');
                
                if (firstMoleculeData.error) {
                    throw new Error(firstMoleculeData.error);
                }
                
                // Use the first molecule's SMILES for visualization
                const userMessage_mol = `show molecule "${firstMoleculeData.smiles}"`;
                appendMessage('user', userMessage_mol);

                const plotResponse_mol = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: `user_input=${encodeURIComponent(userMessage_mol)}&model_choice=${encodeURIComponent(selectedModel)}`
                });
                console.log('========== MOLECULE RESPONSE ==========');
                console.log('Response status:', plotResponse_mol.status);
                console.log('Response headers:', Object.fromEntries(plotResponse_mol.headers.entries()));

                const plotData_mol = await plotResponse_mol.json();
                console.log('========== MOLECULE DATA ==========');
                console.log('Plot data structure:', JSON.stringify(plotData_mol, null, 2));

                if (!plotData_mol.data || !plotData_mol.data['2d']) {
                    console.error('Invalid molecule response structure:', plotData_mol);
                    throw new Error('Invalid molecule response structure');
                }

                const testMoleculeData = {
                    smiles: plotData_mol.data['2d'].smiles,
                    molecular_weight: plotData_mol.data['2d'].molecular_weight,
                    image: plotData_mol.data['2d'].image
                };

                console.log('Extracted molecule data:', testMoleculeData);
                
                // Simulate bot response
                appendMessage('bot', 'Here is the molecule structure:');
                
                // Render molecule
                renderMolecule(testMoleculeData);
                setLastMoleculeData(testMoleculeData);
                
                // Wait for rendering
                await new Promise(resolve => setTimeout(resolve, 1000));
                
                // Test NMR plot after molecule is shown
                console.log('Starting NMR test');
                appendMessage('bot', '🧪 Testing NMR plot command...');

                try {
                    // Request 1H NMR plot for the first molecule
                    const userMessage_nmr = 'show 1H NMR spectrum';
                    appendMessage('user', userMessage_nmr);

                    const plotResponse_nmr = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/x-www-form-urlencoded',
                        },
                        body: `user_input=${encodeURIComponent(userMessage_nmr)}&model_choice=${encodeURIComponent(selectedModel)}`
                    });
                    
                    const plotData_nmr = await plotResponse_nmr.json();
                    console.log('NMR Plot Data:', plotData_nmr);

                    if (plotData_nmr.type === 'error') {
                        console.error('NMR Plot Error:', plotData_nmr.content);
                        throw new Error(`NMR Plot Error: ${plotData_nmr.content}`);
                    }
                    
                    if (plotData_nmr.type === 'plot' || plotData_nmr.type === 'nmr_plot') {
                        const plotRequest = {
                            plot_type: plotData_nmr.plot_type,
                            parameters: plotData_nmr.parameters
                        };
                        console.log('Emitting plot request:', plotRequest);
                        socket.emit('plot_request', plotRequest);
                        // Wait for plot to render
                        await new Promise(resolve => setTimeout(resolve, 2000));
                        
                        // If using random data, inform the user
                        if (plotData_nmr.note) {
                            appendMessage('bot', '⚠️ ' + plotData_nmr.note);
                        }
                        
                        appendMessage('bot', '✅ Plot and NMR visualization test successful');
                    } else if (plotData_nmr.type === 'molecule_plot') {
                        if (!plotData_nmr.data || !plotData_nmr.data['2d']) {
                            console.error('Invalid molecule response structure:', plotData_nmr);
                            throw new Error('Invalid molecule response structure');
                        }
                        socket.emit('molecule_data', plotData_nmr.data);
                        // Wait for molecule to render
                        await new Promise(resolve => setTimeout(resolve, 2000));
                        appendMessage('bot', '✅ Molecule visualization test successful');
                    } else {
                        throw new Error('Invalid plot response');
                    }
                        
                    // Wait 1 second before next test
                    await new Promise(resolve => setTimeout(resolve, 1000));
                                       
                    // Test API with a simple greeting
                    console.log('Starting API test');
                    appendMessage('bot', '🧪 Testing API with greeting...');
                    
                    const userMessage_text = 'hello';
                    appendMessage('user', userMessage_text);                

                    // Send test message to API
                    const apiResponse = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/x-www-form-urlencoded',
                        },
                        body: `user_input=${encodeURIComponent(userMessage_text)}&model_choice=${encodeURIComponent(selectedModel)}`
                    });
                    
                    const apiData = await apiResponse.json();
                    console.log('=== API Response Debug ===');
                    console.log('Full API response:', JSON.stringify(apiData, null, 2));

                    // Extract response text from nested structure
                    let responseText = null;
                    
                    if (apiData.type === 'text_response') {
                        if (typeof apiData.content === 'string') {
                            responseText = apiData.content;
                            console.log('Found direct text response');
                        }
                        // Fallback for legacy nested response formats
                        else if (apiData.content?.content?.response) {
                            responseText = apiData.content.content.response;
                            console.log('Found double-nested response');
                        }
                        else if (apiData.content?.response) {
                            responseText = apiData.content.response;
                            console.log('Found single-nested response');
                        }
                    }
                    // Fallback for direct response
                    else if (apiData.response) {
                        responseText = apiData.response;
                        console.log('Found legacy direct response');
                    }

                    if (!responseText) {
                        console.error('Invalid API response:', apiData);
                        throw new Error('Could not extract response text from API response');
                    }

                    appendMessage('bot', `Bot's reply: ${responseText}`);
                    appendMessage('bot', '✅ API test successful');
                    console.log('API test response:', responseText);
                    await new Promise(resolve => setTimeout(resolve, 1000));
                    
                    // If all tests passed, clean up
                    if (allTestsPassed) {
                        // Clear all visualizations
                        console.log('Cleaning up all visualizations');
                        clearAllVisualizations();
                        clearPlot(); // Clear the plot visualization
                        
                        // Clear chat history last
                        console.log('Clearing chat history');
                        await fetch('/clear_chat', { method: 'POST' });
                        const chatMessages = document.getElementById('chat-messages');
                        if (chatMessages) {
                            chatMessages.innerHTML = '';
                        }
                        
                        appendMessage('bot', '✅ All tests completed successfully');
                    }
                } catch (error) {
                    allTestsPassed = false;
                    console.error('Error in NMR test:', error);
                    appendMessage('bot', '❌ NMR test failed: ' + error.message);
                }
            } catch (error) {
                allTestsPassed = false;
                console.error('Error in visualization test:', error);
                appendMessage('bot', '❌ Visualization test failed: ' + error.message);
            }
        }
    } catch (error) {
        allTestsPassed = false;
        console.error('Error in test:', error);
        appendMessage('bot', '❌ Test failed: ' + error.message);
    } finally {
        testButton.disabled = false;
        if (!allTestsPassed) {
            appendMessage('bot', '❌ Some tests failed - keeping test results visible for review');
        } else {
            // Make sure visualizations are cleared even if there's an error
            clearAllVisualizations();
            clearPlot();
        }
    }
}

// Helper function to get molecule image from server
async function getMoleculeImage(smiles) {
    try {
        console.log('[Frontend] Getting molecule image for SMILES:', smiles);
        const response = await fetch('/generate_molecule_image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ smiles: smiles })
        });
        
        console.log('[Frontend] Response status:', response.status);
        
        if (!response.ok) {
            const errorData = await response.json();
            console.error('[Frontend] Server error:', errorData);
            throw new Error(errorData.error || 'Failed to get molecule image');
        }
        
        const data = await response.json();
        console.log('[Frontend] Successfully received molecule data');
        console.log('[Frontend] Response keys:', Object.keys(data));
        
        if (!data.image) {
            console.error('[Frontend] No image data in response');
            throw new Error('No image data in response');
        }
        
        return data.image;
    } catch (error) {
        console.error('[Frontend] Error getting molecule image:', error);
        console.error('[Frontend] Error stack:', error.stack);
        return null;
    }
}

window.onload = function() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
};

// Socket event handlers for visualizations
socket.on('graph_response', (data) => {
    console.log('[Frontend] Received graph_response event:', data);
    try {
        if (!data || !data.plot) {
            throw new Error('Invalid plot data received');
        }
        
        // Ensure data.plot is a string
        if (typeof data.plot === 'object') {
            data.plot = JSON.stringify(data.plot);
        }
        
        setPlotData(data);
        renderPlot(data);
    } catch (error) {
        console.error('[Frontend] Error handling plot data:', error);
        appendMessage('bot', 'Error displaying plot. Please try again.');
    }
});

// socket.on('molecule_response', (data) => {
//     console.log('[Frontend] Received molecule_response event:', data);
//     try {
//         setLastMoleculeData(data);
//         renderMolecule(data);
//     } catch (error) {
//         console.error('[Frontend] Error handling molecule data:', error);
//         appendMessage('bot', 'Error displaying molecule. Please try again.');
//     }
// });

// socket.on('molecule_data', (data) => {
//     console.log('[Frontend] Received molecule_data event:', data);
//     try {
//         if (!data || !data.image) {
//             console.error('[Frontend] Invalid molecule data received');
//             return;
//         }
        
//         const container = document.querySelector('#molecule-container');
//         if (!container) {
//             console.error('[Frontend] Molecule container not found');
//             return;
//         }
        
//         console.log('[Frontend] Updating molecule container');
//         // Clear existing content
//         container.innerHTML = '';
        
//         // Create and append image
//         const img = document.createElement('img');
//         img.src = `data:image/png;base64,${data.image}`;
//         img.alt = 'Molecule Visualization';
//         img.style.maxWidth = '100%';
//         container.appendChild(img);
        
//         console.log('[Frontend] Successfully displayed molecule');
//     } catch (error) {
//         console.error('[Frontend] Error handling molecule data:', error);
//     }
// });

// socket.on('molecule_image', (data) => {
//     console.log('[Frontend] Received molecule_image event:', data);
//     try {
//         if (!data || !data.image) {
//             console.error('[Frontend] Invalid molecule image data received');
//             return;
//         }
        
//         const container = document.querySelector('#molecule-image-container');
//         if (!container) {
//             console.error('[Frontend] Molecule image container not found');
//             return;
//         }
        
//         console.log('[Frontend] Updating molecule image container');
//         // Clear existing content
//         container.innerHTML = '';
        
//         // Create and append image
//         const img = document.createElement('img');
//         img.src = `data:image/png;base64,${data.image}`;
//         img.alt = 'Molecule Image';
//         img.style.maxWidth = '100%';
//         container.appendChild(img);
        
//         console.log('[Frontend] Successfully displayed molecule image');
//     } catch (error) {
//         console.error('[Frontend] Error handling molecule image:', error);
//     }
// });

socket.on('error', (data) => {
    console.error('[Frontend] Received error:', data);
    // Display error in UI if needed
});

// File handling functions
function handleFileSelect(event, containerId) {
    console.log('[Frontend] Handling file select event');
    const files = event.target.files;
    const uploadButton = document.getElementById(`upload-button-${containerId}`);
    const uploadStatus = document.getElementById(`upload-status-${containerId}`);
    const fileList = document.getElementById(`file-list-${containerId}`);
    
    uploadStatus.className = 'upload-status';
    fileList.innerHTML = '';
    
    if (files && files.length > 0) {
        const file = files[0];
        
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        
        const filePath = document.createElement('span');
        filePath.className = 'file-path';
        filePath.textContent = file.name;
        fileItem.appendChild(filePath);
        
        const removeBtn = document.createElement('button');
        removeBtn.className = 'remove-file';
        removeBtn.innerHTML = '&times;';
        removeBtn.onclick = () => {
            fileItem.remove();
            event.target.value = '';
            uploadButton.disabled = true;
            uploadStatus.className = 'upload-status';
        };
        fileItem.appendChild(removeBtn);
        
        fileList.appendChild(fileItem);
        
        const isValidFile = file.name.endsWith('.csv') || file.name.toLowerCase() === 'molecular_data.json';
        uploadButton.disabled = !isValidFile;
        if (!isValidFile) {
            uploadStatus.textContent = 'Please select a CSV file or molecular_data.json';
            uploadStatus.className = 'upload-status error';
        }
    } else {
        uploadButton.disabled = true;
    }
}

function uploadSelectedFile(containerId) {
    console.log('[Frontend] Uploading selected file');
    const fileInput = document.getElementById(`file-input-${containerId}`);
    const uploadStatus = document.getElementById(`upload-status-${containerId}`);
    const uploadButton = document.getElementById(`upload-button-${containerId}`);
    
    if (!fileInput.files.length) return;
    
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

    uploadButton.disabled = true;
    uploadStatus.textContent = 'Uploading...';
    uploadStatus.className = 'upload-status';
    
    fetch('/upload_file', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            uploadStatus.textContent = `Error: ${data.error}`;
            uploadStatus.className = 'upload-status error';
            uploadButton.disabled = false;
        } else {
            uploadStatus.textContent = data.message;
            uploadStatus.className = 'upload-status success';
            appendMessage('bot', data.message);
        }
    })
    .catch(error => {
        uploadStatus.textContent = 'Error uploading file';
        uploadStatus.className = 'upload-status error';
        uploadButton.disabled = false;
    });
}

// Export file handling functions
window.handleFileSelect = handleFileSelect;
window.uploadSelectedFile = uploadSelectedFile;