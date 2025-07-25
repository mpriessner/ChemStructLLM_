//import { renderMolecule, setLastMoleculeData } from './visualization.js';
import { renderMolecule, setLastMoleculeData, clearNMRPlot } from './visualization.js';

let socket;
let messageInput;
let chatMessages;
let addSpeakButton;
let currentSmiles = null;

export function setCurrentSmiles(smiles) {
    console.log('[Chat] Setting current SMILES:', smiles);
    currentSmiles = smiles;
}

export function initializeChat(socketInstance, msgInput, messages, speakButtonFn) {
    socket = socketInstance;
    messageInput = msgInput;
    chatMessages = messages;
    addSpeakButton = speakButtonFn;
    setupSocketListeners();
}

export function appendMessage(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role === 'user' ? 'user-message' : 'bot-message'}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;
    messageDiv.appendChild(contentDiv);
    
    if (role === 'bot') {
        addSpeakButton(messageDiv, content);
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Function to request HSQC plot
async function requestHSQCPlot() {
    console.log('[Chat] Requesting HSQC plot for current molecule');
    const modelSelect = document.getElementById('model-select');
    const selectedModel = modelSelect ? modelSelect.value : 'gpt-4';
    
    try {
        // First ensure we have the current molecule data from JSON storage
        const response = await fetch('/get_molecular_data');
        const jsonData = await response.json();
        
        if (jsonData.error) {
            console.error('[Chat] Error getting molecular data:', jsonData.error);
            return;
        }
        
        // Now request the HSQC plot
        const plotResponse = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `user_input=${encodeURIComponent('show HSQC')}&model_choice=${encodeURIComponent(selectedModel)}`
        });
        const data = await plotResponse.json();
        handleChatResponse(data);
    } catch (error) {
        console.error('[Chat] Error requesting HSQC plot:', error);
    }
}

function handleChatResponse(data) {
    console.log('[Chat] Handling chat response:', data);
    if (data.error) {
        console.log('[Chat] Error in response:', data.error);
        appendMessage('bot', `Error: ${data.error}`);
        return;
    }
    
    if (data.type === 'error') {
        console.log('[Chat] Error type response:', data);
        let errorMessage = `Error: ${data.content}`;
        if (data.metadata?.reasoning) {
            errorMessage += `\n\nReasoning: ${data.metadata.reasoning}`;
        }
        appendMessage('bot', errorMessage);
        return;
    }
    
    if (data.type === 'clarification') {
        console.log('[Chat] Handling clarification response:', data);
        let message = data.content;
        if (data.metadata?.reasoning) {
            message += `\n\nReasoning: ${data.metadata.reasoning}`;
        }
        appendMessage('bot', message);
        return;
    } else if (data.type === 'molecule' || data.type === 'molecule_plot') {
        console.log('[Chat] Molecule visualization requested');
        appendMessage('bot', 'Generating molecule visualization...');
        
        // Handle both data structures (with and without content wrapper)
        const moleculeData = data.content?.data || data.data;
        if (!moleculeData || !moleculeData['2d']) {
            console.error('[Chat] Invalid molecule response structure:', data);
            appendMessage('bot', 'Error: Invalid molecule response structure');
            return;
        }

        // Clear any existing NMR plot before showing new molecule
        clearNMRPlot();

        // Extract 2D data
        const data2D = moleculeData['2d'];
        
        // Update NMR data keys if present
        if (data2D.nmr_data) {
            data2D.nmr_data = {
                ...data2D.nmr_data,
                proton: data2D.nmr_data['1h'] || data2D.nmr_data.proton,
                carbon: data2D.nmr_data['13c'] || data2D.nmr_data.carbon
            };
        }

        // Prepare molecule data for rendering
        const renderData = {
            smiles: data2D.smiles,
            image: data2D.image,
            format: data2D.format,
            molecular_weight: data2D.molecular_weight,
            nmr_data: data2D.nmr_data
        };

        console.log('[Chat] Extracted molecule data:', renderData);
        
        try {
            // Render the molecule directly instead of using socket events
            renderMolecule(renderData);
            setLastMoleculeData(renderData);
            console.log('[Chat] Molecule rendered successfully');
            
            if (renderData.smiles) {
                setCurrentSmiles(renderData.smiles);
                // Automatically request HSQC plot after molecule is rendered
                requestHSQCPlot();
            }
        } catch (error) {
            console.error('[Chat] Error rendering molecule:', error);
            appendMessage('bot', 'Error: Failed to render molecule visualization');
        }
        return;

    } else if (data.type === 'plot') {
        console.log('[Chat] Plot visualization requested:', data);
        
        // Get plot parameters from server response
        const plotData = {
            plot_type: data.plot_type || data.content?.plot_type || 'proton',
            parameters: data.parameters || data.content?.parameters || {
                title: 'NMR Spectrum',
                x_label: 'Chemical Shift (ppm)',
                y_label: 'Intensity',
                style: 'default'
            }
        };
        
        console.log('[Chat] Emitting plot request:', plotData);
        
        // Set response message based on plot type
        const messages = {
            'hsqc': 'Generating HSQC NMR spectrum...',
            'cosy': 'Generating COSY NMR spectrum...',
            'carbon': 'Generating 13C NMR spectrum...',
            'proton': 'Generating 1H NMR spectrum...'
        };
        const responseMessage = messages[plotData.plot_type] || 'Generating NMR spectrum...';
        
        appendMessage('bot', responseMessage);
        socket.emit('plot_request', plotData);
        return;
    } else if (data.type === 'text_response') {
        console.log('[Chat] Handling text response:', data);
        console.log('[Chat] Response structure:', {
            hasContent: !!data.content,
            contentType: data.content && typeof data.content,
            nestedType: data.content && data.content.type,
            nestedContent: data.content && data.content.content
        });

        // Extract response text from nested structure
        let responseText = null;
        
        // Handle double-nested response
        if (data.content?.content?.response) {
            responseText = data.content.content.response;
            console.log('[Chat] Found double-nested response');
        }
        // Handle single-nested response
        else if (data.content?.response) {
            responseText = data.content.response;
            console.log('[Chat] Found single-nested response');
        }
        // Handle direct content
        else if (typeof data.content === 'string') {
            responseText = data.content;
            console.log('[Chat] Found direct content response');
        }

        if (responseText) {
            appendMessage('bot', responseText);
        } else {
            console.error('[Chat] Could not extract response text from:', data);
            appendMessage('bot', 'Error: Could not process response from server');
        }
    } else if (data.response) {
        // Fallback for legacy response format
        console.log('[Chat] Handling legacy response format:', data);
        appendMessage('bot', data.response);
    }
}

function setupSocketListeners() {
    console.log('[Chat] Setting up socket listeners');
    
    // Handle molecule data events
    socket.on('molecule_data', async (data) => {
        console.log('[Chat] Received molecule data:', data);
        // try{
        // Render the molecule using the visualization module
        // renderMolecule(data);
        // setLastMoleculeData(data);
        // await renderMolecule(data);

        console.log('[Chat] Molecule rendered successfully');
        
        // Update current SMILES if available
        if (data.smiles) {
            setCurrentSmiles(data.smiles);
        }   
        // } catch (error) {
        //     console.error('[Chat] Error rendering molecule:', error);
        //     appendMessage('bot', 'Error: Failed to render molecule visualization');
        // }
    });
    
    // Handle plot data events
    socket.on('plot', async (data) => {
        console.log('[Chat] Received plot data:', data);
        try {
            // Import plot visualization module dynamically
            const { renderPlot } = await import('./analysis/plotVisualization.js');
            console.log('[Chat] Imported plot visualization module');
            
            // Render the plot
            await renderPlot(data);
            console.log('[Chat] Plot rendered successfully');
            
        } catch (error) {
            console.error('[Chat] Error handling plot data:', error);
            appendMessage('bot', `Error displaying plot: ${error.message}`);
        }
    });

    // Handle general messages
    socket.on('message', (data) => {
        console.log('[Chat] Received message:', data);
        if (data && data.content) {
            appendMessage('bot', data.content);
        }
    });

    socket.on('error', (error) => {
        console.error('[Chat] Socket error:', error);
        appendMessage('bot', `Error: ${error.message || 'An error occurred'}`);
    });

    socket.on('connect', () => {
        console.log('[Chat] Socket connected');
    });

    socket.on('disconnect', () => {
        console.log('[Chat] Socket disconnected');
    });
}

export function setupChatHandlers(sendButton, clearChatButton, modelSelect) {
    sendButton.addEventListener('click', () => {
        const message = messageInput.value.trim();
        const model = modelSelect.value;

        if (message) {
            appendMessage('user', message);
            messageInput.value = '';

            fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ user_input: message, model_choice: model })
            })
            .then(response => response.json())
            .then(data => {
                handleChatResponse(data);
            })
            .catch(error => {
                console.error('Error:', error);
                appendMessage('bot', 'An error occurred while processing your message.');
            });
        }
    });

    clearChatButton.addEventListener('click', () => {
        fetch('/clear_chat', { method: 'POST' })
            .then(() => {
                chatMessages.innerHTML = '';
            })
            .catch(error => console.error('Error:', error));
    });

    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            sendButton.click();
        }
    });
}