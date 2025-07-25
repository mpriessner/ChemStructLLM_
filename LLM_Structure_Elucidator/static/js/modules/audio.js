// Audio handling functionality
let socket;
let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let recordButton;
let modelSelect;
let messageInput;
let sendButton;
let retryCount = 0;
const MAX_RETRIES = 3;
const RETRY_DELAY = 1000; // 1 second

function initializeSocket(sharedSocket) {
    socket = sharedSocket;
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        retryCount = 0;

        mediaRecorder.addEventListener('dataavailable', event => {
            audioChunks.push(event.data);
        });

        mediaRecorder.addEventListener('stop', async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            await sendAudioForTranscription(audioBlob, stream);
        });

        mediaRecorder.start();
        isRecording = true;
        updateRecordButtonState();
    } catch (error) {
        console.error('Error accessing microphone:', error);
        socket.emit('error_message', 'Error accessing microphone. Please make sure you have granted microphone permissions.');
    }
}

async function sendAudioForTranscription(audioBlob, stream, isRetry = false) {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');
    formData.append('model_choice', modelSelect.value);

    try {
        console.log(`Sending audio for transcription... ${isRetry ? '(Retry attempt ' + retryCount + ')' : ''}`);
        const response = await fetch('/transcribe', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const data = await response.json();
            console.log('Transcription response:', data);
            
            if (data.error && data.error.type === 'overloaded_error') {
                if (retryCount < MAX_RETRIES) {
                    retryCount++;
                    console.log(`Server overloaded. Retrying in ${RETRY_DELAY}ms... (Attempt ${retryCount}/${MAX_RETRIES})`);
                    socket.emit('error_message', `Server busy. Retrying transcription... (Attempt ${retryCount}/${MAX_RETRIES})`);
                    await new Promise(resolve => setTimeout(resolve, RETRY_DELAY));
                    await sendAudioForTranscription(audioBlob, stream, true);
                    return;
                } else {
                    throw new Error('Server is currently overloaded. Please try again later.');
                }
            }

            if (data.text) {
                messageInput.value = data.text;
                sendButton.click();
            } else if (data.error) {
                throw new Error(data.error.message || 'Transcription failed');
            } else {
                throw new Error('No text in transcription response');
            }
        } else {
            const errorText = await response.text();
            throw new Error(`Transcription failed: ${errorText}`);
        }
    } catch (error) {
        console.error('Error:', error);
        socket.emit('error_message', error.message || 'An error occurred while processing your audio.');
    } finally {
        if (!isRetry || retryCount >= MAX_RETRIES) {
            // Clean up only if not retrying or if max retries reached
            stream.getTracks().forEach(track => track.stop());
        }
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        isRecording = false;
        updateRecordButtonState();
    }
}

function updateRecordButtonState() {
    if (!recordButton) return;
    
    if (isRecording) {
        recordButton.classList.add('recording');
        recordButton.querySelector('i').classList.remove('fa-microphone');
        recordButton.querySelector('i').classList.add('fa-stop');
    } else {
        recordButton.classList.remove('recording');
        recordButton.querySelector('i').classList.remove('fa-stop');
        recordButton.querySelector('i').classList.add('fa-microphone');
    }
}

function toggleRecording() {
    if (!isRecording) {
        startRecording();
    } else {
        stopRecording();
    }
}

function initializeAudioHandlers() {
    recordButton = document.getElementById('micButton');
    modelSelect = document.getElementById('model-select');
    messageInput = document.getElementById('message-input');
    sendButton = document.getElementById('send-button');
    
    if (!recordButton || !modelSelect || !messageInput || !sendButton) {
        console.error('Required audio elements not found');
        return;
    }

    recordButton.addEventListener('click', toggleRecording);
}

async function speakText(text) {
    try {
        const response = await fetch('/text-to-speech', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        if (data.error) {
            console.error('Text-to-speech error:', data.error);
            return;
        }

        // Create and play audio
        const audio = new Audio('data:audio/mpeg;base64,' + data.audio);
        await audio.play();
    } catch (error) {
        console.error('Error playing audio:', error);
    }
}

function addSpeakButton(messageDiv, content) {
    const controlsDiv = document.createElement('div');
    controlsDiv.className = 'message-controls';
    
    const speakButton = document.createElement('button');
    speakButton.className = 'speak-button';
    speakButton.innerHTML = '<i class="fas fa-volume-up"></i>';
    speakButton.onclick = () => speakText(content);
    controlsDiv.appendChild(speakButton);
    messageDiv.appendChild(controlsDiv);
}

export { 
    initializeAudioHandlers,
    initializeSocket,
    addSpeakButton,
    speakText
};