const API_URL = '';
let currentSessionId = null;

// DOM Elements
const fileInput = document.getElementById('file-upload');
const uploadStatus = document.getElementById('upload-status');
const newSessionBtn = document.getElementById('new-session-btn');
const endSessionBtn = document.getElementById('end-session-btn');
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const currentSessionBadge = document.getElementById('current-session-id');

// Initialize
async function init() {
    await createNewSession();
}

// Event Listeners
fileInput.addEventListener('change', handleFileUpload);
newSessionBtn.addEventListener('click', createNewSession);
endSessionBtn.addEventListener('click', endSession);
sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('input', () => {
    sendBtn.disabled = !chatInput.value.trim();
    adjustTextareaHeight();
});
chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Functions
async function createNewSession() {
    try {
        const response = await fetch(`${API_URL}/session/new`, { method: 'POST' });
        const data = await response.json();
        currentSessionId = data.session_id;
        currentSessionBadge.textContent = `Session: ${currentSessionId.slice(0, 8)}...`;

        chatMessages.innerHTML = `
            <div class="message system">
                <div class="message-content">
                    New session started. Upload a document or start chatting!
                </div>
            </div>
        `;
        enableInput(true);
    } catch (error) {
        console.error('Error creating session:', error);
        alert('Failed to create session. Make sure the backend is running!');
    }
}

async function endSession() {
    if (!currentSessionId) return;

    try {
        await fetch(`${API_URL}/session/end`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: currentSessionId })
        });

        currentSessionId = null;
        currentSessionBadge.textContent = 'Session: Ended';
        chatMessages.innerHTML += `
            <div class="message system">
                <div class="message-content">
                    Session ended. Click "New Session" to start again.
                </div>
            </div>
        `;
        enableInput(false);
    } catch (error) {
        console.error('Error ending session:', error);
        alert('Failed to end session. Make sure the backend is running!');
    }
}

async function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    uploadStatus.textContent = 'Uploading & Ingesting...';

    try {
        const response = await fetch(`${API_URL}/ingest`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Upload failed');

        const data = await response.json();
        uploadStatus.textContent = 'Upload Complete!';
        setTimeout(() => uploadStatus.textContent = '', 3000);

        addMessage('system', `Document "${file.name}" uploaded and processed successfully.`);

        // Fetch suggestions
        fetchSuggestions();
    } catch (error) {
        uploadStatus.textContent = 'Error uploading file';
        console.error(error);
        alert('Failed to upload file. Make sure the backend is running!');
    }
}

async function fetchSuggestions() {
    const container = document.getElementById('suggestions-container');
    container.innerHTML = '<div class="status-msg">Generating questions...</div>';

    try {
        const response = await fetch(`${API_URL}/suggest_questions`, { method: 'POST' });
        const data = await response.json();

        container.innerHTML = '';
        if (data.questions && data.questions.length > 0) {
            data.questions.forEach(q => {
                const chip = document.createElement('div');
                chip.className = 'suggestion-chip';
                chip.textContent = q;
                chip.onclick = () => {
                    chatInput.value = q;
                    sendMessage();
                    container.innerHTML = ''; // Clear after use
                };
                container.appendChild(chip);
            });
        }
    } catch (error) {
        console.error('Error fetching suggestions:', error);
        container.innerHTML = '';
    }
}

async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text || !currentSessionId) return;

    // Add user message
    addMessage('user', text);
    chatInput.value = '';
    sendBtn.disabled = true;
    adjustTextareaHeight();

    // Create placeholder for assistant response
    const assistantMsgDiv = document.createElement('div');
    assistantMsgDiv.className = 'message assistant';
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    assistantMsgDiv.appendChild(contentDiv);
    chatMessages.appendChild(assistantMsgDiv);
    scrollToBottom();

    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: text,
                session_id: currentSessionId
            })
        });

        if (!response.ok) throw new Error('Chat request failed');

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullResponse = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            fullResponse += chunk;
            contentDiv.textContent = fullResponse;
            scrollToBottom();
        }
    } catch (error) {
        contentDiv.textContent = 'Error generating response. Is the backend running?';
        console.error(error);
        alert('Failed to send message. Make sure the backend is running!');
    }
}

function addMessage(role, content) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;
    msgDiv.innerHTML = `<div class="message-content">${content}</div>`;
    chatMessages.appendChild(msgDiv);
    scrollToBottom();
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function adjustTextareaHeight() {
    chatInput.style.height = 'auto';
    chatInput.style.height = chatInput.scrollHeight + 'px';
}

function enableInput(enabled) {
    chatInput.disabled = !enabled;
    if (!enabled) sendBtn.disabled = true;
}

// Start
init();
