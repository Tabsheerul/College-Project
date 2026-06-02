// DOM Elements
const currentLetterEl = document.getElementById('current-letter');
const sentenceEl = document.getElementById('current-sentence');
const seeingBoxEl = document.getElementById('currently-seeing-box');
const statusTextEl = document.getElementById('status-text');

// State fetching loop
async function fetchState() {
    try {
        const response = await fetch('/state');
        const data = await response.json();
        
        // Update Live Letter
        if (data.current_letter) {
            currentLetterEl.textContent = data.current_letter;
            seeingBoxEl.classList.add('detecting');
            statusTextEl.textContent = "Sign Detected. Press Space to add.";
            statusTextEl.className = "text-brand-teal font-medium mt-2";
        } else {
            currentLetterEl.textContent = '-';
            seeingBoxEl.classList.remove('detecting');
            statusTextEl.textContent = "System Ready. Show a sign!";
            statusTextEl.className = "text-gray-400 font-medium mt-2";
        }
        
        // Update Sentence
        if (data.current_sentence.length > 0) {
            sentenceEl.textContent = data.current_sentence;
        } else {
            sentenceEl.textContent = '-';
        }
        
    } catch (error) {
        console.error("Error fetching state:", error);
    }
}

// Fetch state every 100ms for real-time updates
setInterval(fetchState, 100);

// Action Sender
async function triggerAction(actionName) {
    try {
        const response = await fetch('/action', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ action: actionName }),
        });
        
        const data = await response.json();
        
        // --- WEB SPEECH API ---
        if (actionName === 'speak' && data.spoken_word) {
            const utterance = new SpeechSynthesisUtterance(data.spoken_word);
            window.speechSynthesis.speak(utterance);
        }

        // Optimistic UI update
        if (data.sentence.length > 0) {
            sentenceEl.textContent = data.sentence;
        } else {
            sentenceEl.textContent = '-';
        }
        
    } catch (error) {
        console.error("Error sending action:", error);
    }
}

// Global Keyboard Listeners
document.addEventListener('keydown', (e) => {
    // Spacebar to add
    if (e.code === 'Space') {
        e.preventDefault(); // prevent scrolling
        triggerAction('add');
    }
    // Backspace to delete
    else if (e.code === 'Backspace') {
        triggerAction('delete');
    }
    // Enter to speak
    else if (e.code === 'Enter') {
        triggerAction('speak');
    }
});
