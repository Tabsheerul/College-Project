// DOM Elements
const currentLetterEl = document.getElementById('current-letter');
const confidenceEl = document.getElementById('confidence');
const badgeEl = document.getElementById('prediction-badge');
const sentenceEl = document.getElementById('current-sentence');
const placeholderEl = document.getElementById('placeholder-text');

// State fetching loop
async function fetchState() {
    try {
        const response = await fetch('/state');
        const data = await response.json();
        
        // Update Live Letter
        if (data.current_letter) {
            currentLetterEl.textContent = data.current_letter;
            confidenceEl.textContent = data.confidence + '%';
            badgeEl.classList.add('detecting');
        } else {
            currentLetterEl.textContent = '-';
            confidenceEl.textContent = '0%';
            badgeEl.classList.remove('detecting');
        }
        
        // Update Sentence
        sentenceEl.textContent = data.current_sentence;
        
        if (data.current_sentence.length > 0) {
            placeholderEl.style.opacity = '0';
        } else {
            placeholderEl.style.opacity = '1';
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
        // Optimistic UI update
        sentenceEl.textContent = data.sentence;
        if (data.sentence.length > 0) {
            placeholderEl.style.opacity = '0';
        } else {
            placeholderEl.style.opacity = '1';
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
        
        // Add a quick flash effect to the badge
        badgeEl.style.transform = 'scale(1.05)';
        setTimeout(() => badgeEl.style.transform = 'scale(1)', 150);
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
