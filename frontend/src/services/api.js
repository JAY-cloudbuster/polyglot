/**
 * API service — communicates with backend.
 */
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:3001';
/**
 * Send audio blob to /analyze endpoint.
 * @param {Blob} audioBlob
 * @returns {Promise<Object>} { verdict, confidence, features_analyzed, feature_breakdown, timestamp }
 */
export async function analyzeAudio(audioBlob) {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');

    const response = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const err = await response.json().catch(() => ({ error: 'Network error' }));
        throw new Error(err.error || err.detail || `Server error (${response.status})`);
    }

    return response.json();
}

/**
 * Send text to /liveness endpoint for semantic check.
 * @param {string} text - User's spoken response
 * @param {string} prompt - The prompt given to the user
 * @returns {Promise<Object>} { relevance_score, is_live, reasoning, source }
 */
export async function checkLiveness(text, prompt) {
    const response = await fetch(`${API_BASE}/liveness`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, prompt }),
    });

    if (!response.ok) {
        const err = await response.json().catch(() => ({ error: 'Network error' }));
        throw new Error(err.error || `Server error (${response.status})`);
    }

    return response.json();
}

/**
 * Check backend health.
 * @returns {Promise<Object>}
 */
export async function checkHealth() {
    const response = await fetch(`${API_BASE}/health`);
    return response.json();
}
