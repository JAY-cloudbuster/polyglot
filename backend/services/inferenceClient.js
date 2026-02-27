/**
 * Inference Client — HTTP wrapper for the Python AI service.
 */

const axios = require("axios");
const FormData = require("form-data");

const AI_SERVICE_URL = process.env.AI_SERVICE_URL || "http://localhost:8000";

/**
 * Send audio buffer to AI inference service for prediction.
 * @param {Buffer} audioBuffer - Raw audio file bytes
 * @param {string} filename - Original filename
 * @returns {Promise<Object>} Prediction result: { label, probability, features_used, total_features }
 */
async function sendToInference(audioBuffer, filename = "audio.wav") {
    const form = new FormData();
    form.append("file", audioBuffer, {
        filename,
        contentType: "audio/wav",
    });

    const response = await axios.post(`${AI_SERVICE_URL}/predict`, form, {
        headers: {
            ...form.getHeaders(),
        },
        timeout: 30000, // 30s timeout for inference
        maxContentLength: 10 * 1024 * 1024,
    });

    return response.data;
}

module.exports = { sendToInference };
