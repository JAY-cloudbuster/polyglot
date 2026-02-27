/**
 * Analysis Controller — Orchestrates audio analysis pipeline.
 */

const { sendToInference } = require("../services/inferenceClient");

async function analyzeAudio(req, res, next) {
    try {
        if (!req.file) {
            return res.status(400).json({ error: "No audio file provided" });
        }

        console.log(
            `[ANALYZE] Received audio: ${req.file.originalname || "recording"} ` +
            `(${req.file.mimetype}, ${(req.file.size / 1024).toFixed(1)} KB)`
        );

        // Forward to AI inference service
        const prediction = await sendToInference(req.file.buffer, req.file.originalname || "audio.wav");

        // Build response
        const response = {
            verdict: prediction.label,
            confidence: prediction.probability,
            reasoning: prediction.reasoning || null,
            features_analyzed: prediction.total_features,
            feature_breakdown: prediction.features_used || prediction.measurements,
            timestamp: new Date().toISOString(),
        };

        console.log(`[ANALYZE] Verdict: ${response.verdict} (${(response.confidence * 100).toFixed(1)}%)`);

        res.json(response);
    } catch (err) {
        console.error("[ANALYZE] Error:", err.message);

        if (err.response) {
            // Error from AI service
            return res.status(err.response.status || 502).json({
                error: "AI service error",
                detail: err.response.data?.detail || err.message,
            });
        }

        if (err.code === "ECONNREFUSED") {
            return res.status(503).json({
                error: "AI service unavailable",
                detail: "Cannot connect to inference service. Is it running?",
            });
        }

        next(err);
    }
}

module.exports = { analyzeAudio };
