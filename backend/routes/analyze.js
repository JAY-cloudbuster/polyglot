/**
 * POST /analyze — Accept audio, forward to AI service, return verdict.
 */

const express = require("express");
const multer = require("multer");
const { analyzeAudio } = require("../controllers/analysisController");

const router = express.Router();

// Configure multer for in-memory file storage
const upload = multer({
    storage: multer.memoryStorage(),
    limits: { fileSize: 10 * 1024 * 1024 }, // 10 MB
    fileFilter: (req, file, cb) => {
        const allowed = [
            "audio/wav", "audio/x-wav", "audio/wave",
            "audio/mpeg", "audio/mp3", "audio/ogg",
            "audio/webm", "audio/flac",
            "application/octet-stream",
        ];
        if (allowed.includes(file.mimetype)) {
            cb(null, true);
        } else {
            cb(new Error(`Unsupported audio format: ${file.mimetype}`), false);
        }
    },
});

router.post("/", upload.single("audio"), analyzeAudio);

module.exports = router;
