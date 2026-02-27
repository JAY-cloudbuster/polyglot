/**
 * GET /health — Health check endpoint for deployment.
 */

const express = require("express");
const router = express.Router();

router.get("/", (req, res) => {
    res.json({
        status: "ok",
        service: "polyglot-ghost-backend",
        timestamp: new Date().toISOString(),
        uptime: process.uptime(),
    });
});

module.exports = router;
