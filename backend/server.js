/**
 * Polyglot Ghost — Backend API Gateway
 * Express server that orchestrates between frontend and AI inference service.
 */

require("dotenv").config();
const express = require("express");
const cors = require("cors");
const morgan = require("morgan");

const analyzeRoutes = require("./routes/analyze");
const livenessRoutes = require("./routes/liveness");
const healthRoutes = require("./routes/health");

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(morgan("dev"));
app.use(express.json());

// Routes
app.use("/analyze", analyzeRoutes);
app.use("/liveness", livenessRoutes);
app.use("/health", healthRoutes);

// Root
app.get("/", (req, res) => {
  res.json({
    service: "polyglot-ghost-backend",
    version: "1.0.0",
    endpoints: ["/analyze", "/liveness", "/health"],
  });
});

// Global error handler
app.use((err, req, res, next) => {
  console.error("[ERROR]", err.message);
  res.status(err.status || 500).json({
    error: err.message || "Internal server error",
  });
});

app.listen(PORT, () => {
  console.log(`\n🔮 Polyglot Ghost Backend running on port ${PORT}`);
  console.log(`   AI Service URL: ${process.env.AI_SERVICE_URL || "http://localhost:8000"}`);
  console.log(`   Health check: http://localhost:${PORT}/health\n`);
});

module.exports = app;
