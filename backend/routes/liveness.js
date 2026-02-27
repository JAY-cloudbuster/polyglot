/**
 * POST /liveness — Semantic liveness check via optional reasoning service.
 */

const express = require("express");
const axios = require("axios");
const router = express.Router();

router.post("/", async (req, res, next) => {
    try {
        const { text, prompt } = req.body;

        if (!text) {
            return res.status(400).json({ error: "Missing 'text' field" });
        }

        const featherlessUrl = process.env.FEATHERLESS_API_URL;
        const featherlessKey = process.env.FEATHERLESS_API_KEY;

        // If Featherless AI is configured, use it for semantic analysis
        if (featherlessUrl && featherlessKey) {
            try {
                const response = await axios.post(
                    featherlessUrl,
                    {
                        model: process.env.FEATHERLESS_MODEL || "meta-llama/Meta-Llama-3.1-8B-Instruct",
                        messages: [
                            {
                                role: "system",
                                content:
                                    "You are a liveness verification system. Evaluate if the user's spoken response is semantically relevant to the given prompt. Return a JSON object with: relevance_score (0-1), is_live (boolean), reasoning (string).",
                            },
                            {
                                role: "user",
                                content: `Prompt given to user: "${prompt || "Please say your name and today's date."}"\nUser's spoken response: "${text}"\n\nEvaluate the semantic relevance.`,
                            },
                        ],
                        max_tokens: 200,
                        temperature: 0.1,
                    },
                    {
                        headers: {
                            Authorization: `Bearer ${featherlessKey}`,
                            "Content-Type": "application/json",
                        },
                        timeout: 15000,
                    }
                );

                const aiContent = response.data.choices?.[0]?.message?.content || "";

                // Try to parse JSON from AI response
                try {
                    const parsed = JSON.parse(aiContent);
                    return res.json({
                        source: "featherless_ai",
                        ...parsed,
                    });
                } catch {
                    return res.json({
                        source: "featherless_ai",
                        relevance_score: 0.5,
                        is_live: true,
                        reasoning: aiContent,
                    });
                }
            } catch (aiError) {
                console.warn("[LIVENESS] Featherless AI call failed, falling back:", aiError.message);
            }
        }

        // Fallback: basic heuristic liveness check
        const wordCount = text.trim().split(/\s+/).length;
        const hasContent = wordCount >= 2;
        const relevanceScore = Math.min(wordCount / 10, 1.0);

        res.json({
            source: "heuristic",
            relevance_score: parseFloat(relevanceScore.toFixed(2)),
            is_live: hasContent,
            reasoning: hasContent
                ? `Response contains ${wordCount} words, suggesting a live speaker.`
                : "Response too short to determine liveness.",
        });
    } catch (err) {
        next(err);
    }
});

module.exports = router;
