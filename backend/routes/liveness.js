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

        const groqUrl = process.env.GROQ_API_URL || "https://api.groq.com/openai/v1/chat/completions";
        const groqKey = process.env.GROQ_API_KEY;

        // If Groq AI is configured, use it for semantic analysis
        if (groqUrl && groqKey) {
            try {
                const response = await axios.post(
                    groqUrl,
                    {
                        model: process.env.GROQ_MODEL || "llama3-8b-8192",
                        messages: [
                            {
                                role: "system",
                                content:
                                    "You are a liveness verification system. Evaluate if the user's spoken response is semantically relevant to the given prompt. Return a JSON object with: relevance_score (0-1), is_live (boolean), reasoning (string).",
                            },
                            {
                                role: "user",
                                content: `Prompt given to user: "${prompt || "Please say your name and today's date."}"\\nUser's spoken response: "${text}"\\n\\nEvaluate the semantic relevance.`,
                            },
                        ],
                        max_tokens: 200,
                        temperature: 0.1,
                    },
                    {
                        headers: {
                            Authorization: `Bearer ${groqKey}`,
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
                        source: "groq_ai",
                        ...parsed,
                    });
                } catch {
                    return res.json({
                        source: "groq_ai",
                        relevance_score: 0.5,
                        is_live: true,
                        reasoning: aiContent,
                    });
                }
            } catch (aiError) {
                console.warn("[LIVENESS] Groq AI call failed, falling back:", aiError.message);
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
