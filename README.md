# Buy or Wait? Financial Agent

This is the solution for the HackerRank Orchestrate (September 2026) challenge "Buy or Wait?".

## Overview
The AI-powered financial agent evaluates user requests to determine affordability. It:
1. **Reconstructs Financial State:** Processes user profiles, events, and recurring commitments.
2. **LLM Extraction:** Uses Google Gemini (`gemini-2.5-flash` and `gemini-1.5-pro` for confidence-based fallback) to extract accurate financial amounts from receipt images and chat messages.
3. **Decision Engine & Forecaster:** Simulates 90-day cash flows to determine if the user can safely afford the item. It will also try to negotiate payment plans (partial payments or installments) and reduce flexible spending if absolutely necessary.

## Setup & Run Instructions
1. Ensure you have `uv` installed.
2. Set your Google Gemini API key:
   ```bash
   # Windows
   $env:GEMINI_API_KEY="your_api_key_here"
   # macOS/Linux
   export GEMINI_API_KEY="your_api_key_here"
   ```
3. Run the complete pipeline:
   ```bash
   uv run python code/main.py
   ```

## Output
- `output.csv`: The final predictions for all 250 requests.
- `evaluation/usage_report.md`: Contains the token usage stats for the final LLM run.
