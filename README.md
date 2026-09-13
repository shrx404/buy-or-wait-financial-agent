# Buy or Wait? Financial Agent

This repository contains the solution for the **HackerRank Orchestrate (September 2026)** challenge: *Buy or Wait?*

The system acts as an AI-powered financial agent that evaluates purchase requests against a user's financial profile. It reconstructs their financial position, parses untrusted multimodal context (messages and receipts), and simulates 90-day cash flow to determine if a purchase is affordable.

---

## 🏗️ System Architecture

The pipeline is highly modularized, with each component handling a specific phase of the financial evaluation.

| Component | Module | Description |
| :--- | :--- | :--- |
| **Data Loader** | `code/data_loader.py` | Loads and structures the various CSV datasets, profiles, and exchange rates. |
| **Untrusted Parser** | `code/untrusted_parser.py` | Leverages Google Gemini (`2.5-flash` with `1.5-pro` fallback) to extract financial facts from chat messages and receipt images. Implements concurrency, caching, and token tracking. |
| **State Builder** | `code/state_builder.py` | Reconstructs the user's financial history. Resolves duplicate/conflicting transactions and extracts recurring, fixed, and flexible spending patterns. |
| **Forecaster** | `code/forecaster.py` | Generates a daily projected cash flow timeline over a 90-day window. Uses caching to simulate if candidate payment plans will breach the minimum balance threshold. |
| **Decision Engine** | `code/decision_engine.py` | Ranks available payment methods (Full, Partial, Installments, Wait). Automatically negotiates reductions in flexible spending categories to afford a purchase if needed. |
| **Orchestrator** | `code/main.py` | The main entry point. Orchestrates data ingestion, runs the pipeline for all 250 requests, and writes the output payload. |

---

## 🧠 Core Logic & Strategy

The agent strictly follows conservative financial rules to protect the user's cash reserves.

### 1. Conflict Resolution (State Builder)
When encountering linked or conflicting historical events, the system prioritizes truth based on the following fallback rules:
*   **Explicit Facts:** Amendments/cancellations parsed directly from LLM image or message analysis.
*   **Timestamp:** Newer records from the same source supersede older estimates.
*   **Settlement Status:** Settled events override pending or scheduled forecasts.
*   **Financial Safety:** When in doubt, the system assumes the more conservative financial impact (highest debit, lowest credit).

### 2. Payment Plan Ranking (Decision Engine)
If a user submits a purchase request, the agent ranks the affordability methods based on cost and safety:
1.  **Full Payment:** (Best) Safe to pay everything upfront today.
2.  **Partial Payment:** Complete a portion today, and the rest before the deadline.
3.  **Installments:** Break it down according to provider options.
4.  **Wait:** Delay the purchase until sufficient cash flow is recovered.
5.  **Not Recommended:** Unsafe to proceed within the 90-day forecast.

### 3. Spending Adjustments
If a purchase is initially unaffordable, the agent scans future forecasted expenses for *flexible* and *unprotected* categories. It attempts to simulate reducing these expenses (by 50%) or stopping them entirely to see if it makes the requested payment plan safe.

---

## 🚀 Setup & Execution

### Prerequisites
*   Python 3.10+
*   [`uv`](https://docs.astral.sh/uv/) (Astral's fast Python package manager)

### Installation & API Keys
1. Clone the repository and navigate to the project root.
2. Set your Google Gemini API key as an environment variable:

```bash
# On Windows (PowerShell)
$env:GEMINI_API_KEY="your_api_key_here"

# On macOS/Linux
export GEMINI_API_KEY="your_api_key_here"
```

### Running the Pipeline
Simply run the orchestrator script using `uv`:

```bash
uv run python code/main.py
```
*Note: Due to multi-threading and API rate limits, the full suite of 250 requests will process dynamically and leverage LLM caching to avoid redundant calls.*

---

## 📁 Output Artifacts

Upon completion, the system produces the required submission files:

| File | Purpose |
| :--- | :--- |
| `output.csv` | The final predictions for all request evaluations, matching the challenge schema. |
| `evaluation/usage_report.md` | Contains the token usage stats and estimated inference costs for the final run. |
| `log.txt` | The unified agent transcript for the session, appended automatically. |
