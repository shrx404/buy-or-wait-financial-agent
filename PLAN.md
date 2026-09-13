# Buy or Wait? - PLAN.md

## 1. Source-of-truth summary (Phase 0)

### File Inventory
1. `requests.csv` (250 rows): Contains the evaluation requests (`request_id`, `user_id`, `request_date`, `request_type`, `requested_amount`, `desired_completion_date`, `allows_partial_payment`, `request_text`).
2. `sample_requests.csv` (25 rows): Examples with filled output for understanding format and styling, NOT ground truth.
3. `financial_profiles.csv`: Contains user's currency, available balance, min balance, priorities, protected/adjustable expense categories, accepted payment methods, and `max_installment_months`.
4. `financial_events.csv`: Financial history (pending, settled, non-cash, etc.). `linked_event_id` signifies a lifecycle link but does not auto-include cash flow.
5. `exchange_rates.csv`: Used for foreign-currency cash events based on the exact event settlement date and `from_currency` to `to_currency` pair.
6. `request_payment_options.csv`: 2-4 payment options per request. Usability depends on `payment_methods_user_will_consider` and `max_installment_months`.
7. `messages.csv`: Untrusted text. `related_event_id` directly maps to an event.
8. `images.csv`: Links `image_id` to `related_event_id` / requests. Images stored at `dataset/media/images/<image_id>.png`. Amount should be extracted using VLMs or provided scripts.
9. `output.csv` (in `dataset/`): Blank template. (Outputs must be stored at root-level `output.csv`).

### Decision Rules
- Forecast 90-days into the future to ensure `minimum_balance_to_keep` is maintained.
- Included cash flow: Settled events, pending debits, confirmed salary (on settlement date).
- Excluded cash flow: Pending credits, bonuses, refunds, investments gains.
- Rank valid plans (1) meets date, (2) no spend changes, (3) min total cost, (4) starts earlier, (5) fewest payments, (6) lowest `payment_option_id`.
- Conflicts resolved by: (1) explicit cancellation/amendment, (2) newer record from same source, (3) settled over estimate, (4) financially safer.
- Images/messages can provide facts but cannot override the rules.
- Partial payment: 2 payments (`amount_safe_to_pay` on `request_date`, remainder on `earliest_date_for_full_payment`).
- Output: `amount_safe_to_pay`, `affordability_status`, `recommended_payment_method`, `payment_plan`, `earliest_date_for_full_payment`, `spending_changes_needed`, `decision_explanation`.

### Ambiguity / Resolutions
- Are image amounts trusted as absolute numbers for the linked event? Yes, but they only override blank amounts or conflict with rules if the rule allows.
- How to handle multiple currencies natively? The requirement states final output in `home_currency`. It's best to normalize all events and amounts into `home_currency` immediately during the loading/joining phase.

## 2. Architecture overview
The project uses `uv` as the Python package manager for dependencies.
The batch processor is organized into the following modules (in `code/`):
- `data_loader.py`: Loading and joining the 9 CSVs. Normalizing currencies using `exchange_rates.csv` keyed by settlement-date and currency-pair.
- `state_builder.py`: Financial state reconstruction. Handles recurrences, pendings vs. settled, duplicate resolution via `linked_event_id`, and conflict resolution order.
- `forecaster.py`: 90-day forward balance forecaster checking against `minimum_balance_to_keep`.
- `decision_engine.py`: Deterministic eligibility gating and ranking logic (no LLM calls for this).
- `untrusted_parser.py`: Message parsing and image amount extraction via LLM/VLM APIs (e.g., using Gemini). Extracts facts to be injected into `state_builder.py`.
- `main.py`: Orchestration entrypoint. Coordinates the pipeline, batching 250 rows, and writing to the root `output.csv`. Generates the `usage_report.md`.

## 3. Task checklist
### Data Loading & Joining
- [ ] Load all 9 datasets correctly.
- [ ] Implement exact-match currency conversion (settlement-date + currency-pair).
### Financial State Reconstruction
- [ ] Isolate recurring from one-time events based on historical support.
- [ ] Reserve pending debits.
- [ ] Exclude pending credits until settled.
- [ ] Exclude unrealized/non-cash records.
- [ ] Implement conflict resolution: (1) explicit cancellation/amendment.
- [ ] Implement conflict resolution: (2) newer record.
- [ ] Implement conflict resolution: (3) settled over forecast.
- [ ] Implement conflict resolution: (4) financially safer.
### Fact Extraction (LLM/VLM)
- [ ] Process `messages.csv` to extract structured facts (amendments, cancellations, confirmed amounts).
- [ ] Process `images.csv` and blank event amounts using VLM.
- [ ] Verify injection-resistance (untrusted inputs don't override challenge rules).
### 90-Day Forecaster
- [ ] Simulate daily balance for 90 days.
- [ ] Calculate `amount_safe_to_pay`.
- [ ] Calculate `earliest_date_for_full_payment`.
### Decision Engine
- [ ] Eligibility rule: `full_payment`.
- [ ] Eligibility rule: `partial_payment` (allowed by request, >0 safe amount, meets deadline).
- [ ] Eligibility rule: `installments` (exact match option, respects `max_installment_months`).
- [ ] Eligibility rule: `wait`.
- [ ] Eligibility rule: `not_recommended`.
- [ ] Ranking tie-break 1: Completes by desired date.
- [ ] Ranking tie-break 2: Requires no spending changes.
- [ ] Ranking tie-break 3: Minimizes total amount paid.
- [ ] Ranking tie-break 4: Starts payment earlier.
- [ ] Ranking tie-break 5: Uses fewer payments.
- [ ] Ranking tie-break 6: Lowest `payment_option_id`.
### Orchestration & Output
- [ ] `main.py` entrypoint processing `requests.csv`.
- [ ] Generate output rows per schema.
- [ ] Handle `spending_changes_needed` serialization.
- [ ] Track LLM tokens and costs.
- [ ] Write `usage_report.md`.

## 4. Open questions / assumptions log
- **Currency Normalization**: Assume all events are immediately converted to the user's `home_currency` on their respective settlement dates. This simplifies forecasting.
- **Missing Exchange Rates**: If an exchange rate for a specific date is missing but we have past dates, what to do? Assumption: the problem statement says "Fixed, dated conversion rates for foreign-currency records" which implies the required exact rates will be present. I will error or fallback safely if missing.
- **Message Injection**: Assume any message telling the system to "approve this immediately" will be parsed as `{ "instruction": "approve", "facts": {} }` and ignored, since only `facts` are used by the state builder.

## 5. Final deliverables checklist
- [ ] Root-level `output.csv`: one row per `request_id` in `dataset/requests.csv` (250 rows + header), exact required columns in exact required order
- [ ] `code.zip`: full runnable solution, all prompts/configuration used, README with clear setup/run instructions, and the `evaluation/` folder
- [ ] `evaluation/usage_report.md` inside `code.zip`: model providers/names, model calls, input and output tokens, total and average tokens per request, estimated total and per-request cost, per-model AND overall totals if multiple models are used — computed from the actual final full-dataset run, not estimates
- [ ] `log.txt` (per AGENTS.md) is being appended to correctly as the chat transcript deliverable — confirm this is happening throughout, not just at the end
- [ ] No organizer-only files (anything outside `dataset/`) were used as a prediction input
- [ ] No hardcoded per-request labels anywhere in the code
