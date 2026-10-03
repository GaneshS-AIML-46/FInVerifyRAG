# Tomorrow: First Real API Run

All tooling is built and 684/684 tests pass (tag: `pre-api-ready`).
Follow these steps **in order** the moment the Claude API key is available.

---

## Step 0 — Set the API key and verify connectivity

```powershell
# Set key for this terminal session
$env:ANTHROPIC_API_KEY = "sk-ant-..."   # paste your key

# Quick sanity check (costs ~$0.001)
python - <<'EOF'
import anthropic, os
c = anthropic.Anthropic()
r = c.messages.create(model="claude-haiku-4-5-20251001", max_tokens=16,
                       messages=[{"role":"user","content":"ping"}])
print("OK:", r.content[0].text)
EOF
```

---

## Step 1 — Fill model prices in configs/experiment.yaml

Open `configs/experiment.yaml` and replace the two `null` pairs under `llm_prices`:

```yaml
llm_prices:
  claude-sonnet-5-5:
    input:  <price per 1 M input tokens in USD>   # e.g. 3.0
    output: <price per 1 M output tokens in USD>  # e.g. 15.0
  claude-haiku-4-5-20251001:
    input:  <price per 1 M input tokens in USD>   # e.g. 0.25
    output: <price per 1 M output tokens in USD>  # e.g. 1.25
```

Current prices are at https://www.anthropic.com/pricing (API tab).

After filling in, verify estimate_cost.py no longer shows PRICE MISSING:

```powershell
python scripts/estimate_cost.py
```

---

## Step 2 — Dry run (zero API calls, inspect prompts)

```powershell
python scripts/draft_questions.py --dry-run
```

Expected: prints 5 prompts and exits. No files written to `data/`.

---

## Step 3 — Pilot draft (50 questions, ~$1–2)

```powershell
python scripts/draft_questions.py --pilot
```

This writes:
- `data/questions_pilot_review.csv`  — 50 rows for your manual review
- `results/cost.jsonl`               — running cost log
- `results/draft_checkpoint.jsonl`   — checkpoint (safe to resume if interrupted)

Review `data/questions_pilot_review.csv` now. Check:
- `evidence_quote` is a verbatim substring of the chunk
- `family` matches the question content
- `answerable=no` rows are genuinely unanswerable
- No template-style or self-referential questions

If any row looks wrong, note it in `data/questions_pilot_review.csv` under the `notes` column
and re-run with a tweaked prompt (edit `scripts/draft_questions.py` → `_build_prompt()`).

When satisfied, say **QUESTIONS OK** to the agent.

---

## Step 4 — Full draft (300–500 questions, ~$5–8)

Only run after you've said QUESTIONS OK on the pilot.

```powershell
python scripts/draft_questions.py
```

Outputs: `data/questions_draft.csv`

---

## Step 5 — Finalize and validate

```powershell
python scripts/finalize_questions.py data/questions_draft.csv
```

Expected outputs:
- `data/questions.jsonl`
- `data/questions_pilot.jsonl`
- `data/gold_chunks.jsonl`

Any template question or missing column raises immediately — fix and re-run.

---

## Step 6 — Export labeling sheet for annotator

```powershell
python scripts/export_labeling_sheet.py \
  --input data/questions_pilot.jsonl \
  --output data/labeling/pilot_sheet_annotator1.csv
```

Send this CSV to annotator 1 (and annotator 2 for ~20% overlap).
**Do not send** the full `questions.jsonl` — it contains `gold_answer_short` and signals.

---

## Reminders

- Never commit `.env` or files containing `sk-ant-`.
- `results/` must contain only real pipeline outputs — synthetic data stays in `results_mock/`.
- Budget caps: question_drafting $8, pilot $10, total $70 (configured in experiment.yaml).
- If interrupted, re-run `draft_questions.py` — it resumes from checkpoint.
- Tag after QUESTIONS OK: `git tag questions-ok-v1 && git push --tags`
