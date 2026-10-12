# RegGuard — Pilot Run Report

**Date**: 2026-10-05  
**Data source**: `results/raw_atoms.jsonl` (parsed 130 unique atoms from 32 questions)  
**Cost source**: results/cost.jsonl  
**Status**: Pre-label — no human annotations, no ground truth, no conformal guarantee yet.

---

## 1. Summary

| Item | Value |
|------|-------|
| Questions in pilot | 32 |
| Total atoms | 130 |
| Accepted atoms | 3 (2.3% coverage) |
| Abstained atoms | 53 |
| Uncertain atoms | 74 |
| Error atoms | 0 |
| Atom types | 1 |
| Strata | 1 |

---

## 2. Cost

| Item | Value |
|------|-------|
| Total cost | $0.00000 |
| Cost per question | $0.00000 |
| Cost per atom | $0.00000 |

> **Note**: results/cost.jsonl

---

## 3. Atom Types

| Atom type | Count | % |
|-----------|-------|---|
| `UNKNOWN` | 130 | 100.0% |

---

## 4. Decisions by Stratum

| Stratum | Count | % |
|---------|-------|---|
| `*|UNKNOWN` | 130 | 100.0% |

---

## 5. Signal Health

| Signal | Mean | Std | Min | Max | Missing | N |
|--------|------|-----|-----|-----|---------|---|
| `s_ret` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0% | 130 |
| `s_ver` | 0.6308 | 0.4156 | 0.0000 | 1.0000 | 0.0% | 130 |
| `s_nli` | N/A | N/A | N/A | N/A | 100.0% | 0 |
| `s_ent` | N/A | N/A | N/A | N/A | 100.0% | 0 |
| `s_div` | N/A | N/A | N/A | N/A | 100.0% | 0 |

> **s_ent** (token entropy) is NaN for all atoms in this pilot — the pipeline ran with stub logprobs (no real vLLM generation yet). This is expected.

---

## 6. Latency Breakdown (mean per atom, seconds)

| Stage | Mean (s) |
|-------|----------|
| retrieve | 0.779 |
| generate | 9.213 |
| extract | 7.611 |
| verify | 14.370 |
| score | 0.183 |
| decide | 0.000 |
| **Total** | **32.156** |

---

## 7. Judge Usage

| Item | Value |
|------|-------|
| Judge calls | 0 / 130 |
| Judge call rate | 0.0% |

> No judge calls in this pilot — all atoms were routed ABSTAINED before reaching the UNCERTAIN band.  
> This is expected: the aggregator is not yet trained on labeled data; default scores push atoms to ABSTAINED.

---

## 8. Sample Atoms

### 8a. Accepted (SUPPORTED / VERIFIED)

**question_id**: `e2fbf445-8edf-455d-8430-bcdf08213d24`  **stratum**: `*|UNKNOWN`  **type**: `UNKNOWN`
**atom**: The aggregate cap on fiscal benefits for each MSME under the extended Interest Equalization Scheme is ₹50 lakhs.
**decision**: `SUPPORTED`  **risk**: `0.2923`  **v1_status**: `MATCH`  **judge**: `-`
**signals** — s_ret: 0.0000  s_nli: NaN  s_div: NaN  s_ent: NaN

---

**question_id**: `9eb1ad56-1fbf-4846-bcc7-aeaef5ce8e49`  **stratum**: `*|UNKNOWN`  **type**: `UNKNOWN`
**atom**: The answer states 'paragraph 1'.
**decision**: `SUPPORTED`  **risk**: `0.2014`  **v1_status**: `MATCH`  **judge**: `-`
**signals** — s_ret: 0.0000  s_nli: NaN  s_div: NaN  s_ent: NaN

---

**question_id**: `e2fbf445-8edf-455d-8430-bcdf08213d24`  **stratum**: `*|UNKNOWN`  **type**: `UNKNOWN`
**atom**: The aggregate cap on fiscal benefits for each MSME under the extended Interest Equalization Scheme is ₹50 lakhs.
**decision**: `SUPPORTED`  **risk**: `0.2923`  **v1_status**: `MATCH`  **judge**: `-`
**signals** — s_ret: 0.0000  s_nli: NaN  s_div: NaN  s_ent: NaN

---

### 8b. Abstained

**question_id**: `79284039-7e7d-4033-912f-a7ab8c656f90`  **stratum**: `*|UNKNOWN`  **type**: `UNKNOWN`
**atom**: The Reserve Bank of India (Commercial Banks – Undertaking of Financial Services) Amendment Directions, 2026 define 'Acqu
**decision**: `ABSTAINED`  **risk**: `0.7597`  **v1_status**: `MISMATCH`  **judge**: `-`
**signals** — s_ret: 0.0000  s_nli: NaN  s_div: NaN  s_ent: NaN

---

**question_id**: `388918f5-7d8b-4c02-82db-da324520431d`  **stratum**: `*|UNKNOWN`  **type**: `UNKNOWN`
**atom**: The answer states '20 November 2008'.
**decision**: `ABSTAINED`  **risk**: `0.8217`  **v1_status**: `MISMATCH`  **judge**: `-`
**signals** — s_ret: 0.0000  s_nli: NaN  s_div: NaN  s_ent: NaN

---

**question_id**: `9cdade8e-f84b-483a-bf0f-a5889f40083b`  **stratum**: `*|UNKNOWN`  **type**: `UNKNOWN`
**atom**: The answer states 'paragraph 5'.
**decision**: `ABSTAINED`  **risk**: `0.7552`  **v1_status**: `MISMATCH`  **judge**: `-`
**signals** — s_ret: 0.0000  s_nli: NaN  s_div: NaN  s_ent: NaN

---

**question_id**: `a6043846-d87e-439d-9a47-bd8ecf408776`  **stratum**: `*|UNKNOWN`  **type**: `UNKNOWN`
**atom**: Chapter IV of the Payment and Settlement Systems Act, 2007 is cited as one of the enabling provisions for issuing the RB
**decision**: `ABSTAINED`  **risk**: `0.7338`  **v1_status**: `MISMATCH`  **judge**: `-`
**signals** — s_ret: 0.0000  s_nli: NaN  s_div: NaN  s_ent: NaN

---

**question_id**: `5d9bd14e-379f-45dd-96b1-c7530d5ba85a`  **stratum**: `*|UNKNOWN`  **type**: `UNKNOWN`
**atom**: The directions regarding Letter of Credit details were issued under section 11(1) of the Foreign Exchange Management Act
**decision**: `ABSTAINED`  **risk**: `0.8198`  **v1_status**: `MISMATCH`  **judge**: `-`
**signals** — s_ret: 0.0000  s_nli: NaN  s_div: NaN  s_ent: NaN

---


## 9. Limitations

This report is **pre-label** and should not be used to draw conclusions about hallucination rate or system reliability.

| Limitation | Impact |
|------------|--------|
| No human labels | Cannot compute hallucination rate, precision, recall, or F1 |
| No Cohen's κ | Inter-annotator agreement unknown |
| No ground truth | Accepted vs. abstained is driven by untrained aggregator (heuristic weights) |
| Conformal guarantee not certified | LTT/Mondrian calibration requires labeled calibration split |
| s_ent all NaN | Token entropy signal unavailable without real vLLM generation |
| All atoms have empty `atom_text` | Raw atoms in this file are from test fixtures, not real pipeline output |
| Only SECTION family questions | Groq daily limit exhausted before cross-family sampling completed |
| cost.jsonl absent | Cost is estimated, not measured from API responses |

---

## 10. Next Steps

1. **Human labeling** — annotators label each atom as `supported / unsupported / unclear` using `scripts/export_labeling_sheet.py`. Minimum 150 atoms needed for calibration.
2. **Compute Cohen's κ** — double-annotate 20% of atoms with `scripts/kappa.py`.
3. **Train aggregator** — `python aggregate/train.py` on the labeled `agg_train` split.
4. **Certify thresholds** — `python -m guarantee.certify --calib_path ...` to run LTT/Mondrian on calibration split.
5. **Re-run pipeline on real questions** — run the full M2→M8 chain on `data/questions_pilot_review.csv` to produce real atoms (not test fixtures).
6. **Get valid Anthropic API key** — for cross-family question drafting (RATE_THRESHOLD, DATE_TIMELINE, ENTITY_APPLICABILITY, NOT_ANSWERABLE) once Groq daily limit resets.

---

*Generated automatically by `scripts/gen_pilot_report.py` on 2026-10-05. No LLM calls were made.*
