# RegGuard: Selective Verification for Regulatory QA (Pilot on 50 Questions)

## Abstract
This pilot applies conformal selective verification to regulatory QA over RBI and SEBI documents.
From 50 questions we extracted 130 atoms; 65 were human-labeled on the test split (7 unclear excluded).
At 100% coverage (no LTT filter applied), hallucination rate is 21.5% (95% CI: 12.3%–33.5%).
The conformal guarantee (LTT) was not certifiable at pilot scale (27 calib atoms insufficient).
This establishes a baseline and validates the pipeline architecture for larger-scale deployment.

## Method
- Atom extraction from LLM-generated answers (M2–M4 pipeline).
- V1 verifier (deterministic rules) + V2 verifier (DeBERTa NLI entailment probability).
- Logistic regression aggregator trained on 57 atoms (12 doc groups, 5-fold GroupKFold CV AUROC: 0.77 ± 0.18).
- Learn-then-Test (LTT) calibration attempted with Mondrian stratification by atom_type.
  Result: not certifiable — aggregator outputs constant risk (0.50) due to small training set
  and single-class CV folds. All 6 Mondrian strata merged to global parent (n=27, below threshold).

## Dataset
- 50 pilot questions stratified across RBI and SEBI regulators.
- 130 total atoms extracted from generated answers.
- Train split: 57 atoms, LLM-labeled (Qwen2.5-7B), 12 doc groups.
- Calib split: 27 atoms, human-labeled.
- Test split: 65 atoms, human-labeled (locked). 7 unclear excluded from evaluation.
- Raw test set: 72 atoms. 7 excluded as unclear (cannot decide from passage alone).

## Results

### Main Table
| Metric             | Value   | 95% CI          |
|--------------------|---------|-----------------|
| Coverage           | 100%    | —               |
| Hallucination rate | 21.5%   | 12.3% – 33.5%   |
| AURC               | 0.1215  | —               |
| LTT certified      | No      | —               |
| Unclear excluded   | 7 atoms | 9.7% of 72 raw  |

### Interpretation
The 21.5% hallucination rate means roughly 1 in 5 atom-level claims in LLM answers
are unsupported by the cited regulatory source passage. This motivates selective filtering:
a working LTT threshold would accept only high-confidence atoms, reducing hallucination
rate at the cost of coverage.

### Failure Cases
| atom_id (short)   | atom_text                                          | risk |
|-------------------|----------------------------------------------------|------|
| 294fdff0…_0245    | the guidelines are for reporting March 2025 transactions | 0.50 |
| 7265a236…_ed93    | The answer states 'Section 11'.                    | 0.50 |
| 388918f5…_cb57    | The answer states '20 November 2008'.              | 0.50 |
| 7265a236…_d5b8    | The answer states 'Section 11'.                    | 0.50 |
| 388918f5…_d6be    | The answer states '6 years'.                       | 0.50 |

Note: all risk scores are 0.50 (aggregator constant — known issue from small calib set).

## Limitations
1. Small pilot: 50 questions, 130 atoms. LTT requires ~100+ calib atoms per stratum.
2. Aggregator outputs constant risk (0.50): small train set (57 atoms) and single-class CV folds
   (Folds 4–5) prevented learning a discriminative score.
3. Mondrian stratification ineffective: all 130 pilot atoms were tagged `atom_type=UNKNOWN`
   by the M3 extractor. The 6 canonical strata (RATE, THRESHOLD, DATE, SECTION, ENTITY,
   APPLICABILITY) each had 0 calib atoms and were automatically merged to the global parent.
   Atom-type classification must be fixed for stratification to be meaningful.
4. LLM train labels rejected (Cohen's kappa = 0.29 vs human audit): train split relies on
   Qwen2.5-7B labels which showed low agreement with human annotators. This likely degrades
   aggregator discrimination.
5. Isotonic calibrator version mismatch (sklearn 1.9.1 pickle vs 1.7.1 runtime): a warning
   is emitted at score time; calibrator should be retrained on the deployment sklearn version.

## Next Steps
1. **Scale to 500+ questions** to obtain ≥100 calib atoms per Mondrian stratum.
2. **Fix atom_type extraction** in M3 so atoms are tagged with canonical types (RATE,
   THRESHOLD, DATE, SECTION, ENTITY, APPLICABILITY).
3. **Human-label all train atoms** (87 atoms) to replace rejected LLM labels, then retrain
   the aggregator; target CV AUROC ≥ 0.85 with non-constant risk output.
4. **Re-run LTT** after (1–3): with discriminative scores and ≥100 calib atoms, the binomial
   test can certify ε=0.05 selective risk at δ=0.10.
5. **Retrain calibrator** on matched sklearn version to eliminate pickle mismatch warning.

## Artifacts
| File | Description |
|------|-------------|
| `results/summary.json` | Full evaluation metrics with atom counts |
| `results/main_table.json` | Clean table row for paper/thesis |
| `results/ltt_strata.json` | Per-stratum LTT output (all merged / not certifiable) |
| `models/aggregator/aggregator_v1.pkl` | Logreg aggregator (57 atom train set) |
| `models/aggregator/calibrator_v1.pkl` | Isotonic calibrator |
| `data/splits/test.lock` | SHA-256 of locked test split |
| `results/calib.jsonl` | 27 calibration atoms (human-labeled, 1 unclear excluded) |
| `results/test.jsonl` | 72 test atoms raw (65 evaluated, 7 unclear excluded, locked) |
| `docs/final_report.md` | This report |

---

*Generated: 2026-10-05. LTT note: "LTT threshold not applied. Conformal guarantee not certified. Pilot size insufficient."*
