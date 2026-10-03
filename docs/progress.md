# RegGuard Pipeline Progress

## Phase Checklist

| Phase | Status | Date | Commit | Notes |
|-------|--------|------|--------|-------|
| A: Verify current build | ✅ DONE | 2026-10-03 | pipeline-ready-v1 | 644/644 tests pass; check_publishable correctly FAILs; s_mech clean; strict guards work |
| B: Document inventory and splits | ⬜ TODO | | | STOP: needs INVENTORY OK from human |
| C: Draft questions | ⬜ TODO | | | STOP: needs QUESTIONS OK from human |
| D: Pilot run (50 questions) | ⬜ TODO | | | STOP: needs PILOT OK from human |
| E: Labeling support | ⬜ TODO | | | STOP: needs PILOT LABELS DONE from human |
| F: Full run | ⬜ TODO | | | STOP: needs LABELS DONE from human |
| G: Finalize data, train, apply rules | ⬜ TODO | | | STOP: needs DECISIONS OK from human |
| H: Calibration, baselines, final eval | ⬜ TODO | | | STOP: needs APPROVE FINAL TEST RUN from human |

---

## Phase A — Detail

**Status**: DONE (2026-10-03)

### Fixes applied (all small, targeted)

1. **`aggregate/model.py`**: Handle `UnpicklingError` when pkl file is corrupt — return `None` (fallback) instead of crashing.

2. **`guarantee/ltt.py`**:
   - Added `use_hb: bool = True` param to `ltt_threshold()` — switches between `hb_pvalue` and `binom_pvalue`.
   - Added `n_grid` and `use_hb` params to `ltt_thresholds_pair()`.
   - Added `target_delta` and `mean_lambda` keys to `simulate_violation_rate()` return dict.
   - Fixed `build_mondrian_thresholds()` label parsing: now treats `1`, `"1"`, `"unsupported"`, `"outdated"`, `True` all as wrong.

3. **`guarantee/simulate.py`**: Fixed crash accessing `result["mean_lambda"]` before it was set by `simulate_violation_rate`.

4. **`guarantee/certify.py`**: Full rewrite. Now accepts `n_min`, `run_simulation`, `eps_abstain`, `strategy`, `window_months`, `half_life_days`, `reference_date`, `n_grid`, `strict`. Sets `stratum_name` and `abstain_above` on `Thresholds` objects. `load_thresholds()` now backward-compatible.

5. **`models/guarantee/thresholds.json`**: Replaced corrupt placeholder with proper schema: `*|*`, `accept_below=null`, `n_accepted=0`. The `check_publishable` gate and `run_all` now work correctly with it.

6. **`eval/run_all.py`**:
   - Fixed `KeyError: main_table` when `skip_baselines=True` (line 371: use `in` check before indexing).
   - Changed data loading to NOT do annotator validation at evaluation time (signal/M7 checks fire first, annotator check belongs at label-ingestion time, not evaluation time).

7. **`eval/check_publishable.py`**: Added `sys.path.insert(0, str(ROOT))` so `tests` package is importable when running standalone.

### Test results

- **644 passed, 0 failed** (was 604 passed, 40 failed before fixes)
- `check_publishable.py`: correctly prints NOT PUBLISHABLE with 13 specific [FAIL] reasons
- Strict mode raises `StrictFailure` for missing aggregator ✅
- Strict mode raises `StrictFailure` for `synthetic=True` data ✅
- `s_mech` scan: zero occurrences in main pipeline code (only guard-rejection checks in `common/validation.py`) ✅

### Evidence

```
644 passed in 549.17s (0:09:09)
```

check_publishable.py output (abbreviated):
```
NOT PUBLISHABLE
  [FAIL] paper_valid is false (simulated data, fallbacks, or strict failures present)
  [FAIL] Test split has 0 atoms, need >= 500
  [FAIL] Inter-annotator kappa not reported (results/kappa.json missing)
  [FAIL] leakage_report.json missing (must pass document-level leakage check)
  [FAIL] Stratum *|* has None threshold (must be certified or merged)
  [FAIL] Stratum *|* has 0 coverage
  [FAIL] Clopper-Pearson intervals for violation rate missing in seeds_summary
  [FAIL] Baselines do not have distinct AURC (or missing baseline data)
  [FAIL] Only 0 seeds run, need 5
  [FAIL] Missing interval for hallucination_rate in seeds_summary
  [FAIL] Missing interval for coverage in seeds_summary
  [FAIL] Missing interval for aurc in seeds_summary
  [FAIL] Missing interval for violation_rate in seeds_summary
```

---

## Next Step: Phase B

Build `scripts/make_doc_inventory.py` and `scripts/assign_doc_splits.py`.
No human action needed before Phase B.
