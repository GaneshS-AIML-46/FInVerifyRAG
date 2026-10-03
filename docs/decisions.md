# RegGuard — Frozen Design Decisions

> These decisions are frozen. Do not change thresholds, labels, or rules
> to improve results. Any change requires explicit approval and a new entry here.

---

## D-01  Temporal scope  (added 2026-10-03)

**Decision:** All evaluation is scoped to a single corpus snapshot dated
**2026-10-03**.  Every question carries `query_date = 2026-10-03`.  The
retrieval date-filter is retained in the code but will pass all inventory
documents because every document's `effective_from` ≤ snapshot date after
inventory curation.

**Rationale:** Legal currency and supersession tracking are out of scope
for this research prototype.  Fixing a single snapshot date removes the need
to track which circular supersedes which and avoids contamination of the
calibration set by documents that were retrospectively withdrawn.

**Consequences:**
- `effective_from` values in `doc_inventory.csv` are kept as-is (not modified).
- The `detect_supersession.py` output (`data/supersession_proposed.csv`) is
  produced for reference only; it does NOT modify the inventory automatically.
- The question family "superseded / date-dependent" is removed.  New mix:
  RATE_THRESHOLD 30 % / SECTION 20 % / DATE_TIMELINE 15 % /
  ENTITY_APPLICABILITY 25 % / NOT_ANSWERABLE 10 %.

---

## D-02  Annotation scope  (added 2026-10-03)

**Decision:** Scope is **faithfulness to the cited source snapshot as of
2026-10-03**.  Legal currency and supersession are **out of scope**.

**Annotator instruction (frozen):** Judge each atom against the cited source
passage as given, not against current law.  An atom is `supported` if the
cited passage, taken verbatim, supports the claim — regardless of whether
the underlying circular has been superseded since the snapshot date.

---

## D-03  Signals in main pipeline  (carried over from Phase A)

Active signals: `s_ret`, `s_ver` (categorical), `s_nli`, `s_ent`, `s_div`.
`s_mech` is removed from the main pipeline; it lives only in
`experiments/drift/` and is blocked by strict-mode guards.

---

## D-04  Conformal guarantee parameters  (carried over from Phase A)

ε_accept = 0.05,  ε_abstain = 0.20,  δ = 0.10.
Mondrian strata = (atom_type, source_regulator).
These are frozen; do not change to improve results.