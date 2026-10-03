# RegGuard — Annotation Guideline (v1.0 · 2026-10-03)

> This document is frozen once Phase E begins.  Changes before that point
> must be tracked with a version bump and approved by the PI.

---

## 1  Scope

**Scope:** faithfulness to the cited source snapshot as of **2026-10-03**.
Legal currency and supersession are **out of scope**.

Annotators evaluate whether a system-generated answer atom is supported by
the retrieved source passage — not by current law, not by any more recent
circular, and not by general regulatory knowledge.

---

## 2  Annotator blinding

Annotators **must not** see the system's decision (accept / abstain / reject),
its risk score, or any signal values before submitting their label.  The
annotation tool must display only: the question, the atom, and the cited
source passage(s).

---

## 3  Label definitions

| Label | Meaning |
|---|---|
| **supported** | The cited source passage explicitly and unambiguously supports the atom. |
| **unsupported** | The cited source passage contradicts the atom, or the atom makes a claim not present in the passage. |
| **unclear** | The passage is ambiguous, partially relevant, or the atom cannot be cleanly mapped to the passage. |

Use **unclear** when in genuine doubt.  Do not guess.

---

## 4  Superseded and outdated circulars

**Rule (frozen):** Judge each atom against the cited source passage as
given, **not** against current law.

If the retrieved passage is from a circular that has been superseded since
2026-10-03, annotate normally: if the passage supports the atom, label it
`supported`.  Do not penalise the atom for citing an older circular — that
is a retrieval quality issue, not an annotation issue.

Exception: if the question explicitly asks about current regulatory
requirements AND the cited passage is explicitly marked as superseded within
the passage text itself (e.g., "this direction stands withdrawn"), then the
atom may be labelled `unsupported` because the passage itself signals
invalidity.

---

## 5  Numeric thresholds and rates

Copy the exact figure from the passage.  If the passage says "not less than
15 per cent", the atom must say "15 per cent" or equivalent — rounding or
paraphrase that changes the numeric meaning is `unsupported`.

---

## 6  Atoms about applicability / entity scope

Check that the passage explicitly names or describes the entity category.
General implication is not enough — the passage must state or clearly
enumerate the category for the atom to be `supported`.

---

## 7  NOT_ANSWERABLE questions

Some questions are intentionally not answerable from the corpus (family
`NOT_ANSWERABLE`).  For these, any atom that claims to answer the question
should be labelled `unsupported` unless the passage genuinely and explicitly
answers it (which would indicate a question-drafting error to flag).

---

## 8  Inter-annotator agreement

Target Cohen's κ ≥ 0.70 on the pilot batch (Phase D).  Disagreements are
resolved by majority vote among three annotators.  If κ < 0.70, the
guideline is refined before Phase F.

---

## 9  What to flag for PI review

- Atom is about a date or rate that the annotator believes is wrong but
  the passage says it (possible OCR or indexing error).
- Question appears unanswerable but was labelled with `supported` by the
  system — flag the question_id.
- Source passage appears truncated or garbled.
