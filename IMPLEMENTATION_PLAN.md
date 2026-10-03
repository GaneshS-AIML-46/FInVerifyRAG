# RegGuard: Complete Module-wise Implementation Plan

How to use this file: work through the modules in the order given in Section 2. Every module has the same layout: **Goal, Inputs/Outputs, Components, Tasks (checklist), Key code, Config, Tests, Done when, Pitfalls**. Tick the checkboxes as you go. Data contracts (`Chunk`, `Atom`, ...) are defined in the README under "Integration contracts" and live in `common/schemas.py`.

---

## 1. Global conventions

| Item | Rule |
|------|------|
| Python | 3.10+, type hints everywhere, `pydantic` or `dataclasses` for schemas |
| Config | One YAML per module in `configs/`, one `pipeline.yaml` that references them |
| Seeds | `seed: 42` in `pipeline.yaml`, set for `random`, `numpy`, `torch` |
| Storage | JSONL for records, Parquet for big tables, FAISS/Qdrant for vectors |
| Logging | `logging` + Weights & Biases or MLflow for experiments |
| Tests | `pytest`, fixtures in `tests/fixtures/<module>/` |
| Branching | One branch per module, PR review before merge |
| Data splits | `agg_train` / `calibration` / `test`, split by **question family or document**, never by atom |
| Pinning | Model names + revisions, prompt versions, corpus snapshot date in `pipeline.yaml` |

### Order of work and dependency graph

```
M0 common ──► M1 ingest ──► M2 retrieval ──► M3 generation ──► M4 atoms ──► M5 verify ──► M6 signals+aggregate ──► M7 guarantee ──► M8 judge+decision ──► M9 app
                    └────────────────────────── M10 benchmark + eval (runs in parallel from week 2) ─────────────────────────────┘
```

---

## 2. Models used and where they are trained

| # | Model | Module | Role | Train? |
|---|-------|--------|------|--------|
| 1 | Qwen2.5-7B-Instruct | M3, M4, M6 | Generator, atom extractor, logprob/attention signals | Frozen (optional LoRA for M4) |
| 2 | DeBERTa-v3-large (MNLI init) | M5 | **V2 verifier (NLI)** | **Fine-tune (required)** |
| 3 | BGE-M3 (or E5) | M2 | Dense retriever | Optional |
| 4 | bge-reranker-v2-m3 | M2 | Reranker | Optional |
| 5 | Qwen2.5-72B or API LLM | M8 | **V3 judge** (UNCERTAIN atoms only) | Frozen |
| 6 | LogReg / LightGBM | M6 | Multi-signal aggregator | Trained |

Training order matters: **NLI (M5) first, then aggregator (M6), then calibration (M7)**, each on disjoint data.

---

## M0: `common/` (shared foundation)

**Goal:** everything other modules share, so nobody re-invents it.

**Components**
- `schemas.py`: all dataclasses (`Chunk`, `RetrievalResult`, `GeneratedAnswer`, `Atom`, `VerifiedAtom`, `ScoredAtom`, `Thresholds`, `Decision`).
- `config.py`: loads YAML, merges with CLI overrides, validates.
- `io.py`: `read_jsonl`, `write_jsonl`, `save_parquet`, dataclass <-> dict.
- `logging.py`: logger + W&B init.
- `dates.py`: parse and compare Indian date formats (`12 March 2024`, `12/03/2024`, `FY 2023-24`).
- `llm_client.py`: OpenAI-compatible client wrapper for vLLM and API models, with retries and a disk cache keyed on (model, prompt hash).

**Tasks**
- [ ] Write and freeze schemas (add `schema_version`).
- [ ] Config loader with unit tests.
- [ ] LLM client with caching (saves a lot of compute during experiments).
- [ ] `scripts/run_pipeline.py` skeleton that chains module interface functions.
- [ ] Fixture folder with one example object per schema.

**Done when:** every other module can `from common.schemas import ...` and round-trip objects to JSON.

---

## M1: `ingest/` (corpus, chunks, version graph)

**Goal:** a clean, time-aware corpus.

**Inputs:** official websites and PDFs. **Outputs:** `data/processed/chunks.jsonl` (list of `Chunk`), `data/processed/version_graph.json`.

**Components**
| File | Purpose |
|------|---------|
| `download/sebi.py`, `rbi.py`, `incometax.py`, `mf.py` | Scrapers with polite rate limiting, save raw PDF + metadata |
| `parse/pdf_parser.py` | PyMuPDF/pdfplumber text extraction; OCR fallback (Tesseract) for scans |
| `parse/clean.py` | Remove headers/footers, fix hyphenation, normalize unicode and rupee symbols |
| `chunk/legal_chunker.py` | Split by regulation, section, clause, sub-clause; keep parent path (`Reg 52 > (4) > (a)`) |
| `chunk/metadata.py` | Extract issue date, effective dates, supersession text ("in supersession of circular ...") |
| `version_graph/build.py` | Nodes = documents/sections; edges = `supersedes`, `amends`, `renumbers` |
| `version_graph/query.py` | `in_force(section_id, date)`, `history(section_id)` |

**Tasks**
- [ ] Start with 2 regulators (SEBI, Income Tax), then extend.
- [ ] Download and store raw files with source URL and download date.
- [ ] Parse and clean; manually inspect 20 documents.
- [ ] Legal-structure chunker (regex on "Section", "Regulation", "Clause", numbered lists).
- [ ] Extract dates and supersession phrases with regex, then verify a sample by hand.
- [ ] Build the version graph; write `in_force()`.
- [ ] Corpus statistics notebook (docs per regulator, chunks, date coverage).

**Key code**
```python
def in_force(graph, section_id: str, date: str) -> str | None:
    """Return the doc_id version of section_id valid on `date`, else None."""
    versions = sorted(graph.history(section_id), key=lambda v: v.effective_from)
    for v in reversed(versions):
        if v.effective_from <= date and (v.effective_to is None or date <= v.effective_to):
            return v.doc_id
    return None
```

**Config (`configs/ingest.yaml`):** sources, output dirs, chunk max tokens (e.g. 400), overlap (0 for legal chunks, parent path kept instead).

**Tests:** parser returns non-empty text; every chunk has `regulator`, `issue_date`, `source_url`; `in_force` returns the right version on 3 hand-picked amendments.

**Done when:** at least 95% of chunks have valid dates, and the version graph resolves known amendments correctly.

**Pitfalls:** dates hidden in tables or footers; renumbered sections (section 80C becoming something else); scanned circulars. Keep the raw file next to every chunk for audits.

---

## M2: `retrieval/` (Regulatory RAG)

**Goal:** given a query and a date, return correct and *current* evidence.

**Inputs:** chunks, query, query date. **Outputs:** `RetrievalResult`.

**Components**
| File | Purpose |
|------|---------|
| `bm25.py` | BM25 index (rank_bm25 or Pyserini) over chunk text + section ids |
| `dense.py` | Embed with BGE-M3, store in FAISS/Qdrant |
| `fusion.py` | Reciprocal rank fusion of BM25 and dense lists |
| `rerank.py` | Cross-encoder reranker, top 5 to 8 |
| `temporal.py` | Filter by `effective_from/to`, drop `superseded_by` chunks, handle "as on date" queries |
| `query_rewrite.py` | (Optional) resolve follow-up questions using chat history |
| `retriever.py` | `retrieve(query, query_date) -> RetrievalResult` |

**Tasks**
- [ ] Build BM25 and dense indexes (`scripts/build_index.py`).
- [ ] RRF fusion (`score = sum(1 / (k + rank))`, k=60).
- [ ] Reranker on top 30 candidates.
- [ ] Temporal filter using the version graph.
- [ ] Follow-up query rewriting (small prompt to Qwen).
- [ ] Build a 200-question retrieval test set (question -> gold chunk ids).
- [ ] Report recall@5, recall@10, MRR, and the **outdated-chunk rate** (how often a superseded chunk is returned, should be near 0).

**Key code**
```python
def rrf(rank_lists: list[list[str]], k: int = 60) -> list[str]:
    scores = {}
    for lst in rank_lists:
        for r, cid in enumerate(lst, 1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + r)
    return sorted(scores, key=scores.get, reverse=True)
```

**Config:** `bm25_k: 50`, `dense_k: 50`, `rerank_top_in: 30`, `final_k: 6`, embedding model name.

**Tests:** temporal filter removes a known superseded chunk; retrieval returns at least one gold chunk for 10 fixture queries.

**Done when:** recall@6 above 0.85 on your test set and outdated-chunk rate below 2%.

**Pitfalls:** exact section references ("Regulation 52(4)") work better with BM25, so do not drop it; dense embeddings ignore dates, so the temporal filter must be a hard rule.

---

## M3: `generation/` (answer generation)

**Goal:** a grounded answer with chunk citations and token log-probs.

**Components**
- `prompts/answer_v1.txt`: system prompt (answer only from evidence, cite `[chunk_id]`, say "not found in evidence" when unsure, keep numbers exactly as written).
- `generator.py`: builds the prompt from `RetrievalResult`, calls vLLM, returns `GeneratedAnswer` with `token_logprobs`.
- `vllm_client.py`: thin wrapper (or reuse `common/llm_client.py`).

**Tasks**
- [ ] Write and version the prompt.
- [ ] Serve Qwen2.5-7B-Instruct with vLLM (`--dtype bfloat16`).
- [ ] Request `logprobs` so M6 can compute `s_ent`.
- [ ] Also store the **no-context answer** (same question, no evidence). M6 needs it for `s_div`.
- [ ] Parse citations from the answer text.
- [ ] Log prompt version, model revision, temperature (use 0 for main runs).

**Key code**
```python
def generate(rr: RetrievalResult) -> GeneratedAnswer:
    ctx = "\n\n".join(f"[{c.chunk_id}] ({c.regulator}, {c.issue_date}) {c.text}" for c in rr.chunks)
    resp = llm.chat(system=SYSTEM_PROMPT, user=f"Evidence:\n{ctx}\n\nQuestion: {rr.query}",
                    temperature=0.0, logprobs=True)
    return GeneratedAnswer(rr.query, resp.text, parse_citations(resp.text),
                           resp.token_logprobs, MODEL_ID)
```

**Tests:** answer always contains at least one citation for fixture questions; logprobs length equals token count.

**Done when:** 100 questions run end to end, with logprobs stored.

**Pitfalls:** the model paraphrasing numbers ("five per cent" vs "5%"), which is exactly what V1 normalization must handle. Do not fine-tune this model (keeps the guarantee story clean).

---

## M4: `atoms/` (atom extraction)

**Goal:** turn the answer into small, individually checkable facts.

**Atom types:** `RATE`, `THRESHOLD`, `SECTION`, `DATE`, `ENTITY`, `APPLICABILITY`.

**Components**
| File | Purpose |
|------|---------|
| `regex_extractors.py` | Deterministic extractors: percentages, rupee amounts (lakh/crore), dates, section/regulation references |
| `llm_extractor.py` | Qwen call with JSON schema, returns atoms with type and span |
| `merge.py` | Merge and deduplicate (overlapping spans, prefer regex for literals, LLM for ENTITY/APPLICABILITY) |
| `span_align.py` | Align each atom to character offsets in `answer_text` |
| `lora/train_extractor.py` | (Optional) LoRA fine-tune for extraction |

**Tasks**
- [ ] Regex extractors for the four literal types, with unit tests on 50 sentences.
- [ ] LLM extractor prompt with JSON output and a schema validator (retry on invalid JSON).
- [ ] Merge logic with deterministic ids (`atom_id = hash(answer_id, span, text)`).
- [ ] Link each atom to the chunk cited nearest to it in the answer (`cited_chunk_id`).
- [ ] Hand-label 100 answers and measure extraction recall/precision per type.
- [ ] (Optional) LoRA fine-tune if recall is below target.

**Key code**
```python
PCT = re.compile(r"\b\d+(?:\.\d+)?\s?(?:%|per\s?cent|percent)\b", re.I)
INR = re.compile(r"(?:₹|Rs\.?|INR)\s?\d[\d,]*(?:\.\d+)?\s?(?:lakh|lakhs|crore|crores|thousand)?", re.I)
SEC = re.compile(r"\b(?:section|regulation|clause|rule)\s+\d+[A-Z]?(?:\(\w+\))*", re.I)
```

**Tests:** each regex on positive and negative examples; merge removes duplicates; JSON-invalid LLM output triggers a retry.

**Done when:** extraction recall at least 95% on RATE, THRESHOLD, DATE, SECTION and at least 85% on ENTITY, APPLICABILITY on your labeled sample.

**Pitfalls:** missed atoms are **silent failures** (an unverified claim goes out). Regex backstop is mandatory. Keep atoms short (one fact each).

---

## M5: `verify/` (verification cascade V1 and V2)

**Goal:** check each atom against the evidence. V1 is cheap and exact, V2 is semantic.

**Components**
| File | Purpose |
|------|---------|
| `normalizers.py` | Canonicalize numbers, units, dates, section references |
| `v1_deterministic.py` | Literal match: exact value/amount, date, section reference, rule/keyword |
| `v2_nli.py` | NLI entailment (DeBERTa-v3) returning `p(entail)`, `p(contradict)`, `p(neutral)` |
| `nli_data/make_pairs.py` | Build NLI training pairs with synthetic perturbations |
| `nli_data/perturb.py` | Number/date/section/applicability perturbation functions |
| `nli_train/train.py` | Fine-tuning script (HF Trainer) |
| `cascade.py` | `verify(atoms, rr) -> list[VerifiedAtom]` |

### V1 tasks
- [ ] Normalizers: `"₹5 lakh" == "500000"`, `"5 per cent" == "5%" == "5.0%"`, `"12/03/2024" == "12 March 2024"`, `"Section 80C(2)" == "80C(2)"`.
- [ ] Matching logic per type: RATE/THRESHOLD compare normalized numeric value in the cited chunk (fall back to all retrieved chunks); DATE compare normalized dates; SECTION check the reference exists in evidence.
- [ ] Return `MATCH`, `MISMATCH` (evidence has a *different* value for the same quantity), `NOT_FOUND`, or `NA`.

```python
def norm_amount(s: str) -> float | None:
    m = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*(lakh|lakhs|crore|crores|thousand)?", s, re.I)
    if not m: return None
    v = float(m.group(1).replace(",", ""))
    mult = {"lakh": 1e5, "lakhs": 1e5, "crore": 1e7, "crores": 1e7, "thousand": 1e3}
    return v * mult.get((m.group(2) or "").lower(), 1)
```

### V2 tasks (NLI verifier, fine-tuned)
- [ ] Start from an MNLI-pretrained DeBERTa-v3-large checkpoint.
- [ ] **Build training pairs** `(premise = evidence sentence/chunk, hypothesis = atom as a sentence)`:
  - Entailment: atoms taken from true statements in the corpus.
  - Contradiction: perturb the number, date, section, or applicability condition; use the *old superseded* version of a rule as a contradicting hypothesis.
  - Neutral: atom about a different topic or an unrelated section.
- [ ] Add human-labeled pairs from the benchmark (M10) for the final stage.
- [ ] Train 3 to 5 epochs, lr around 1e-5, batch 16, bf16, early stopping on per-type macro-F1.
- [ ] Report accuracy and F1 per atom type and per regulator.
- [ ] Cascade: run V2 only for atoms V1 could not settle, or for APPLICABILITY/ENTITY atoms.

```python
def cascade(atom, rr):
    v1 = v1_check(atom, rr)
    if atom.type in {"RATE","THRESHOLD","DATE","SECTION"} and v1 in {"MATCH","MISMATCH"}:
        return VerifiedAtom(atom, v1, ..., v2_entail_prob=None)
    p = nli_entail_prob(premise=best_evidence(atom, rr), hypothesis=atom_to_sentence(atom))
    return VerifiedAtom(atom, v1, ..., p)
```

**Config:** NLI checkpoint path, max length 512, batch size, entailment threshold (this is *not* the final threshold, M7 sets it).

**Tests:** 30 normalization cases; V1 correctly labels MATCH/MISMATCH on synthetic pairs; NLI predicts contradiction on perturbed numbers.

**Done when:** V1 precision above 98% on literal atoms, V2 macro-F1 above 0.85 on held-out pairs.

**Pitfalls:** NLI models are weak on numbers, which is why V1 goes first and why perturbation-based training data matters. **Keep NLI training documents disjoint from calibration/test documents.** V3 (judge) is not called here.

---

## M6: `signals/` and `aggregate/` (multi-signal scoring)

**Goal:** turn everything known about an atom into one risk score in [0, 1].

### Signals (`signals/`)
| Signal | File | How to compute |
|--------|------|----------------|
| `s_div` context divergence | `s_div.py` | Compare answer with evidence vs. no-context answer (embedding cosine distance on the atom's sentence, or token-level KL if you have both distributions) |
| `s_ret` retrieval strength | `s_ret.py` | Reranker score of the best chunk supporting the atom (normalized) |
| `s_ver` verification result | `s_ver.py` | V1 encoding: MATCH=0, NOT_FOUND=0.5, MISMATCH=1, NA=0.5 (or one-hot) |
| `s_nli` entailment | `s_nli.py` | `1 - p(entail)` from V2 (0.5 if skipped, plus a `nli_skipped` flag feature) |
| `s_ent` model uncertainty | `s_ent.py` | Mean and max token entropy (or negative logprob) over the atom's token span |

### Aggregator (`aggregate/`)
- `dataset.py`: builds a feature table (one row per atom: signals, atom type, regulator, label `wrong = 1`).
- `train.py`: logistic regression baseline, then LightGBM. Cross-validate by question family.
- `calibrate_probs.py`: isotonic regression or Platt scaling so `risk` behaves like a probability.
- `model.py`: `score(verified_atoms, answer, rr) -> list[ScoredAtom]`.

**Tasks**
- [ ] Implement each signal as a pure function with tests.
- [ ] Build the feature table on the `agg_train` split.
- [ ] Train LogReg, then LightGBM; compare AUROC/AUPRC per atom type.
- [ ] Signal ablation (drop one signal at a time) and save the table for the paper.
- [ ] Save the model and the feature order (`aggregator_v1.pkl`, `features.json`).

**Key code**
```python
X = feat_df[["s_div","s_ret","s_ver","s_nli","s_ent","nli_skipped"] + type_onehot]
clf = LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=4)
clf.fit(X_train, y_train)          # y = 1 if atom is wrong/unsupported/outdated
risk = IsotonicRegression(out_of_bounds="clip").fit(clf.predict_proba(X_val)[:,1], y_val)
```

**Done when:** AUROC above 0.85 for wrong-atom detection on a held-out split, ablation table produced.

**Pitfalls:** **the aggregator must not see calibration or test data**; NLI outputs on its training data must come from a model that did not train on those pairs (or use cross-fitting).

---

## M7: `guarantee/` (risk control and drift calibration)

**Goal:** choose accept/abstain thresholds with a statistical guarantee.

**Definition (write this in the paper):** among atoms the system *accepts* (risk at or below lambda), the fraction that are wrong is at most epsilon, with probability at least 1 - delta over the calibration data.

**Components**
| File | Purpose |
|------|---------|
| `ltt.py` | Learn-then-Test with valid p-values and fixed-sequence testing |
| `mondrian.py` | Stratum definition (regulator x atom type), fallback hierarchy, delta allocation |
| `drift.py` | Drift detectors and calibration strategies (weighted, sliding-window, triggered) |
| `certify.py` | End-to-end: build strata, calibrate, save `Thresholds` with certificates |
| `simulate.py` | Repeated random calibration/test splits to check the violation rate |

### Learn-then-Test (per stratum)
1. Grid `lambda` in [0, 1] (e.g. 200 points), sorted ascending (most conservative first).
2. For each `lambda`: `n` = number of accepted atoms (`risk <= lambda`), `k` = number of accepted atoms that are wrong.
3. p-value for H0 "true selective risk > epsilon": `p = BinomCDF(k; n, epsilon)` (or Hoeffding-Bentkus for a distribution-free bound).
4. **Fixed-sequence testing:** walk lambda upward, keep the largest lambda while `p <= delta_stratum`; stop at the first failure.
5. Choose the certified lambda as `accept_below`. Set `abstain_above` from a second, looser epsilon (e.g. atoms above it are almost surely wrong), the band in between is UNCERTAIN.

```python
from scipy.stats import binom

def ltt_threshold(risk, wrong, eps, delta, grid):
    best = None
    for lam in sorted(grid):
        acc = risk <= lam
        n, k = acc.sum(), wrong[acc].sum()
        if n == 0: continue
        p = binom.cdf(k, n, eps)          # P(Bin(n, eps) <= k)
        if p <= delta: best = lam          # still certified, try a larger lambda
        else: break                        # fixed-sequence stop
    return best                            # None => abstain on everything in this stratum
```

### Mondrian stratification
- Strata: `regulator|atom_type` (e.g. `SEBI|RATE`).
- If a stratum has fewer than `n_min` calibration atoms (e.g. 100), merge upward: `*|atom_type`, then `*|*`.
- Split delta across strata (Bonferroni: `delta / n_strata`) or report per-stratum guarantees.
- **Cluster correction:** atoms from the same question are correlated. Either calibrate at question level (one random atom per question per draw), or use cluster-robust bounds, and state which one you use.

### Drift-aware calibration
| Strategy | How |
|----------|-----|
| Static (baseline) | Calibrate once on old data |
| Sliding window | Recalibrate on the most recent N months only |
| Recency-weighted | Weight calibration atoms by age or by importance weights from a drift classifier; use effective sample size in place of `n` |
| Triggered | Recalibrate when a drift signal fires (below) |

Triggers: (a) KS or MMD test on the risk-score distribution between calibration window and recent traffic; (b) a regulatory-change event from the version graph (amendment, renumbering, supersession) affecting the stratum's sections.
Drift types to simulate: value drift (rates change), identifier drift (section renumbered), supersession drift, addition drift (new regulation).

**Tasks**
- [ ] Implement `ltt.py` and unit-test on synthetic data with known risk.
- [ ] Implement Mondrian with fallbacks.
- [ ] Simulation: 1000 random calibration/test splits, confirm violation rate is at most delta.
- [ ] Implement the drift detectors and the three strategies.
- [ ] Drift experiment: calibrate on pre-amendment data, test post-amendment, compare static vs. drift-aware.
- [ ] Write the guarantee statement with its assumptions (exchangeability within a window; empirical under drift).
- [ ] Save `thresholds.json` in the `Thresholds` schema.

**Done when:** simulated violation rate at most delta on i.i.d. data, and the drift experiment shows the static method breaking while the drift-aware one recovers.

**Pitfalls:** small strata make everything abstain (use fallbacks); calibration set must be disjoint from the aggregator training set; do not claim a guarantee under drift that you only show empirically.

---

## M8: `judge/` and `decision/` (final decision and V3)

**Goal:** turn thresholds into per-atom outcomes and call the judge only when needed.

**Components**
- `decision/router.py`: uses `Thresholds` for the atom's stratum: `risk <= accept_below` gives SUPPORTED, `risk >= abstain_above` gives ABSTAINED, otherwise UNCERTAIN.
- `judge/judge_llm.py`: strong LLM (Qwen2.5-72B or API) with atom + evidence, returns `{verdict: VERIFIED|NOT_VERIFIED, rationale}`.
- `judge/prompts/judge_v1.txt`: strict prompt, evidence only, quote the supporting sentence.
- `decision/combine.py`: combine atom decisions into the answer-level output (answer is "supported" only if all atoms pass; unsupported atoms are removed or flagged).
- `decision/answer_rewrite.py`: (optional) regenerate or trim the answer without abstained atoms.

**Tasks**
- [ ] Router with stratum lookup and fallback (`SEBI|RATE` -> `*|RATE` -> `*|*`).
- [ ] Judge prompt and JSON output parsing.
- [ ] Cap judge calls per query and log cost/latency.
- [ ] **Guarantee handling for the judge** (choose one and document it):
  - (a) treat "system + judge" as one pipeline and calibrate the end-to-end risk on the calibration set (accepted set = SUPPORTED plus judge-VERIFIED), or
  - (b) estimate the judge's error rate on a labeled set and combine bounds (union bound).
- [ ] Answer-level combination rule.

```python
def decide(scored, th):
    out = []
    for s in scored:
        t = th.get(stratum(s), th["*|*"])
        if s.risk <= t.accept_below:  status = "SUPPORTED"
        elif s.risk >= t.abstain_above: status = "ABSTAINED"
        else:
            j = judge(s)               # only UNCERTAIN atoms
            status = "VERIFIED" if j.verdict == "VERIFIED" else "NOT_VERIFIED"
        out.append(Decision(s.verified.atom.atom_id, status, s.risk))
    return out
```

**Done when:** judge is called only on UNCERTAIN atoms, and the fraction of atoms resolved by V1, V2, and the judge is logged.

**Pitfalls:** judge from the same family as the generator shares blind spots (prefer another family); judge decisions sit outside the calibrated thresholds unless handled as above.

---

## M9: `app/` (user application layer)

**Goal:** a usable demo and API.

**Components**
- `api.py` (FastAPI): `POST /ask {question, date?, history?}` returns `{answer, atoms:[{text,type,status,risk_badge}], sources:[{doc,section,date,url}], verification:{status, risk, explanation}}`.
- `pipeline.py`: chains M2 to M8 using the interface functions.
- `ui/app.py` (Streamlit): question box, answer with colored atom highlights, source panel with links, risk badge (Low / Medium / High), abstention explanation.
- `badge.py`: maps risk to Low/Medium/High (using stratum thresholds).

**Tasks**
- [ ] Wire the pipeline end to end.
- [ ] Response schema with pydantic.
- [ ] Abstained atoms are never shown as facts (show a "could not verify" note instead).
- [ ] Source links with metadata (regulator, document, date).
- [ ] Simple session history for follow-up questions.
- [ ] Latency logging per stage.

**Done when:** demo answers 10 sample questions, with statuses and sources shown.

---

## M10: `bench/` and `eval/` (benchmark, baselines, experiments)

**Start in week 2, this is the critical path.**

### Benchmark (`bench/`)
- `question_gen/`: draft questions from chunks with an LLM (rates, thresholds, sections, dates, applicability), plus follow-up and document-specific questions. Target 1,500 to 3,000 questions.
- `annotation/`: run the pipeline, then annotators label each atom `supported / unsupported / outdated`. Pre-label with an LLM, humans verify everything used for calibration and test.
- `agreement.py`: double-annotate 15%, report Cohen's kappa.
- `temporal_tags.py`: mark questions whose answer changed after an amendment (drift test set).
- `splits.py`: `agg_train` / `calibration` / `test` (by question family or document), plus time-ordered splits for drift.
- `nli_pairs/`: separate labeled pairs for NLI training (disjoint documents from calibration/test).

### Baselines (`eval/baselines/`)
1. Plain RAG (no verification).
2. Self-consistency / SelfCheckGPT.
3. NLI-only verifier (no cascade, no guarantee).
4. LLM-as-judge on every atom.
5. Split-conformal factuality without Mondrian or drift.

### Metrics (`eval/metrics.py`)
- Hallucination rate among accepted atoms.
- Coverage (fraction answered vs. abstained/uncertain).
- Risk-coverage curve, AURC.
- Empirical violation rate versus epsilon over many random splits.
- Cost (judge calls per query) and latency per stage.
- Retrieval metrics from M2, extraction recall from M4.

### Experiments (`eval/`)
- [ ] Main comparison table (RegGuard vs. baselines).
- [ ] Ablations: remove V1, V2, each signal, Mondrian, drift calibration, the judge.
- [ ] Drift experiment (`drift_exp.py`): static vs. sliding vs. weighted vs. triggered.
- [ ] Generalization: repeat with Llama-3.1-8B or Mistral as generator.
- [ ] Human evaluation on a domain-expert sample.
- [ ] Plots: risk-coverage, violation-rate histogram, drift figure, cost breakdown.

**Done when:** all tables and figures are generated by one command (`python -m eval.run_all`).

**Pitfalls:** leakage between splits; labelers seeing model outputs biased toward the model's answer (mask the risk score during annotation); LLM-prelabeled data used without human check.

---

## 3. Integration checklist

| Step | Check |
|------|-------|
| M1 -> M2 | Retrieval reads `chunks.jsonl` and returns only in-force chunks |
| M2 -> M3 | Generator receives `RetrievalResult`, output has citations |
| M3 -> M4 | Atoms have valid spans inside `answer_text` |
| M4 -> M5 | Every atom gets a `VerifiedAtom` (no drops) |
| M5 -> M6 | Feature table has no NaNs (missing NLI encoded with flag) |
| M6 -> M7 | Risk scores on the calibration split come from a model trained on `agg_train` only |
| M7 -> M8 | `thresholds.json` covers every stratum via fallback |
| M8 -> M9 | API returns statuses for all atoms |
| All -> M10 | `run_all` reproduces every table with a fixed seed |

## 4. Master progress tracker

- [ ] M0 common
- [ ] M1 ingest (corpus, version graph)
- [ ] M2 retrieval
- [ ] M3 generation
- [ ] M4 atoms
- [ ] M5 verify (V1, V2 + NLI fine-tune)
- [ ] M6 signals + aggregator
- [ ] M7 guarantee (LTT, Mondrian, drift)
- [ ] M8 judge + decision
- [ ] M9 app
- [ ] M10 benchmark + evaluation
- [ ] Paper draft (method, experiments, limitations, ethics)

## 5. Suggested time per module (1 to 3 people)

| Module | Effort |
|--------|--------|
| M0 | 3 to 4 days |
| M1 | 1.5 to 2 weeks |
| M2 | 1 to 1.5 weeks |
| M3 | 3 to 4 days |
| M4 | 1 week (+ 3 days if LoRA) |
| M5 | 2 weeks (NLI data + training) |
| M6 | 1.5 weeks |
| M7 | 2 weeks |
| M8 | 1 week |
| M9 | 3 to 5 days |
| M10 | runs in parallel, about 4 to 5 weeks total, plus 2 weeks of final experiments |
