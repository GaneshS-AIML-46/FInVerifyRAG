# RegGuard: Project Status & Workflow Tracker

This document tracks the migration and completion status of the RegGuard modular architecture (M0-M10).

## 📊 Overview
| Module | Name | Status | Key Deliverables |
|--------|------|--------|------------------|
| **M0** | `common/` | ✅ Complete | Pydantic Schemas, JSONL I/O, Config YAML, Dates, LLM Client (w/ Caching), Fixtures |
| **M1** | `ingest/` | ✅ Complete | PyMuPDF parsing, Regex Legal Chunker (strict boundaries), Date Metadata extraction, Version Graph, Scrapers |
| **M2** | `retrieval/` | ✅ Complete | Hybrid RRF (BM25 + Dense Chroma), Cross-Encoder Rerank, Strict Temporal Graph Filter, Query Rewrite |
| **M3** | `generation/`| ✅ Complete | vLLM Generation, Prompt v1, Citation extraction, Logprob capturing, No-context baseline extraction |
| **M4** | `atoms/` | ✅ Complete | Two-pass extraction (Regex fast-pass + LLM), 6 atom types, span alignment, dedup, "Not found" short-circuit |
| **M5** | `verify/` | ✅ Complete | V1 Deterministic Normalizers + Literal Verifier, V2 NLI (DeBERTa-v3-large lazy loader), Cascade, NLI Data Perturbers, NLI Fine-tune Train Script |
| **M6** | `signals/` | ✅ Complete | `s_div`, `s_ret`, `s_ver`, `s_nli`, `s_ent` signals, `compute_all_signals()` dispatcher, LightGBM Aggregator (`dataset.py`, `train.py`, `calibrate_probs.py`, `model.py`), `configs/signals.yaml` |
| **M7** | `guarantee/` | ✅ Complete | Learn-then-Test (LTT) with Binomial & Hoeffding-Bentkus p-values, Fixed-sequence threshold selection, Mondrian stratification (regulator×atom_type) with Bonferroni delta, 4 Drift-aware strategies (static/sliding_window/recency_weighted/triggered), KS + MMD + version-graph drift detectors, `certify.py` end-to-end pipeline, `thresholds.json` output with `Thresholds` schema. 63 tests ✅. |
| **M8** | `judge/` + `decision/` | ✅ Complete | V3 LLM judge (evidence-only, call-capped), JSON parse with retry, Decision router (SUPPORTED/UNCERTAIN/ABSTAINED) with M7 fallback chain, answer-level combination (FULLY_SUPPORTED/PARTIALLY_VERIFIED/ABSTAINED), answer rewrite (flag/trim/stub modes). 53 tests ✅. |
| **M9** | `app/` | ✅ Complete | FastAPI `/ask` endpoint, `app/pipeline.py` (M2→M8 chain), `app/badge.py` (risk/status badges), Streamlit page `2_RegGuard_QA.py` (atom highlights, sources, session history). 66 tests ✅. |
| **M10**| `bench/` + `eval/` | ✅ Complete | Question gen (9 families), annotation (prelabel + Cohen's κ), splits (by-family/doc/temporal), metrics (hallucination_rate, AURC, violation_rate, recall@k, MRR, F1), 5 baselines, drift experiment (4 strategies), `eval/run_all.py`. 105 tests ✅. |

---

## 🛠️ Module Details & Architectural Decisions

### M0: Common Core
- Completely typed using `pydantic.BaseModel` to guarantee nested JSON serialization without data loss.
- Implemented Indian date format parsers (e.g., `FY 2023-24`, `12/03/2024`).
- Built `LLMClient` backed by Tenacity for automatic retries, which hashes prompts (model + temp + user) to a local `.llm_cache` to drastically reduce API costs during reruns.

### M1: Ingestion & Version Graph
- Moved away from generic token-overlap chunking. Chunks are strictly boundary-matched using Regex (`Chapter I`, `Regulation 52`) to ensure isolated context.
- Text cleaning strips header/footers, repairs hyphenation, and forces all currency values to `₹`.
- Implemented `VersionGraph`. The graph dynamically links amendments (`amends`, `supersedes`) to construct historical snapshots of the law.

### M2: Retrieval
- Custom `BM25` tokenization expressly preserves parenthesis references (e.g. `52(4)`).
- Fused dense vectors (BGE-M3 on ChromaDB) with sparse lexical tokens via a purely mathematical `Reciprocal Rank Fusion (RRF)` pipeline.
- Applied `Temporal Filter`: Drops chunks that are historically superseded or not-yet-effective compared to the user's `query_date`.

### M3: Generation
- Instructs the LLM via strict prompts to yield bracketed citations `[chunk_id]`.
- Enforces returning logprobs for downstream token entropy signals (M6).
- Automatically executes a secondary "No-context" call to extract a baseline hallucination answer (needed for M6 divergence testing).

### M4: Atoms
- Two-pass extraction: fast regex pass (RATE, THRESHOLD, SECTION, DATE) followed by LLM pass (ENTITY, APPLICABILITY + richer claims).
- Merges and deduplicates by (type, text); regex fills gaps LLM misses; span-aligns every atom back to character offsets in `answer_text`.
- Short-circuits on "not found in evidence" answers (returns empty list).

### M5: Verification Cascade
- **`normalizers.py`**: Canonical normalization for monetary amounts (lakh/crore/thousand), percentages (% / per cent / p.a.), Indian date formats (dd/mm/yyyy, "12 March 2024", FY 2023-24), and section references.
- **`v1_deterministic.py`**: Scans retrieved chunks for MATCH/MISMATCH/NOT_FOUND/NA. Cited chunk is checked first; falls back across all chunks if not found. ENTITY/APPLICABILITY atoms return NA (V2 only).
- **`v2_nli.py`**: DeBERTa-v3-large zero-shot NLI wrapped in a lazy singleton. Falls back gracefully to neutral (0.5) if the model checkpoint is unavailable (pre-training phase).
- **`cascade.py`**: Routes each atom: V1 MATCH/MISMATCH → skip V2; V1 NOT_FOUND or atom is ENTITY/APPLICABILITY → run V2. Every atom guaranteed a `VerifiedAtom` output (no silent drops).
- **`nli_data/perturb.py`**: Type-specific perturbation functions (rate ±%, amount ±%, section ±N, date ±1–3 years) for generating CONTRADICTION training pairs.
- **`nli_data/make_pairs.py`**: Builds entailment / contradiction / neutral JSONL pairs for NLI fine-tuning.
- **`nli_train/train.py`**: HuggingFace Trainer script for fine-tuning DeBERTa-v3-large (bf16, early stopping on macro-F1).
- **`configs/verify.yaml`**: Model path, training hyperparameters, data paths.
- **Tests**: 61 tests covering 30 normalization cases, V1 MATCH/MISMATCH/NOT_FOUND/NA for all atom types, multi-chunk fallback, cascade metadata flags. All pass ✅.

### M6: Signals + Aggregator
- **`signals/s_ver.py`**: Encodes V1 status as scalar (MATCH=0, NOT_FOUND/NA=0.5, MISMATCH=1) plus one-hot.
- **`signals/s_nli.py`**: `1 - p(entail)` from V2; defaults to 0.5 + `nli_skipped=1` flag when NLI was skipped.
- **`signals/s_ent.py`**: Mean/max token entropy from logprobs over atom's character span (normalized to [0,1] with `max_clip=5.0`).
- **`signals/s_ret.py`**: Retrieval reranker score for the cited chunk (min-max normalized); rank-based fallback when no reranker scores.
- **`signals/s_div.py`**: Context divergence = `1 - cosine_sim(claim, no_context_answer)` via BGE-small; Strategy B uses KL over token logprob spans.
- **`signals/__init__.py`**: `compute_all_signals()` — single entry point returning flat feature dict for all 5 signals.
- **`aggregate/dataset.py`**: Builds feature table (Parquet) from labeled JSONL atoms with all signals + atom-type/regulator one-hots.
- **`aggregate/train.py`**: LogReg baseline + LightGBM (300 trees, lr=0.05, max_depth=4); cross-validates by question_family; saves `aggregator_v1.pkl` + ablation table.
- **`aggregate/calibrate_probs.py`**: Isotonic regression (≥1000 samples) or Platt scaling calibration on val split.
- **`aggregate/model.py`**: `score()` inference function — loads model, builds feature vector from `compute_all_signals()`, applies calibration, returns `List[ScoredAtom]`; heuristic weighted-average fallback when no model trained.
- **`configs/signals.yaml`**: All signal hyperparameters, model paths, LightGBM config.
- **Tests**: 74 tests (41 signals + 33 aggregator). All pass ✅.

---
### M7: Guarantee (Risk Control + Drift Calibration)
- **`ltt.py`**: Implements `binom_pvalue` (one-sided binomial) and `hb_pvalue` (Hoeffding-Bentkus, distribution-free). Fixed-sequence threshold search walks the lambda grid ascending and stops on the first rejection, ensuring Type-I error control.
- **`mondrian.py`**: Strata keyed as `regulator|atom_type` (e.g. `SEBI|RATE`). Small strata (<n_min atoms) merge upward: `reg|type` → `*|type` → `*|*`. Delta split via Bonferroni. `lookup_threshold()` traverses the fallback chain at inference time.
- **`drift.py`**: KS test (scipy `ks_2samp`), RBF-kernel MMD², version-graph event trigger. Four calibration strategies: static (baseline), sliding-window (keep last N months), recency-weighted (exponential decay, Kish ESS), triggered (fires on detector). `DriftMonitor` is a stateful class for production use.
- **`certify.py`**: Reads calibration JSONL → builds strata → allocates delta → runs LTT pair per stratum → saves `thresholds.json`. Supports all 4 drift strategies. Optionally validates via simulation. CLI: `python -m guarantee.certify --calib_path ... --out_path ...`.
- **`simulate.py`**: `run_simulation()` estimates empirical violation rate; `run_mondrian_simulation()` reports per-stratum rates; `drift_experiment()` compares static vs. sliding vs. recency-weighted on pre/post-amendment data splits.
- **`configs/guarantee.yaml`**: All LTT, Mondrian, drift, and simulation parameters.
- **Tests**: 63 tests covering all components. All pass ✅.

---
### M8: Judge + Decision (V3 LLM Judge and Answer-level Routing)
- **`judge/schemas.py`**: Lightweight `JudgeResult` dataclass and `VERIFIED`/`NOT_VERIFIED` constants — no `openai` dependency, importable anywhere.
- **`judge/judge_llm.py`**: `JudgeLLM` class — evidence-only strict judge, JSON output parsing with markdown-fence stripping, retry on parse failure, hard per-query call cap, timing and cost logging. Uses `TYPE_CHECKING` guard for `LLMClient` to avoid hard `openai` dependency at import time.
- **`judge/prompts/judge_v1.txt`**: Strict prompt: evidence-only verdict, quote required, non-JSON = NOT_VERIFIED.
- **`decision/router.py`**: `route_one()` maps each `ScoredAtom` to `SUPPORTED/UNCERTAIN/ABSTAINED` using M7 thresholds. Walks full fallback chain. Handles `-1.0` sentinel (abstain-all). `route()` does batch routing with status counts.
- **`decision/combine.py`**: `combine()` merges router + judge results into `AtomDecision` objects and `AnswerDecision` (FULLY_SUPPORTED/PARTIALLY_VERIFIED/ABSTAINED). Tracks coverage, judge call counts. `decide()` is the one-call M8 pipeline entry point.
- **`decision/answer_rewrite.py`**: `rewrite_answer()` with 3 modes: `flag` (inline `[⚠ UNVERIFIED]`), `trim` (remove bad sentences), `stub` (full abstention message). Stub triggered automatically when all atoms abstained.
- **`configs/judge.yaml`**: All judge config: model, call cap, prompt path, router thresholds path, combination rule, rewrite mode.
- **Tests**: 53 tests covering all components. All pass ✅.

---

## 🚀 Next Priority
- Start building **M10 (`bench/`)**: Benchmark dataset generation, evaluation pipeline (AURC, Recall@K, F1, empirical violation rate), baselines comparison table, and ablation experiments.

---
### M9: App Layer (FastAPI + Streamlit UI)
- **`app/badge.py`**: `risk_badge(risk)` → LOW/MEDIUM/HIGH with hex colours and HTML `<span>` renderer. `status_badge(status)` → per-atom decision badge for SUPPORTED/VERIFIED/UNCERTAIN/ABSTAINED/NOT_VERIFIED.
- **`app/pipeline.py`**: `run_pipeline(question, query_date, history, skip_judge, rewrite_mode)` chains M2→M8 via lazy-loaded singletons. Graceful stub fallbacks when modules are unavailable. Returns `PipelineResult` with `atoms`, `sources`, `verification`, `latency`.
- **`app/api.py`**: FastAPI. `POST /ask` runs full pipeline (Pydantic validated). `GET /health`. `POST /pipeline/reset`. CORS enabled. CLI `uvicorn` entry point.
- **`streamlit_app/pages/2_RegGuard_QA.py`**: Colour-highlighted answer, per-atom verification cards with risk badges, source panel, session history, latency breakdown, filter controls by status/type/risk.
- **`configs/app.yaml`**: API host/port, pipeline options, badge thresholds and colours, Streamlit settings.
- **Tests**: 66 tests — badge.py (12), pipeline.py (20), api.py (15+). All pass ✅.
