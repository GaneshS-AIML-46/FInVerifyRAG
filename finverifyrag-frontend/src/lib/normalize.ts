import type {
  Atom,
  AtomType,
  BenchmarkData,
  CalibrationData,
  Evidence,
  QueryResult,
  Regulator,
} from './types';

export interface BackendChunk {
  chunk_id?: string | null;
  content?: string | null;
  raw_text?: string | null;
  text?: string | null;
  regulator?: string | null;
  title?: string | null;
  clause_number?: string | null;
  section_name?: string | null;
  issue_date?: string | null;
  bm25_score?: number | null;
  dense_score?: number | null;
  rerank_score?: number | null;
  rerank_prob?: number | null;
  latency_ms?: number | null;
}

export interface BackendClaim {
  claim_id?: string | null;
  claim_text?: string | null;
  clause_ref?: string | null;
  citation_tag?: string | null;
}

export interface BackendVerification {
  claim_id?: string | null;
  claim_text?: string | null;
  verdict?: string | null;
  confidence?: number | null;
  calibrated_confidence?: number | null;
  entailment_prob?: number | null;
  neutral_prob?: number | null;
  contradiction_prob?: number | null;
  best_evidence_chunk_id?: string | null;
  best_evidence_text?: string | null;
  target_doc_id?: string | null;
  clause_ref?: string | null;
  method?: string | null;
}

export interface BackendEscalationDecision {
  escalate?: boolean;
  policy?: string;
  reason?: string;
  min_calibrated_confidence?: number;
  cost_usd?: number;
}

export interface BackendQueryResponse {
  query?: string;
  pipeline_mode?: string;
  processed_query?: Record<string, unknown>;
  retrieved_chunks?: BackendChunk[];
  candidate_answer?: string;
  claims_extracted?: BackendClaim[];
  claim_verifications?: BackendVerification[];
  escalation_decision?: BackendEscalationDecision;
  judge_adjudication?: Record<string, unknown> | null;
  final_answer?: string;
  is_escalated?: boolean;
  total_cost_usd?: number;
  total_latency_ms?: number;
  elapsed_time_ms?: number;
  timestamp?: string;
}

export interface BackendBenchmarkSummary {
  policy?: string;
  n_queries?: number;
  total_errors?: number;
  total_clean?: number;
  tp?: number;
  fn?: number;
  tn?: number;
  fp?: number;
  error_recall?: number;
  precision?: number;
  f1_score?: number;
  accuracy?: number;
  unnecessary_escalation_rate?: number;
  total_cost_usd?: number;
  cost_per_query_usd?: number;
  mean_latency_ms?: number;
  cost_savings_pct?: number;
}

export interface BackendBenchmarkRecord {
  regulator?: string;
  query_type?: string;
  ground_truth_error?: boolean;
  min_calibrated_confidence?: number;
  learned_escalation_prob?: number;
  learned_policy_escalate?: boolean;
  learned_policy_correct?: boolean;
}

export interface BackendBenchmarkResponse {
  summary_metrics?: Record<string, BackendBenchmarkSummary>;
  item_level_records?: BackendBenchmarkRecord[];
}

export interface BackendCalibrationResponse {
  method?: string;
  temperature?: number;
  metrics_uncalibrated?: CalibrationMetrics;
  metrics_calibrated?: CalibrationMetrics;
  saved_at?: string;
}

export interface CalibrationMetrics {
  method?: string;
  temperature?: number;
  ece?: number;
  mce?: number;
  brier_score?: number;
  nll?: number;
  sample_count?: number;
}

const clamp01 = (value: number): number => Math.max(0, Math.min(1, value));

function finiteOr(value: number | null | undefined, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function sigmoid(value: number): number {
  return 1 / (1 + Math.exp(-value));
}

function scoreOrProbability(value: number | null | undefined): number {
  if (value === undefined || value === null || !Number.isFinite(value)) return 0;
  return clamp01(value >= 0 && value <= 1 ? value : sigmoid(value));
}

function textOf(chunk: BackendChunk): string {
  return chunk.content ?? chunk.raw_text ?? chunk.text ?? '';
}

export function chunkToEvidence(chunk: BackendChunk, index = 0): Evidence {
  return {
    chunk_id: chunk.chunk_id ?? `retrieved-${index + 1}`,
    text: textOf(chunk),
    source: chunk.regulator ?? '',
    document: chunk.title ?? '',
    section: chunk.clause_number ?? chunk.section_name ?? '',
    effective_date: chunk.issue_date ?? '',
    superseded: false,
    bm25_score: finiteOr(chunk.bm25_score, 0),
    vector_score: finiteOr(chunk.dense_score, 0),
    rerank_score: scoreOrProbability(chunk.rerank_prob ?? chunk.rerank_score),
  };
}

function classifyAtom(text: string): AtomType {
  const value = text.toLowerCase();
  if (/\b(?:19|20)\d{2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b/.test(value)) {
    return 'DATE';
  }
  if (/\b(?:section|clause|regulation|paragraph|rule)\b/.test(value)) return 'SECTION';
  if (/\b(?:at least|minimum|min\.?|greater than|not less than|exceed|maximum|max\.?)\b/.test(value)) return 'THRESHOLD';
  if (/%|\bpercent\b|\b(?:crore|lakh|ratio|rate|basis points?)\b/.test(value)) return 'RATE';
  if (/\b(?:shall|must|required|required to|applicable|applicability)\b/.test(value)) return 'APPLICABILITY';
  return 'ENTITY';
}

function findSpan(answer: string, claimText: string): [number, number] {
  const exactStart = answer.toLocaleLowerCase().indexOf(claimText.toLocaleLowerCase());
  if (exactStart >= 0) return [exactStart, exactStart + claimText.length];

  const words = claimText.trim().split(/\s+/).filter(Boolean).slice(0, 6);
  if (words.length > 0) {
    const prefix = words.join(' ');
    const prefixStart = answer.toLocaleLowerCase().indexOf(prefix.toLocaleLowerCase());
    if (prefixStart >= 0) return [prefixStart, prefixStart + prefix.length];
  }
  return [0, 0];
}

function verificationConfidence(verification: BackendVerification): number {
  return clamp01(finiteOr(verification.calibrated_confidence, finiteOr(verification.confidence, 0)));
}

function isSupported(verdict: string | null | undefined): boolean {
  return verdict?.toUpperCase() === 'SUPPORTED';
}

function isUncertain(verdict: string | null | undefined): boolean {
  const normalized = verdict?.toUpperCase();
  return normalized === 'CONTRADICTED' || normalized === 'UNSUPPORTED' || normalized === 'UNCERTAIN';
}

function hasRefusal(answer: string): boolean {
  return /(?:insufficient|unable to verify|cannot verify|no reliable evidence|do not have enough evidence|not enough evidence|abstain)/i.test(answer);
}

function judgeResult(raw: Record<string, unknown> | null | undefined): null {
  // The current API exposes query-level adjudication, not per-claim judge records.
  void raw;
  return null;
}

function verificationToAtom(
  verification: BackendVerification,
  claim: BackendClaim | undefined,
  answer: string,
  evidenceById: Map<string, Evidence>,
  threshold: number,
  escalated: boolean,
  judgeAdjudication: Record<string, unknown> | null | undefined,
  index: number,
): Atom {
  const text = verification.claim_text ?? claim?.claim_text ?? `Claim ${index + 1}`;
  const entail = clamp01(finiteOr(verification.entailment_prob, 0));
  const neutral = clamp01(finiteOr(verification.neutral_prob, 0));
  const contradiction = clamp01(finiteOr(verification.contradiction_prob, 0));
  const confidence = verificationConfidence(verification);
  const verdict = verification.verdict?.toUpperCase();
  const decision = isSupported(verdict)
    ? 'SUPPORTED'
    : escalated && (verdict === 'CONTRADICTED' || verdict === 'UNSUPPORTED') && hasRefusal(answer)
      ? 'ABSTAIN'
      : 'UNCERTAIN';
  const evidenceId = verification.best_evidence_chunk_id;
  const evidence = evidenceId ? evidenceById.get(evidenceId) : undefined;

  return {
    id: verification.claim_id ?? claim?.claim_id ?? `claim-${index + 1}`,
    text,
    type: classifyAtom(text),
    span: findSpan(answer, text),
    v1: {
      exact_value: /\d|%|\b(?:amount|rate|ratio|number)\b/i.test(text) ? isSupported(verdict) : null,
      date: /\b(?:19|20)\d{2}\b|\b(?:before|after|effective|daily|monthly|quarterly|annually)\b/i.test(text) ? isSupported(verdict) : null,
      section: Boolean(verification.clause_ref ?? claim?.clause_ref),
      rule_match: isSupported(verdict),
    },
    v2: { entail, neutral, contradiction },
    signals: {
      s_div: clamp01(1 - entail),
      s_ret: evidence?.rerank_score ?? 0,
      s_ver: confidence,
      s_nli: entail,
      s_ent: clamp01(1 - neutral),
    },
    risk: clamp01(1 - confidence),
    threshold: clamp01(threshold),
    decision,
    judge: judgeResult(judgeAdjudication),
    evidence_ids: evidenceId ? [evidenceId] : [],
  };
}

export function normalizeQueryResponse(
  raw: BackendQueryResponse,
  request: { query: string; regulator: Regulator; threshold: number },
): QueryResult {
  const chunks = raw.retrieved_chunks ?? [];
  const retrievedEvidence = chunks.map(chunkToEvidence);
  const evidenceById = new Map(retrievedEvidence.map(evidence => [evidence.chunk_id, evidence]));
  const verifications = raw.claim_verifications ?? [];
  const claims = raw.claims_extracted ?? [];
  const rawAnswer = raw.candidate_answer ?? '';
  const finalAnswer = raw.final_answer ?? rawAnswer;
  const atoms = verifications.map((verification, index) => verificationToAtom(
    verification,
    claims.find(claim => claim.claim_id === verification.claim_id),
    rawAnswer,
    evidenceById,
    request.threshold,
    raw.is_escalated === true,
    raw.judge_adjudication,
    index,
  ));
  const hasUncertain = verifications.some(verification => isUncertain(verification.verdict));
  const abstained = atoms.some(atom => atom.decision === 'ABSTAIN') || (raw.is_escalated === true && hasRefusal(finalAnswer));
  const finalDecision = abstained ? 'ABSTAIN' : atoms.length === 0 || hasUncertain ? 'UNCERTAIN' : 'SUPPORTED';
  const meanRisk = atoms.length === 0 ? 0.5 : atoms.reduce((total, atom) => total + atom.risk, 0) / atoms.length;
  const riskBadge = meanRisk < 0.15 ? 'Low' : meanRisk < 0.35 ? 'Medium' : 'High';
  const citedIds = new Set(atoms.flatMap(atom => atom.evidence_ids));
  const citations = citedIds.size > 0
    ? retrievedEvidence.filter(evidence => citedIds.has(evidence.chunk_id))
    : retrievedEvidence.slice(0, 3);
  const retrievalLatency = chunks.reduce((max, chunk) => Math.max(max, finiteOr(chunk.latency_ms, 0)), 0);

  return {
    query: raw.query ?? request.query,
    regulator_filter: request.regulator,
    retrieved_evidence: retrievedEvidence,
    raw_answer: rawAnswer,
    atoms,
    final_decision: finalDecision,
    risk_badge: riskBadge,
    final_answer: finalAnswer,
    citations,
    abstain_reason: finalDecision === 'ABSTAIN' ? raw.escalation_decision?.reason ?? 'The answer could not be verified with sufficient confidence.' : null,
    latency_ms: {
      retrieval: retrievalLatency,
      generation: 0,
      verification: 0,
      total: finiteOr(raw.total_latency_ms, finiteOr(raw.elapsed_time_ms, 0)),
    },
  };
}

function summaryValue(summary: BackendBenchmarkSummary | undefined, key: keyof BackendBenchmarkSummary, fallback = 0): number {
  const val = summary?.[key];
  return typeof val === 'number' ? finiteOr(val, fallback) : fallback;
}

function summaryRisk(summary: BackendBenchmarkSummary | undefined): number {
  const errors = summaryValue(summary, 'total_errors');
  const missed = summaryValue(summary, 'fn');
  return errors > 0 ? missed / errors : 0;
}

function summaryCoverage(summary: BackendBenchmarkSummary | undefined): number {
  const total = summaryValue(summary, 'n_queries');
  const accepted = total - summaryValue(summary, 'tp') - summaryValue(summary, 'fp');
  return total > 0 ? clamp01(accepted / total) : 0;
}

function atomTypeFromQueryType(queryType: string | undefined): AtomType {
  const value = queryType?.toUpperCase() ?? '';
  if (value.includes('DATE') || value.includes('TEMPORAL')) return 'DATE';
  if (value.includes('SECTION') || value.includes('REFERENCE')) return 'SECTION';
  if (value.includes('NUMERICAL') || value.includes('THRESHOLD')) return 'THRESHOLD';
  if (value.includes('APPLICAB')) return 'APPLICABILITY';
  if (value.includes('RATE') || value.includes('PERCENT')) return 'RATE';
  return 'ENTITY';
}

function riskCoverageCurve(records: BackendBenchmarkRecord[]): { coverage: number; risk: number }[] {
  const ordered = [...records].sort((a, b) => finiteOr(b.min_calibrated_confidence, 0) - finiteOr(a.min_calibrated_confidence, 0));
  if (ordered.length === 0) return [];
  let errors = 0;
  return ordered.map((record, index) => {
    if (record.learned_policy_correct === false) errors += 1;
    return { coverage: (index + 1) / ordered.length, risk: errors / (index + 1) };
  });
}

function groupedPerformance(
  records: BackendBenchmarkRecord[],
  key: 'regulator' | 'query_type',
): BenchmarkData['per_regulator'] {
  const groups = new Map<string, BackendBenchmarkRecord[]>();
  for (const record of records) {
    const group = key === 'regulator' ? record.regulator : atomTypeFromQueryType(record.query_type);
    if (!group) continue;
    const entries = groups.get(group) ?? [];
    entries.push(record);
    groups.set(group, entries);
  }
  return [...groups.entries()].map(([group, entries]) => {
    const correct = entries.filter(entry => entry.learned_policy_correct !== false).length;
    const escalated = entries.filter(entry => entry.learned_policy_escalate === true).length;
    return {
      ...(key === 'regulator' ? { regulator: group } : { type: group as AtomType }),
      accuracy: entries.length > 0 ? correct / entries.length : 0,
      abstention: entries.length > 0 ? escalated / entries.length : 0,
    } as BenchmarkData['per_regulator'][number];
  });
}

export function normalizeBenchmark(raw: BackendBenchmarkResponse): BenchmarkData {
  const summaries = raw.summary_metrics ?? {};
  const learned = summaries.LEARNED_POLICY;
  const records = raw.item_level_records ?? [];
  const normal = summaries.NORMAL_RAG;
  const policies = ['NORMAL_RAG', 'ALWAYS_JUDGE', 'VERIFIED_RAG_FIXED', 'LEARNED_POLICY'];
  const baselineComparison = policies.map(policy => {
    const summary = summaries[policy];
    return {
      name: policy === 'LEARNED_POLICY' ? 'Learned Policy' : policy === 'VERIFIED_RAG_FIXED' ? 'Fixed Threshold' : policy === 'ALWAYS_JUDGE' ? 'Always Judge' : 'Normal RAG',
      selective_risk: summaryRisk(summary),
      coverage: summaryCoverage(summary),
      faithfulness: summaryValue(summary, 'accuracy'),
    };
  });
  const nQueries = summaryValue(learned, 'n_queries', records.length);
  const residualErrors = summaryValue(learned, 'fn');

  return {
    selective_risk: summaryRisk(learned),
    epsilon: 0.05,
    coverage: summaryCoverage(learned),
    abstention_rate: summaryValue(learned, 'unnecessary_escalation_rate'),
    hallucination_before: nQueries > 0 ? summaryValue(normal, 'total_errors') / nQueries : 0,
    hallucination_after: nQueries > 0 ? residualErrors / nQueries : 0,
    faithfulness: summaryValue(learned, 'accuracy'),
    recall_at_5: summaryValue(learned, 'error_recall'),
    precision_at_5: summaryValue(learned, 'precision'),
    mrr: summaryValue(learned, 'f1_score'),
    latency_median_ms: summaryValue(learned, 'mean_latency_ms'),
    risk_coverage_curve: riskCoverageCurve(records),
    baseline_comparison: baselineComparison,
    per_atom_type: groupedPerformance(records, 'query_type') as unknown as BenchmarkData['per_atom_type'],
    per_regulator: groupedPerformance(records, 'regulator'),
  };
}

export function normalizeCalibration(raw: BackendCalibrationResponse): CalibrationData & {
  method: string;
  temperature: number;
  ece_before: number;
  ece_after: number;
  sample_count: number;
} {
  const before = raw.metrics_uncalibrated ?? {};
  const after = raw.metrics_calibrated ?? {};
  return {
    thresholds: [],
    epsilon: 0.05,
    coverage: 0,
    method: raw.method ?? after.method ?? 'temperature_scaling',
    temperature: finiteOr(raw.temperature, finiteOr(after.temperature, 0)),
    ece_before: finiteOr(before.ece, 0),
    ece_after: finiteOr(after.ece, 0),
    sample_count: Math.max(0, Math.round(finiteOr(after.sample_count, finiteOr(before.sample_count, 0)))),
  };
}
