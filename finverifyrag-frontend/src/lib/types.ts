// FinVerifyRAG — Core types mirroring the backend data contract (Section 8.3)

export type Regulator = 'RBI' | 'SEBI' | 'All';

export type AtomType =
  | 'RATE'
  | 'THRESHOLD'
  | 'SECTION'
  | 'DATE'
  | 'ENTITY'
  | 'APPLICABILITY';

export type VerdictType = 'SUPPORTED' | 'UNCERTAIN' | 'ABSTAIN';
export type RiskBadge = 'Low' | 'Medium' | 'High';

export interface Evidence {
  chunk_id: string;
  text: string;
  source: string;
  document: string;
  section: string;
  effective_date: string;
  superseded: boolean;
  bm25_score: number;
  vector_score: number;
  rerank_score: number;
}

export interface V1Result {
  exact_value: boolean | null;
  date: boolean | null;
  section: boolean | null;
  rule_match: boolean | null;
}

export interface V2Result {
  entail: number;
  neutral: number;
  contradiction: number;
}

export interface Signals {
  s_div: number;
  s_ret: number;
  s_ver: number;
  s_nli: number;
  s_ent: number;
}

export interface JudgeResult {
  verdict: 'Verified' | 'Not Verified';
  reasoning: string;
}

export interface Atom {
  id: string;
  text: string;
  type: AtomType;
  span: [number, number];
  v1: V1Result;
  v2: V2Result;
  signals: Signals;
  risk: number;
  threshold: number;
  decision: VerdictType;
  judge: JudgeResult | null;
  evidence_ids: string[];
}

export interface LatencyMs {
  retrieval: number;
  generation: number;
  verification: number;
  total: number;
}

export interface QueryResult {
  query: string;
  regulator_filter: Regulator;
  retrieved_evidence: Evidence[];
  raw_answer: string;
  atoms: Atom[];
  final_decision: VerdictType;
  risk_badge: RiskBadge;
  final_answer: string;
  citations: Evidence[];
  abstain_reason: string | null;
  latency_ms: LatencyMs;
}

export interface BenchmarkData {
  selective_risk: number;
  epsilon: number;
  coverage: number;
  abstention_rate: number;
  hallucination_before: number;
  hallucination_after: number;
  faithfulness: number;
  recall_at_5: number;
  precision_at_5: number;
  mrr: number;
  latency_median_ms: number;
  risk_coverage_curve: { coverage: number; risk: number }[];
  baseline_comparison: {
    name: string;
    selective_risk: number;
    coverage: number;
    faithfulness: number;
  }[];
  per_atom_type: {
    type: AtomType;
    accuracy: number;
    abstention: number;
  }[];
  per_regulator: {
    regulator: string;
    accuracy: number;
    abstention: number;
  }[];
}

export interface CalibrationData {
  thresholds: {
    type: AtomType;
    regulator: string;
    threshold: number;
    guarantee_passes: boolean;
  }[];
  epsilon: number;
  coverage: number;
}

export interface TestQuery {
  id: string;
  category: string;
  query: string;
  expected_behavior: string;
  verdict?: VerdictType;
  status: 'pass' | 'fail' | 'not_run';
  trace_id?: string;
}

export interface PipelineStage {
  stage: number;
  name: string;
  status: 'pending' | 'running' | 'done' | 'error';
  latency_ms?: number;
  payload?: unknown;
}
