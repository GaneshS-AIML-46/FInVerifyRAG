import { useState } from 'react';
import { motion } from 'framer-motion';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { DEMO_RESULT } from '../demo/demoData';
import { VerdictBadge, AtomTypeChip } from '../components/VerdictBadge';

const STAGES = [
  { num: 1,  name: 'User Input',               desc: 'Raw query and detected type' },
  { num: 2,  name: 'Regulatory RAG',           desc: 'Hybrid BM25 + vector retrieval, reranking, temporal filter' },
  { num: 3,  name: 'Answer Generation',        desc: 'Prompt, model output, token count' },
  { num: 4,  name: 'Atom Extraction',          desc: 'Atomic claims with type classification' },
  { num: 5,  name: 'M4 Verification Cascade',  desc: 'V1 deterministic → V2 NLI per atom' },
  { num: 6,  name: 'M5 Multi-Signal Scoring',  desc: 'Six-signal heatmap per atom' },
  { num: 7,  name: 'M6 Guarantee Layer',       desc: 'Threshold family, certification status' },
  { num: 8,  name: 'Final Decision',           desc: 'Per-atom verdict table' },
  { num: 9,  name: 'Judge LLM',               desc: 'Uncertain atoms reviewed by stronger model' },
  { num: 10, name: 'Final Output',            desc: 'User-facing verified answer' },
];

const SIGNAL_KEYS = ['s_div','s_ret','s_ver','s_nli','s_ent'] as const;

function StageCard({ stage, result, expanded, onToggle }: {
  stage: typeof STAGES[0]; result: typeof DEMO_RESULT;
  expanded: boolean; onToggle: () => void;
}) {
  const { num, name, desc } = stage;
  const isActive = num <= 10; // All done in demo

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: (num - 1) * 0.05 }}
      className="card"
      style={{ padding: 0, overflow: 'hidden' }}
    >
      <button
        onClick={onToggle}
        style={{
          width: '100%', display: 'flex', alignItems: 'center', gap: 14, padding: '14px 18px',
          background: 'none', border: 'none', cursor: 'pointer', color: 'inherit',
        }}
        aria-expanded={expanded}
        id={`stage-btn-${num}`}
      >
        {/* Stage number */}
        <div style={{
          width: 32, height: 32, borderRadius: '50%', flexShrink: 0,
          background: isActive ? 'var(--brass-500)1A' : 'var(--ink-700)',
          border: `1px solid ${isActive ? 'var(--brass-500)' : 'var(--ink-700)'}`,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontFamily: 'var(--font-mono)', fontSize: 13, color: isActive ? 'var(--brass-500)' : 'var(--ink-500)',
          fontWeight: 600,
        }}>
          {num}
        </div>
        <div style={{ flex: 1, textAlign: 'left' }}>
          <p style={{ fontSize: 14, fontWeight: 600, color: 'var(--paper-100)' }}>{name}</p>
          <p style={{ fontSize: 12, color: 'var(--paper-300)' }}>{desc}</p>
        </div>
        {isActive && (
          <span style={{ fontSize: 11, color: 'var(--ok)', fontFamily: 'var(--font-mono)', marginRight: 8 }}>done</span>
        )}
        {expanded ? <ChevronDown size={16} color="var(--ink-500)" /> : <ChevronRight size={16} color="var(--ink-500)" />}
      </button>

      {expanded && (
        <div style={{ padding: '0 18px 18px', borderTop: 'var(--border)' }}>
          {num === 1 && (
            <div style={{ paddingTop: 12 }}>
              <p style={{ fontSize: 13, color: 'var(--paper-100)', marginBottom: 8 }}><strong>Query:</strong> {result.query}</p>
              <p style={{ fontSize: 13, color: 'var(--paper-300)' }}><strong>Regulator:</strong> {result.regulator_filter}</p>
              <p style={{ fontSize: 13, color: 'var(--paper-300)' }}><strong>Type:</strong> Natural language query</p>
            </div>
          )}
          {num === 2 && (
            <div style={{ paddingTop: 12, display: 'flex', flexDirection: 'column', gap: 10 }}>
              {result.retrieved_evidence.map(ev => (
                <div key={ev.chunk_id} style={{ padding: '10px 14px', borderRadius: 8, background: 'var(--ink-900)', border: 'var(--border)' }}>
                  <div style={{ display: 'flex', gap: 8, marginBottom: 6, flexWrap: 'wrap' }}>
                    <span className="chip" style={{ color: 'var(--brass-500)', background: 'var(--brass-500)1A', borderColor: 'var(--brass-500)44' }}>{ev.source}</span>
                    <span className="chip" style={{ color: 'var(--paper-300)', background: 'var(--ink-700)' }}>{ev.section}</span>
                    {ev.superseded && <span className="chip" style={{ color: 'var(--stop)', background: 'var(--stop-bg)' }}>Superseded</span>}
                  </div>
                  <p style={{ fontSize: 12, color: 'var(--paper-300)', lineHeight: 1.6 }}>{ev.text.slice(0, 200)}…</p>
                  <div style={{ marginTop: 8, display: 'flex', gap: 16, fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--ink-500)' }}>
                    <span>BM25: {ev.bm25_score.toFixed(2)}</span>
                    <span>Vector: {ev.vector_score.toFixed(2)}</span>
                    <span>Rerank: {ev.rerank_score.toFixed(2)}</span>
                  </div>
                </div>
              ))}
              <p style={{ fontSize: 11, color: 'var(--ink-500)', fontStyle: 'italic' }}>Retrieval: {result.latency_ms.retrieval}ms</p>
            </div>
          )}
          {num === 3 && (
            <div style={{ paddingTop: 12 }}>
              <p style={{ fontSize: 12, color: 'var(--paper-300)', marginBottom: 6 }}>Model: Qwen2.5-7B-Instruct · {result.latency_ms.generation}ms</p>
              <div style={{ padding: 12, borderRadius: 8, background: 'var(--ink-900)', border: 'var(--border)' }}>
                <p style={{ fontSize: 13, color: 'var(--paper-100)', lineHeight: 1.7 }}>{result.raw_answer}</p>
              </div>
            </div>
          )}
          {num === 4 && (
            <div style={{ paddingTop: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
              {result.atoms.map(a => (
                <div key={a.id} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '8px 12px', borderRadius: 8, background: 'var(--ink-900)', border: 'var(--border)' }}>
                  <AtomTypeChip type={a.type} />
                  <p style={{ fontSize: 12, color: 'var(--paper-300)', flex: 1 }}>{a.text}</p>
                </div>
              ))}
            </div>
          )}
          {num === 5 && (
            <div style={{ paddingTop: 12 }}>
              <p style={{ fontSize: 12, color: 'var(--paper-300)', marginBottom: 8 }}>V1 resolved: {result.atoms.filter(a => a.v1.exact_value || a.v1.rule_match).length} atoms · V2 escalated: {result.atoms.filter(a => a.v2.entail > 0.5).length} atoms. V3 not used.</p>
              {result.atoms.map(a => (
                <div key={a.id} style={{ marginBottom: 8, padding: 10, borderRadius: 8, background: 'var(--ink-900)', border: 'var(--border)' }}>
                  <div style={{ display: 'flex', gap: 8, marginBottom: 4 }}>
                    <AtomTypeChip type={a.type} />
                    <span style={{ fontSize: 11, color: a.v1.exact_value ? 'var(--ok)' : 'var(--warn)', fontFamily: 'var(--font-mono)' }}>
                      V1: {a.v1.exact_value ? 'pass' : 'fail'}
                    </span>
                    <span style={{ fontSize: 11, color: a.v2.entail > 0.7 ? 'var(--ok)' : 'var(--warn)', fontFamily: 'var(--font-mono)' }}>
                      V2: {(a.v2.entail * 100).toFixed(0)}% entail
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
          {num === 6 && (
            <div style={{ paddingTop: 12, overflowX: 'auto' }}>
              <p style={{ fontSize: 12, color: 'var(--paper-300)', marginBottom: 10 }}>Signal heatmap (rows = atoms, columns = signals)</p>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: 'var(--ink-500)', fontWeight: 500 }}>Atom</th>
                    {SIGNAL_KEYS.map(s => <th key={s} style={{ padding: '6px 8px', color: 'var(--ink-500)', fontFamily: 'var(--font-mono)', fontWeight: 500 }}>{s}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {result.atoms.map(a => (
                    <tr key={a.id} style={{ borderTop: 'var(--border)' }}>
                      <td style={{ padding: '6px 8px', color: 'var(--paper-300)' }}><AtomTypeChip type={a.type} /></td>
                      {SIGNAL_KEYS.map(s => {
                        const v = a.signals[s];
                        const intensity = Math.round(v * 255).toString(16).padStart(2, '0');
                        return (
                          <td key={s} style={{ padding: '6px 8px', textAlign: 'center' }}>
                            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, padding: '3px 6px', borderRadius: 4, background: `#34B89A${intensity}`, color: 'var(--paper-100)' }}>
                              {v.toFixed(2)}
                            </span>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {num === 7 && (
            <div style={{ paddingTop: 12 }}>
              {result.atoms.map(a => (
                <div key={a.id} style={{ display: 'flex', gap: 10, marginBottom: 8, padding: '8px 12px', borderRadius: 8, background: 'var(--ink-900)', border: 'var(--border)' }}>
                  <AtomTypeChip type={a.type} />
                  <span style={{ fontSize: 12, color: 'var(--paper-300)' }}>τ = {a.threshold.toFixed(2)}</span>
                  <span style={{ fontSize: 12, color: 'var(--ok)', fontFamily: 'var(--font-mono)' }}>✓ Certified</span>
                </div>
              ))}
            </div>
          )}
          {num === 8 && (
            <div style={{ paddingTop: 12 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr>
                    {['Atom', 'Type', 'Risk', 'Threshold', 'Verdict'].map(h => (
                      <th key={h} style={{ textAlign: 'left', padding: '6px 8px', color: 'var(--ink-500)', fontWeight: 500, borderBottom: 'var(--border)' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.atoms.map(a => (
                    <tr key={a.id} style={{ borderBottom: 'var(--border)' }}>
                      <td style={{ padding: '8px', fontSize: 11, color: 'var(--paper-300)', maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{a.text}</td>
                      <td style={{ padding: '8px' }}><AtomTypeChip type={a.type} /></td>
                      <td style={{ padding: '8px', fontFamily: 'var(--font-mono)', color: 'var(--paper-100)' }}>{a.risk.toFixed(3)}</td>
                      <td style={{ padding: '8px', fontFamily: 'var(--font-mono)', color: 'var(--ink-500)' }}>{a.threshold.toFixed(2)}</td>
                      <td style={{ padding: '8px' }}><VerdictBadge verdict={a.decision} size="sm" /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {num === 9 && (
            <div style={{ paddingTop: 12 }}>
              {result.atoms.filter(a => a.judge).map(a => (
                <div key={a.id} style={{ padding: '12px 14px', borderRadius: 8, background: 'var(--ink-900)', border: 'var(--border)' }}>
                  <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
                    <AtomTypeChip type={a.type} />
                    <span className="chip" style={{ color: 'var(--ok)', background: 'var(--ok-bg)', borderColor: 'var(--ok)44' }}>Judge: {a.judge!.verdict}</span>
                  </div>
                  <p style={{ fontSize: 12, color: 'var(--paper-300)', lineHeight: 1.6 }}>{a.judge!.reasoning}</p>
                </div>
              ))}
              {result.atoms.every(a => !a.judge) && <p style={{ fontSize: 13, color: 'var(--ink-500)', paddingTop: 8 }}>No uncertain atoms escalated to Judge LLM.</p>}
            </div>
          )}
          {num === 10 && (
            <div style={{ paddingTop: 12 }}>
              <p style={{ fontSize: 14, color: 'var(--paper-100)', lineHeight: 1.7, marginBottom: 12 }}>{result.final_answer}</p>
              <VerdictBadge verdict={result.final_decision} size="md" />
            </div>
          )}
        </div>
      )}
    </motion.div>
  );
}

export default function Pipeline() {
  const [expanded, setExpanded] = useState<number | null>(null);
  const result = DEMO_RESULT;
  const latencies = [result.latency_ms.retrieval, 0, result.latency_ms.generation, 0, result.latency_ms.verification, 0, 0, 0, 0, 0];
  const maxLat = Math.max(...latencies);

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto', padding: '24px 24px 48px' }}>
      <h1 style={{ fontFamily: 'var(--font-display)', fontSize: 32, fontWeight: 700, color: 'var(--paper-100)', marginBottom: 6 }}>
        Pipeline Trace
      </h1>
      <p style={{ color: 'var(--paper-300)', fontSize: 14, marginBottom: 24 }}>
        Step-by-step trace of all 10 stages for: <em>"{result.query}"</em>
      </p>
      <p style={{ fontSize: 11, color: 'var(--ink-500)', marginBottom: 20, fontStyle: 'italic' }}>Sample data — demo mode</p>

      {/* Latency waterfall */}
      <div className="card" style={{ marginBottom: 24 }}>
        <p style={{ fontSize: 12, color: 'var(--ink-500)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 12 }}>Latency waterfall</p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {[
            { label: 'Retrieval',     ms: result.latency_ms.retrieval },
            { label: 'Generation',    ms: result.latency_ms.generation },
            { label: 'Verification',  ms: result.latency_ms.verification },
            { label: 'Total',         ms: result.latency_ms.total },
          ].map(({ label, ms }) => (
            <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <span style={{ fontSize: 12, color: 'var(--paper-300)', width: 90, flexShrink: 0 }}>{label}</span>
              <div style={{ flex: 1, height: 8, background: 'var(--ink-700)', borderRadius: 4 }}>
                <div style={{ width: `${(ms / (result.latency_ms.total || 1)) * 100}%`, height: '100%', background: label === 'Total' ? 'var(--brass-500)' : 'var(--ok)', borderRadius: 4, transition: 'width 600ms' }} />
              </div>
              <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--paper-100)', width: 50, textAlign: 'right' }}>{ms}ms</span>
            </div>
          ))}
        </div>
      </div>

      {/* Stage cards */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {STAGES.map(s => (
          <StageCard
            key={s.num}
            stage={s}
            result={result}
            expanded={expanded === s.num}
            onToggle={() => setExpanded(expanded === s.num ? null : s.num)}
          />
        ))}
      </div>
    </div>
  );
}
