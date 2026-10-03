import { useState } from 'react';

interface Module {
  id: string;
  label: string;
  x: number; y: number; w: number; h: number;
  desc: string;
  color: string;
}

const MODULES: Module[] = [
  { id: 'input',    label: 'User Input',          x: 30,  y: 20,  w: 120, h: 44, desc: 'Raw regulatory question. Type detected (NL / follow-up / doc query).', color: 'var(--ink-700)' },
  { id: 'rag',      label: 'Regulatory RAG',       x: 30,  y: 90,  w: 120, h: 44, desc: 'Hybrid BM25 + vector retrieval over RBI/SEBI corpus. Reranking and temporal filtering removes superseded documents.', color: 'var(--ink-700)' },
  { id: 'gen',      label: 'Answer Generation',    x: 30,  y: 160, w: 120, h: 44, desc: 'Qwen2.5-7B-Instruct generates a fluent answer grounded in retrieved evidence.', color: 'var(--ink-700)' },
  { id: 'extract',  label: 'Atom Extraction',      x: 30,  y: 230, w: 120, h: 44, desc: 'Answer is decomposed into atomic factual claims, each typed (RATE, THRESHOLD, SECTION, DATE, ENTITY, APPLICABILITY).', color: 'var(--ink-700)' },
  { id: 'm4',       label: 'M4 Verification',      x: 200, y: 160, w: 130, h: 60, desc: 'Two-stage cascade: V1 deterministic (exact match, date, section, rule) → V2 NLI (entailment probability). V3 (judge) only for uncertain atoms.', color: 'var(--brass-500)22' },
  { id: 'm5',       label: 'M5 Multi-Signal',      x: 200, y: 240, w: 130, h: 60, desc: 'Five signals: s_div (divergence), s_ret (retrieval), s_ver (verification), s_nli (NLI score), s_ent (entropy). Combined into a risk score.', color: 'var(--brass-500)22' },
  { id: 'm6',       label: 'M6 Guarantee Layer',   x: 370, y: 200, w: 130, h: 60, desc: 'Calibrated threshold τ per atom type and regulator. Risk-coverage guarantee: selective risk ≤ ε on held-out set. Drift alarm and weighted/adaptive/rolling recalibration.', color: 'var(--ok)22' },
  { id: 'judge',    label: 'Judge LLM',             x: 370, y: 290, w: 130, h: 44, desc: 'Stronger LLM reviews uncertain atoms. Returns Verified / Not Verified with reasoning.', color: 'var(--warn)22' },
  { id: 'output',   label: 'Final Output',          x: 550, y: 220, w: 120, h: 44, desc: 'Verified answer with per-atom verdict, citations, risk badge, and abstain explanation if applicable.', color: 'var(--ok)22' },
];

const ARROWS: [string, string][] = [
  ['input', 'rag'], ['rag', 'gen'], ['gen', 'extract'], ['extract', 'm4'],
  ['m4', 'm5'], ['m5', 'm6'], ['m6', 'output'], ['m6', 'judge'], ['judge', 'output'],
];

function cx(m: Module) { return m.x + m.w / 2; }
function cy(m: Module) { return m.y + m.h / 2; }

export default function Architecture() {
  const [hovered, setHovered] = useState<string | null>(null);
  const hovM = MODULES.find(m => m.id === hovered);
  const byId = Object.fromEntries(MODULES.map(m => [m.id, m]));

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto', padding: '24px 24px 64px' }}>
      <h1 style={{ fontFamily: 'var(--font-display)', fontSize: 32, fontWeight: 700, color: 'var(--paper-100)', marginBottom: 6 }}>
        Architecture
      </h1>
      <p style={{ color: 'var(--paper-300)', fontSize: 14, marginBottom: 24 }}>
        Interactive pipeline diagram. Hover any block to see what it does.
      </p>

      <div className="card" style={{ padding: 24, overflow: 'auto' }}>
        <svg
          viewBox="0 0 720 380"
          width="100%"
          style={{ display: 'block', minWidth: 500 }}
          role="img"
          aria-label="FinVerifyRAG pipeline architecture diagram"
        >
          {/* Arrows */}
          <defs>
            <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--ink-500)" />
            </marker>
          </defs>
          {ARROWS.map(([from, to]) => {
            const f = byId[from], t = byId[to];
            if (!f || !t) return null;
            return (
              <line
                key={`${from}-${to}`}
                x1={cx(f)} y1={cy(f)}
                x2={cx(t)} y2={cy(t)}
                stroke={hovered === from || hovered === to ? 'var(--brass-500)' : 'var(--ink-500)'}
                strokeWidth={hovered === from || hovered === to ? 2 : 1}
                markerEnd="url(#arrow)"
                style={{ transition: 'stroke 200ms, stroke-width 200ms' }}
              />
            );
          })}

          {/* Module blocks */}
          {MODULES.map(m => (
            <g key={m.id}
              onMouseEnter={() => setHovered(m.id)}
              onMouseLeave={() => setHovered(null)}
              style={{ cursor: 'pointer' }}
              role="button"
              aria-label={`Module: ${m.label}`}
              tabIndex={0}
              onFocus={() => setHovered(m.id)}
              onBlur={() => setHovered(null)}
            >
              <rect
                x={m.x} y={m.y} width={m.w} height={m.h} rx={10}
                fill={hovered === m.id ? 'var(--brass-500)22' : m.color}
                stroke={hovered === m.id ? 'var(--brass-500)' : 'var(--ink-700)'}
                strokeWidth={hovered === m.id ? 2 : 1}
                style={{ transition: 'fill 200ms, stroke 200ms' }}
              />
              <text
                x={m.x + m.w / 2} y={m.y + m.h / 2}
                textAnchor="middle" dominantBaseline="middle"
                fill={hovered === m.id ? 'var(--brass-300)' : 'var(--paper-100)'}
                fontSize={11} fontFamily="var(--font-ui)" fontWeight="600"
                style={{ transition: 'fill 200ms', pointerEvents: 'none' }}
              >
                {m.label}
              </text>
            </g>
          ))}

          {/* Module label markers */}
          {[{ id: 'm4', tag: 'M4' }, { id: 'm5', tag: 'M5' }, { id: 'm6', tag: 'M6' }].map(({ id, tag }) => {
            const m = byId[id];
            return (
              <text key={id} x={m.x + 6} y={m.y + 14} fill="var(--brass-500)" fontSize={9} fontFamily="var(--font-mono)" fontWeight={700}>{tag}</text>
            );
          })}
        </svg>

        {/* Description panel */}
        <div style={{
          marginTop: 20, minHeight: 72, padding: '14px 18px', borderRadius: 10,
          background: 'var(--ink-900)', border: 'var(--border)',
          transition: 'all 200ms',
        }}>
          {hovM ? (
            <>
              <p style={{ fontSize: 14, fontWeight: 600, color: 'var(--brass-500)', marginBottom: 6 }}>{hovM.label}</p>
              <p style={{ fontSize: 13, color: 'var(--paper-300)', lineHeight: 1.6 }}>{hovM.desc}</p>
            </>
          ) : (
            <p style={{ fontSize: 13, color: 'var(--ink-500)', fontStyle: 'italic' }}>Hover a block above to see its description.</p>
          )}
        </div>
      </div>

      {/* Module explanations */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 14, marginTop: 24 }}>
        {[
          { title: 'M4 — Verification Cascade', body: 'Two-stage deterministic + NLI verification. V1 catches exact numerical and reference matches. V2 uses natural language inference to catch semantic misalignment. Together they resolve most atoms without the Judge.' },
          { title: 'M5 — Multi-Signal Scoring', body: 'Six complementary signals are aggregated into a single risk score per atom. No single signal dominates; the ensemble is robust to any one signal being noisy or unavailable.' },
          { title: 'M6 — Guarantee Layer',       body: 'Learn-then-test conformal calibration fits a threshold τ per atom type and regulator on a held-out calibration set, then certifies selective risk ≤ ε on the test set. Drift alarms trigger adaptive recalibration.' },
        ].map(({ title, body }) => (
          <div key={title} className="card" style={{ padding: '18px 20px' }}>
            <p style={{ fontFamily: 'var(--font-display)', fontSize: 16, fontWeight: 600, color: 'var(--brass-500)', marginBottom: 8 }}>{title}</p>
            <p style={{ fontSize: 13, color: 'var(--paper-300)', lineHeight: 1.7 }}>{body}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
