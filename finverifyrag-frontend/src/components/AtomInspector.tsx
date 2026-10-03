import { useState } from 'react';
import type { Atom, Evidence } from '../lib/types';
import { VerdictBadge, AtomTypeChip } from './VerdictBadge';
import {
  FileText, CheckSquare, BarChart2, Activity, Gauge, MessageSquare,
} from 'lucide-react';
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer, Tooltip,
} from 'recharts';

const TABS = [
  { id: 'evidence', label: 'Evidence',  Icon: FileText     },
  { id: 'v1',       label: 'V1 Det.',   Icon: CheckSquare  },
  { id: 'v2',       label: 'V2 NLI',    Icon: BarChart2    },
  { id: 'signals',  label: 'Signals',   Icon: Activity     },
  { id: 'decision', label: 'Decision',  Icon: Gauge        },
  { id: 'judge',    label: 'Judge LLM', Icon: MessageSquare },
];

function TabBtn({ id, label, Icon, active, onClick }: {
  id: string; label: string; Icon: typeof FileText;
  active: boolean; onClick: () => void;
}) {
  return (
    <button
      id={`atom-inspector-tab-${id}`}
      aria-selected={active}
      onClick={onClick}
      style={{
        display: 'flex', alignItems: 'center', gap: 5,
        padding: '7px 12px', borderRadius: 8, fontSize: 12, fontWeight: 500,
        color: active ? 'var(--brass-500)' : 'var(--paper-300)',
        background: active ? 'var(--brass-500)1A' : 'transparent',
        border: active ? '1px solid var(--brass-500)44' : '1px solid transparent',
        cursor: 'pointer', transition: 'all 150ms', whiteSpace: 'nowrap',
      }}
    >
      <Icon size={13} aria-hidden /> {label}
    </button>
  );
}

function EvidenceTab({ atom, evidences }: { atom: Atom; evidences: Evidence[] }) {
  const relevant = evidences.filter(e => atom.evidence_ids.includes(e.chunk_id));
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {relevant.map(ev => (
        <div key={ev.chunk_id} className="card" style={{ padding: 16 }}>
          <div style={{ display: 'flex', gap: 8, marginBottom: 8, flexWrap: 'wrap' }}>
            <span className="chip" style={{ color: 'var(--brass-500)', background: 'var(--brass-500)1A', borderColor: 'var(--brass-500)44' }}>{ev.source}</span>
            <span className="chip" style={{ color: 'var(--paper-300)', background: 'var(--ink-700)', fontSize: 11 }}>{ev.section}</span>
            <span className="chip" style={{ color: 'var(--paper-300)', background: 'var(--ink-700)', fontSize: 11 }}>{ev.effective_date}</span>
            {ev.superseded && <span className="chip" style={{ color: 'var(--stop)', background: 'var(--stop-bg)' }}>Superseded</span>}
          </div>
          <p style={{ fontSize: 13, color: 'var(--paper-300)', lineHeight: 1.6, fontFamily: 'var(--font-ui)' }}>{ev.text}</p>
          <div style={{ marginTop: 10, display: 'flex', gap: 16, fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--ink-500)' }}>
            <span>BM25: {ev.bm25_score.toFixed(2)}</span>
            <span>Vector: {ev.vector_score.toFixed(2)}</span>
            <span>Rerank: {ev.rerank_score.toFixed(2)}</span>
          </div>
          <p style={{ marginTop: 4, fontSize: 11, color: 'var(--ink-500)' }}>{ev.document}</p>
        </div>
      ))}
    </div>
  );
}

function V1Tab({ atom }: { atom: Atom }) {
  const checks = [
    { label: 'Exact value match', value: atom.v1.exact_value },
    { label: 'Date match',        value: atom.v1.date },
    { label: 'Section reference', value: atom.v1.section },
    { label: 'Rule / keyword',    value: atom.v1.rule_match },
  ];
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {checks.map(({ label, value }) => (
        <div key={label} style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '10px 14px', background: 'var(--ink-900)', borderRadius: 8, border: 'var(--border)',
        }}>
          <span style={{ fontSize: 13, color: 'var(--paper-300)' }}>{label}</span>
          {value === null ? (
            <span style={{ fontSize: 12, color: 'var(--ink-500)', fontFamily: 'var(--font-mono)' }}>N/A</span>
          ) : value ? (
            <span style={{ fontSize: 12, color: 'var(--ok)', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>✓ Pass</span>
          ) : (
            <span style={{ fontSize: 12, color: 'var(--stop)', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>✗ Fail</span>
          )}
        </div>
      ))}
    </div>
  );
}

function V2Tab({ atom }: { atom: Atom }) {
  const { entail, neutral, contradiction } = atom.v2;
  const total = entail + neutral + contradiction;
  const segments = [
    { label: 'Entailment',    value: entail,       color: 'var(--ok)' },
    { label: 'Neutral',       value: neutral,       color: 'var(--warn)' },
    { label: 'Contradiction', value: contradiction, color: 'var(--stop)' },
  ];
  return (
    <div>
      <p style={{ fontSize: 12, color: 'var(--paper-300)', marginBottom: 12 }}>NLI probability distribution over retrieved evidence:</p>
      <div style={{ height: 24, borderRadius: 4, overflow: 'hidden', display: 'flex', marginBottom: 12 }}>
        {segments.map(s => (
          <div key={s.label} style={{ flex: s.value / total, background: s.color, transition: 'flex 500ms' }} />
        ))}
      </div>
      {segments.map(s => (
        <div key={s.label} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ width: 10, height: 10, borderRadius: 2, background: s.color, display: 'inline-block' }} />
            <span style={{ fontSize: 13, color: 'var(--paper-300)' }}>{s.label}</span>
          </div>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--paper-100)' }}>
            {(s.value * 100).toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  );
}

function SignalsTab({ atom }: { atom: Atom }) {
  const data = [
    { signal: 's_div',  label: 'Divergence',  value: atom.signals.s_div,  desc: 'Semantic divergence from evidence' },
    { signal: 's_ret',  label: 'Retrieval',   value: atom.signals.s_ret,  desc: 'Retrieval confidence score' },
    { signal: 's_ver',  label: 'Verification',value: atom.signals.s_ver,  desc: 'Deterministic verification score' },
    { signal: 's_nli',  label: 'NLI',         value: atom.signals.s_nli,  desc: 'Natural language inference score' },
    { signal: 's_ent',  label: 'Entropy',     value: atom.signals.s_ent,  desc: 'Prediction entropy (uncertainty)' },
  ];
  return (
    <div>
      <div style={{ height: 220 }}>
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={data.map(d => ({ subject: d.label, value: d.value * 100 }))}>
            <PolarGrid stroke="var(--ink-700)" />
            <PolarAngleAxis dataKey="subject" tick={{ fontSize: 11, fill: 'var(--paper-300)', fontFamily: 'var(--font-ui)' }} />
            <Radar dataKey="value" stroke="var(--brass-500)" fill="var(--brass-500)" fillOpacity={0.15} />
            <Tooltip
              contentStyle={{ background: 'var(--ink-800)', border: 'var(--border)', borderRadius: 8, fontSize: 12 }}
              formatter={(v: number) => [`${v.toFixed(1)}%`, 'Score']}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 8 }}>
        {data.map(d => (
          <div key={d.signal} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '6px 10px', borderRadius: 6, background: 'var(--ink-900)' }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--brass-500)', width: 48, flexShrink: 0 }}>{d.signal}</span>
            <div style={{ flex: 1, height: 5, background: 'var(--ink-700)', borderRadius: 3 }}>
              <div style={{ width: `${d.value * 100}%`, height: '100%', background: 'var(--brass-500)', borderRadius: 3, transition: 'width 600ms' }} />
            </div>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--paper-100)', width: 36, textAlign: 'right' }}>{(d.value * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function DecisionTab({ atom }: { atom: Atom }) {
  const pct = atom.risk * 100;
  const threshold = atom.threshold * 100;
  return (
    <div>
      <p style={{ fontSize: 12, color: 'var(--paper-300)', marginBottom: 16 }}>
        Aggregated risk score vs. calibrated threshold for <strong style={{ color: 'var(--paper-100)' }}>{atom.type}</strong> atoms.
        A score below the threshold → <span style={{ color: 'var(--ok)' }}>Supported</span>; above → abstain or escalate.
      </p>
      {/* Gauge */}
      <div style={{ position: 'relative', marginBottom: 20 }}>
        <div style={{ height: 28, background: 'var(--ink-900)', borderRadius: 4, overflow: 'visible', position: 'relative', border: 'var(--border)' }}>
          {/* Fill */}
          <div style={{
            position: 'absolute', left: 0, top: 0, bottom: 0,
            width: `${Math.min(pct * 2, 100)}%`,
            background: pct < threshold ? 'var(--ok)' : 'var(--stop)',
            borderRadius: 4, transition: 'width 700ms',
          }} />
          {/* Threshold notch */}
          <div style={{
            position: 'absolute', top: -6, bottom: -6,
            left: `${Math.min(threshold * 2, 100)}%`,
            width: 2, background: 'var(--brass-500)',
            borderRadius: 2,
          }}>
            <span style={{
              position: 'absolute', top: -18, left: -16, fontSize: 10,
              fontFamily: 'var(--font-mono)', color: 'var(--brass-500)', whiteSpace: 'nowrap',
            }}>τ = {atom.threshold.toFixed(2)}</span>
          </div>
        </div>
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
        <span style={{ color: 'var(--paper-300)' }}>Risk score: <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--paper-100)' }}>{atom.risk.toFixed(3)}</span></span>
        <VerdictBadge verdict={atom.decision} size="sm" />
      </div>
    </div>
  );
}

function JudgeTab({ atom }: { atom: Atom }) {
  if (!atom.judge) {
    return (
      <p style={{ fontSize: 13, color: 'var(--ink-500)', fontStyle: 'italic' }}>
        Judge LLM was not invoked — atom was resolved at V1 or V2 level.
      </p>
    );
  }
  const { verdict, reasoning } = atom.judge;
  return (
    <div>
      <div style={{ marginBottom: 12 }}>
        <span
          className="chip"
          style={{
            color: verdict === 'Verified' ? 'var(--ok)' : 'var(--stop)',
            background: verdict === 'Verified' ? 'var(--ok-bg)' : 'var(--stop-bg)',
            borderColor: (verdict === 'Verified' ? 'var(--ok)' : 'var(--stop)') + '44',
            fontSize: 13,
          }}
        >
          Judge: {verdict}
        </span>
      </div>
      <p style={{ fontSize: 13, color: 'var(--paper-300)', lineHeight: 1.7 }}>{reasoning}</p>
    </div>
  );
}

interface AtomInspectorProps {
  atom: Atom | null;
  evidences: Evidence[];
}

export default function AtomInspector({ atom, evidences }: AtomInspectorProps) {
  const [tab, setTab] = useState<string>('evidence');

  if (!atom) {
    return (
      <div className="card" style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: 12 }}>
        <Activity size={32} color="var(--ink-500)" />
        <p style={{ color: 'var(--ink-500)', fontSize: 14 }}>Click an atom to inspect it</p>
      </div>
    );
  }

  return (
    <div className="card" style={{ height: '100%', display: 'flex', flexDirection: 'column', gap: 0, padding: 0, overflow: 'hidden' }}>
      {/* Header */}
      <div style={{ padding: '16px 20px', borderBottom: 'var(--border)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
          <AtomTypeChip type={atom.type} />
          <VerdictBadge verdict={atom.decision} size="sm" />
        </div>
        <p style={{ fontSize: 13, color: 'var(--paper-100)', lineHeight: 1.5 }}>"{atom.text}"</p>
      </div>

      {/* Tabs */}
      <div style={{ padding: '10px 16px', borderBottom: 'var(--border)', display: 'flex', gap: 4, overflowX: 'auto', flexWrap: 'wrap' }}>
        {TABS.filter(t => t.id !== 'judge' || atom.decision === 'UNCERTAIN').map(t => (
          <TabBtn key={t.id} {...t} active={tab === t.id} onClick={() => setTab(t.id)} />
        ))}
      </div>

      {/* Tab content */}
      <div style={{ flex: 1, overflowY: 'auto', padding: 20 }}>
        {tab === 'evidence' && <EvidenceTab atom={atom} evidences={evidences} />}
        {tab === 'v1'       && <V1Tab atom={atom} />}
        {tab === 'v2'       && <V2Tab atom={atom} />}
        {tab === 'signals'  && <SignalsTab atom={atom} />}
        {tab === 'decision' && <DecisionTab atom={atom} />}
        {tab === 'judge'    && <JudgeTab atom={atom} />}
      </div>
    </div>
  );
}
