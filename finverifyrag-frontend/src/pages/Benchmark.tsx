import { motion } from 'framer-motion';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine,
  ResponsiveContainer, BarChart, Bar, Legend,
} from 'recharts';
import { DEMO_BENCHMARK } from '../demo/demoData';

function KpiCard({ label, value, sub, color }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="card"
      style={{ padding: '20px 22px', textAlign: 'center' }}
    >
      <p style={{ fontFamily: 'var(--font-mono)', fontSize: 28, fontWeight: 700, color: color ?? 'var(--brass-500)', fontVariantNumeric: 'tabular-nums', lineHeight: 1 }}>
        {value}
      </p>
      <p style={{ fontSize: 12, color: 'var(--paper-300)', marginTop: 6 }}>{label}</p>
      {sub && <p style={{ fontSize: 11, color: 'var(--ink-500)', marginTop: 4 }}>{sub}</p>}
    </motion.div>
  );
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: 'var(--ink-800)', border: 'var(--border)', borderRadius: 8, padding: '10px 14px', fontSize: 12 }}>
      <p style={{ color: 'var(--paper-300)', marginBottom: 4 }}>{label ?? payload[0]?.payload?.name}</p>
      {payload.map((p: any) => (
        <p key={p.dataKey} style={{ color: p.color, fontFamily: 'var(--font-mono)' }}>
          {p.name}: {typeof p.value === 'number' ? (p.value < 1 ? (p.value * 100).toFixed(1) + '%' : p.value.toFixed(0)) : p.value}
        </p>
      ))}
    </div>
  );
};

const ATOM_COLORS: Record<string, string> = {
  RATE: 'var(--atom-rate)', THRESHOLD: 'var(--atom-threshold)',
  SECTION: 'var(--atom-section)', DATE: 'var(--atom-date)',
  ENTITY: 'var(--atom-entity)', APPLICABILITY: 'var(--atom-applicability)',
};

const BENCHMARK_DATE_LABEL = new Date().toLocaleDateString();

export default function Benchmark() {
  const b = DEMO_BENCHMARK;

  return (
    <div style={{ maxWidth: 1300, margin: '0 auto', padding: '24px 24px 64px' }}>
      <h1 style={{ fontFamily: 'var(--font-display)', fontSize: 32, fontWeight: 700, color: 'var(--paper-100)', marginBottom: 6 }}>
        Benchmark Dashboard
      </h1>
      <p style={{ color: 'var(--paper-300)', fontSize: 14, marginBottom: 4 }}>
        Final evaluation results · Qwen2.5-7B-Instruct · RBI + SEBI corpora
      </p>
      <p style={{ fontSize: 11, color: 'var(--stop)', fontStyle: 'italic', marginBottom: 28 }}>⚠ Sample data — demo mode. Load real benchmark_results.json in production.</p>

      {/* KPI grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: 12, marginBottom: 32 }}>
        <KpiCard label="Selective Risk" value={`${(b.selective_risk * 100).toFixed(1)}%`} sub={`ε target: ${(b.epsilon * 100).toFixed(0)}%`} color="var(--ok)" />
        <KpiCard label="Coverage" value={`${(b.coverage * 100).toFixed(0)}%`} />
        <KpiCard label="Abstention Rate" value={`${(b.abstention_rate * 100).toFixed(0)}%`} />
        <KpiCard label="Hallucination ↓" value={`${(b.hallucination_before * 100).toFixed(0)}% → ${(b.hallucination_after * 100).toFixed(1)}%`} color="var(--ok)" />
        <KpiCard label="Faithfulness" value={`${(b.faithfulness * 100).toFixed(1)}%`} />
        <KpiCard label="Recall@5" value={`${(b.recall_at_5 * 100).toFixed(1)}%`} />
        <KpiCard label="Precision@5" value={`${(b.precision_at_5 * 100).toFixed(1)}%`} />
        <KpiCard label="MRR" value={b.mrr.toFixed(3)} />
        <KpiCard label="Median Latency" value={`${b.latency_median_ms}ms`} color="var(--paper-300)" />
      </div>

      {/* Two-column charts */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        {/* Risk–Coverage Curve */}
        <div className="card">
          <p style={{ fontSize: 14, fontWeight: 600, color: 'var(--paper-100)', marginBottom: 4 }}>Risk–Coverage Curve</p>
          <p style={{ fontSize: 11, color: 'var(--paper-300)', marginBottom: 16 }}>Operating point vs. ε guarantee line</p>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={b.risk_coverage_curve} margin={{ left: 0, right: 8, top: 4, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--ink-700)" />
              <XAxis dataKey="coverage" tickFormatter={v => `${(v*100).toFixed(0)}%`}
                tick={{ fill: 'var(--paper-300)', fontSize: 11 }} label={{ value: 'Coverage', position: 'insideBottom', offset: -2, fill: 'var(--ink-500)', fontSize: 11 }} />
              <YAxis tickFormatter={v => `${(v*100).toFixed(0)}%`}
                tick={{ fill: 'var(--paper-300)', fontSize: 11 }} label={{ value: 'Risk', angle: -90, position: 'insideLeft', fill: 'var(--ink-500)', fontSize: 11 }} />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine y={b.epsilon} stroke="var(--stop)" strokeDasharray="5 3" label={{ value: `ε=${(b.epsilon*100).toFixed(0)}%`, position: 'right', fill: 'var(--stop)', fontSize: 11 }} />
              <Line dataKey="risk" stroke="var(--brass-500)" strokeWidth={2} dot={{ fill: 'var(--brass-500)', r: 3 }} activeDot={{ r: 5 }} />
              {/* Operating point */}
              <ReferenceLine x={b.coverage} stroke="var(--ok)" strokeDasharray="4 2" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Baseline comparison */}
        <div className="card">
          <p style={{ fontSize: 14, fontWeight: 600, color: 'var(--paper-100)', marginBottom: 4 }}>Baseline Comparison</p>
          <p style={{ fontSize: 11, color: 'var(--paper-300)', marginBottom: 16 }}>Selective risk vs. faithfulness</p>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={b.baseline_comparison} layout="vertical" margin={{ left: 120, right: 16, top: 4, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--ink-700)" horizontal={false} />
              <XAxis type="number" tickFormatter={v => `${(v*100).toFixed(0)}%`} tick={{ fill: 'var(--paper-300)', fontSize: 10 }} domain={[0, 1]} />
              <YAxis type="category" dataKey="name" tick={{ fill: 'var(--paper-300)', fontSize: 10 }} width={120} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: 11, color: 'var(--paper-300)' }} />
              <Bar dataKey="selective_risk" name="Selective Risk" fill="var(--stop)" radius={[0, 3, 3, 0]} />
              <Bar dataKey="faithfulness" name="Faithfulness" fill="var(--ok)" radius={[0, 3, 3, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Per atom type */}
      <div className="card" style={{ marginBottom: 20 }}>
        <p style={{ fontSize: 14, fontWeight: 600, color: 'var(--paper-100)', marginBottom: 16 }}>Performance by Atom Type</p>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: 12 }}>
          {b.per_atom_type.map(({ type, accuracy, abstention }) => (
            <div key={type} style={{ padding: '14px 16px', borderRadius: 10, border: '1px solid', borderColor: ATOM_COLORS[type] + '44', background: ATOM_COLORS[type] + '11' }}>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: ATOM_COLORS[type], fontWeight: 600 }}>{type}</span>
              <div style={{ marginTop: 10 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <span style={{ fontSize: 11, color: 'var(--paper-300)' }}>Accuracy</span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--paper-100)' }}>{(accuracy * 100).toFixed(0)}%</span>
                </div>
                <div style={{ height: 5, background: 'var(--ink-700)', borderRadius: 3 }}>
                  <div style={{ width: `${accuracy * 100}%`, height: '100%', background: ATOM_COLORS[type], borderRadius: 3 }} />
                </div>
              </div>
              <div style={{ marginTop: 8 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <span style={{ fontSize: 11, color: 'var(--paper-300)' }}>Abstention</span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--paper-100)' }}>{(abstention * 100).toFixed(0)}%</span>
                </div>
                <div style={{ height: 5, background: 'var(--ink-700)', borderRadius: 3 }}>
                  <div style={{ width: `${abstention * 100}%`, height: '100%', background: 'var(--warn)', borderRadius: 3 }} />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Per regulator */}
      <div className="card">
        <p style={{ fontSize: 14, fontWeight: 600, color: 'var(--paper-100)', marginBottom: 16 }}>Performance by Regulator</p>
        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
          {b.per_regulator.map(({ regulator, accuracy, abstention }) => (
            <div key={regulator} className="card" style={{ flex: '1 1 180px', padding: '16px 20px', background: 'var(--ink-900)' }}>
              <p style={{ fontFamily: 'var(--font-display)', fontSize: 20, fontWeight: 700, color: 'var(--brass-500)', marginBottom: 8 }}>{regulator}</p>
              <p style={{ fontSize: 13, color: 'var(--paper-300)' }}>Accuracy: <span style={{ color: 'var(--ok)', fontFamily: 'var(--font-mono)' }}>{(accuracy * 100).toFixed(1)}%</span></p>
              <p style={{ fontSize: 13, color: 'var(--paper-300)' }}>Abstention: <span style={{ color: 'var(--warn)', fontFamily: 'var(--font-mono)' }}>{(abstention * 100).toFixed(0)}%</span></p>
            </div>
          ))}
        </div>
        <p style={{ fontSize: 11, color: 'var(--ink-500)', marginTop: 16 }}>Dataset · Qwen2.5-7B-Instruct · {BENCHMARK_DATE_LABEL}</p>
      </div>
    </div>
  );
}
