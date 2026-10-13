import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { DEMO_CALIBRATION } from '../demo/demoData';
import type { AtomType } from '../lib/types';
import { CheckCircle, XCircle, AlertTriangle } from 'lucide-react';

const ATOM_COLORS: Record<AtomType, string> = {
  RATE: 'var(--atom-rate)', THRESHOLD: 'var(--atom-threshold)',
  SECTION: 'var(--atom-section)', DATE: 'var(--atom-date)',
  ENTITY: 'var(--atom-entity)', APPLICABILITY: 'var(--atom-applicability)',
};

type DriftType = 'value' | 'identifier' | 'supersession' | 'addition';

const DRIFT_CARDS: { id: DriftType; title: string; desc: string }[] = [
  { id: 'value',       title: 'Value Drift',       desc: 'Rates or percentages change (e.g., CRR revised from 4% to 4.5%)' },
  { id: 'identifier',  title: 'Identifier Drift',  desc: 'Section numbers renumbered in a new circular' },
  { id: 'supersession',title: 'Supersession Drift', desc: 'Old rule replaced by a new regulation' },
  { id: 'addition',    title: 'Addition Drift',     desc: 'Entirely new regulation introduced into corpus' },
];

const buildDriftData = (stage: 'before' | 'during' | 'after') => {
  const base = [0.043, 0.043, 0.043, 0.043, 0.043];
  const during = [0.043, 0.065, 0.091, 0.112, 0.128];
  const after = [0.043, 0.065, 0.091, 0.072, 0.049];
  const vals = stage === 'before' ? base : stage === 'during' ? during : after;
  return vals.map((risk, i) => ({ t: i, risk }));
};

export default function CalibrationDrift() {
  const cal = DEMO_CALIBRATION;
  const [selectedDrift, setSelectedDrift] = useState<DriftType | null>(null);
  const [driftStage, setDriftStage] = useState<'before' | 'during' | 'after'>('before');
  const [strategy, setStrategy] = useState<'weighted' | 'adaptive' | 'rolling'>('weighted');
  const [simulating, setSimulating] = useState(false);

  const simulate = async () => {
    setSimulating(true);
    setDriftStage('before');
    await new Promise(r => setTimeout(r, 600));
    setDriftStage('during');
    await new Promise(r => setTimeout(r, 1200));
    setDriftStage('after');
    setSimulating(false);
  };

  const regs = ['RBI', 'SEBI'];
  const atomTypes: AtomType[] = ['RATE', 'THRESHOLD', 'SECTION', 'DATE', 'ENTITY', 'APPLICABILITY'];

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto', padding: '24px 24px 64px' }}>
      <h1 style={{ fontFamily: 'var(--font-display)', fontSize: 32, fontWeight: 700, color: 'var(--paper-100)', marginBottom: 6 }}>
        Calibration &amp; Drift
      </h1>
      <p style={{ color: 'var(--paper-300)', fontSize: 14, marginBottom: 4 }}>
        Per-atom thresholds, risk guarantee, and drift simulation.
      </p>
      <p style={{ fontSize: 11, color: 'var(--ink-500)', fontStyle: 'italic', marginBottom: 28 }}>Sample data — demo mode</p>

      {/* ε control */}
      <div className="card" style={{ marginBottom: 24, padding: '16px 20px', display: 'flex', alignItems: 'center', gap: 20, flexWrap: 'wrap' }}>
        <div>
          <p style={{ fontSize: 11, color: 'var(--ink-500)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 4 }}>Target ε (selective risk)</p>
          <p style={{ fontFamily: 'var(--font-mono)', fontSize: 28, color: 'var(--brass-500)' }}>{(cal.epsilon * 100).toFixed(0)}%</p>
        </div>
        <div>
          <p style={{ fontSize: 11, color: 'var(--ink-500)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 4 }}>Coverage</p>
          <p style={{ fontFamily: 'var(--font-mono)', fontSize: 28, color: 'var(--paper-100)' }}>{(cal.coverage * 100).toFixed(0)}%</p>
        </div>
        <div style={{ flex: 1, minWidth: 200 }}>
          <p style={{ fontSize: 12, color: 'var(--paper-300)', marginBottom: 6 }}>Guarantee status: Fitted on calibration set · Validated on held-out test set</p>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {cal.thresholds.every(t => t.guarantee_passes) ? (
              <span className="chip" style={{ color: 'var(--ok)', background: 'var(--ok-bg)', borderColor: 'var(--ok)44' }}>
                <CheckCircle size={12} /> Guarantee holds (selective risk ≤ ε)
              </span>
            ) : (
              <span className="chip" style={{ color: 'var(--warn)', background: 'var(--warn-bg)', borderColor: 'var(--warn)44' }}>
                <AlertTriangle size={12} /> Partial failure — recalibration needed
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Threshold table */}
      <div className="card" style={{ marginBottom: 24, overflowX: 'auto' }}>
        <p style={{ fontSize: 14, fontWeight: 600, color: 'var(--paper-100)', marginBottom: 16 }}>Per-Atom Thresholds by Regulator</p>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr>
              <th style={{ textAlign: 'left', padding: '8px 10px', color: 'var(--ink-500)', fontWeight: 500, borderBottom: 'var(--border)' }}>Atom Type</th>
              {regs.map(r => (
                <th key={r} colSpan={2} style={{ textAlign: 'center', padding: '8px 10px', color: 'var(--ink-500)', fontWeight: 500, borderBottom: 'var(--border)' }}>{r}</th>
              ))}
            </tr>
            <tr>
              <th style={{ padding: '4px 10px', borderBottom: 'var(--border)' }} />
              {regs.map(r => (
                <>
                  <th key={r+'-t'} style={{ padding: '4px 10px', color: 'var(--ink-500)', fontSize: 10, fontWeight: 500, borderBottom: 'var(--border)', textAlign: 'center' }}>τ</th>
                  <th key={r+'-g'} style={{ padding: '4px 10px', color: 'var(--ink-500)', fontSize: 10, fontWeight: 500, borderBottom: 'var(--border)', textAlign: 'center' }}>Guarantee</th>
                </>
              ))}
            </tr>
          </thead>
          <tbody>
            {atomTypes.map(type => (
              <tr key={type} style={{ borderBottom: 'var(--border)' }}>
                <td style={{ padding: '10px 10px' }}>
                  <span className="chip" style={{ color: ATOM_COLORS[type], background: ATOM_COLORS[type] + '22', borderColor: ATOM_COLORS[type] + '44' }}>{type}</span>
                </td>
                {regs.map(reg => {
                  const entry = cal.thresholds.find(t => t.type === type && t.regulator === reg);
                  return (
                    <>
                      <td key={reg+'-t'} style={{ padding: '10px', textAlign: 'center', fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--paper-100)' }}>
                        {entry?.threshold.toFixed(2) ?? '—'}
                      </td>
                      <td key={reg+'-g'} style={{ padding: '10px', textAlign: 'center' }}>
                        {entry?.guarantee_passes
                          ? <CheckCircle size={16} color="var(--ok)" aria-label="Pass" />
                          : <XCircle size={16} color="var(--stop)" aria-label="Fail" />}
                      </td>
                    </>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Drift simulator */}
      <div className="card">
        <p style={{ fontSize: 14, fontWeight: 600, color: 'var(--paper-100)', marginBottom: 4 }}>Drift Simulator</p>
        <p style={{ fontSize: 12, color: 'var(--paper-300)', marginBottom: 16 }}>Simulate a regulatory change and watch recalibration restore the guarantee.</p>

        {/* Drift type cards */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 10, marginBottom: 16 }}>
          {DRIFT_CARDS.map(dc => (
            <button key={dc.id}
              onClick={() => setSelectedDrift(dc.id)}
              style={{
                textAlign: 'left', padding: '14px 16px', borderRadius: 10, cursor: 'pointer',
                border: '1px solid', borderColor: selectedDrift === dc.id ? 'var(--brass-500)' : 'var(--ink-700)',
                background: selectedDrift === dc.id ? 'var(--brass-500)11' : 'var(--ink-900)',
                color: 'inherit', transition: 'all 150ms',
              }}
              aria-pressed={selectedDrift === dc.id}
            >
              <p style={{ fontSize: 13, fontWeight: 600, color: selectedDrift === dc.id ? 'var(--brass-500)' : 'var(--paper-100)', marginBottom: 4 }}>{dc.title}</p>
              <p style={{ fontSize: 11, color: 'var(--paper-300)' }}>{dc.desc}</p>
            </button>
          ))}
        </div>

        {/* Strategy selector */}
        <div style={{ display: 'flex', gap: 8, marginBottom: 14, flexWrap: 'wrap', alignItems: 'center' }}>
          <span style={{ fontSize: 12, color: 'var(--paper-300)' }}>Recalibration strategy:</span>
          {(['weighted', 'adaptive', 'rolling'] as const).map(s => (
            <button key={s} onClick={() => setStrategy(s)}
              style={{
                padding: '4px 12px', borderRadius: 6, fontSize: 12, cursor: 'pointer', border: 'var(--border)',
                background: strategy === s ? 'var(--brass-500)1A' : 'transparent',
                color: strategy === s ? 'var(--brass-500)' : 'var(--paper-300)',
              }}
            >{s}</button>
          ))}
        </div>

        {/* Simulate button */}
        <button
          id="simulate-drift-btn"
          onClick={simulate}
          disabled={!selectedDrift || simulating}
          className="btn btn-brass"
          style={{ marginBottom: 16, opacity: !selectedDrift || simulating ? 0.6 : 1 }}
        >
          {simulating ? 'Simulating…' : 'Simulate Drift'}
        </button>

        {/* Stage indicator */}
        <AnimatePresence>
          {selectedDrift && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <div style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
                {(['before', 'during', 'after'] as const).map(s => (
                  <span key={s} className="chip" style={{
                    color: driftStage === s ? 'var(--brass-500)' : 'var(--ink-500)',
                    background: driftStage === s ? 'var(--brass-500)1A' : 'transparent',
                    borderColor: driftStage === s ? 'var(--brass-500)44' : 'var(--ink-700)',
                  }}>{s}</span>
                ))}
              </div>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={buildDriftData(driftStage)} margin={{ left: 0, right: 8, top: 4, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--ink-700)" />
                  <XAxis dataKey="t" tick={{ fill: 'var(--paper-300)', fontSize: 11 }} label={{ value: 'Time →', position: 'insideRight', fill: 'var(--ink-500)', fontSize: 11 }} />
                  <YAxis tickFormatter={v => `${(v*100).toFixed(1)}%`} tick={{ fill: 'var(--paper-300)', fontSize: 11 }} domain={[0, 0.15]} />
                  <Tooltip contentStyle={{ background: 'var(--ink-800)', border: 'var(--border)', borderRadius: 8, fontSize: 12 }} formatter={(v: any) => [`${(Number(v ?? 0)*100).toFixed(1)}%`, 'Risk']} />
                  <ReferenceLine y={cal.epsilon} stroke="var(--stop)" strokeDasharray="5 3" label={{ value: 'ε', position: 'right', fill: 'var(--stop)', fontSize: 12 }} />
                  <Line
                    dataKey="risk" stroke={driftStage === 'during' ? 'var(--stop)' : driftStage === 'after' ? 'var(--ok)' : 'var(--brass-500)'}
                    strokeWidth={2} dot={{ r: 4 }} animationDuration={600}
                  />
                </LineChart>
              </ResponsiveContainer>
              <p style={{ fontSize: 11, color: 'var(--paper-300)', marginTop: 8, fontStyle: 'italic' }}>
                {driftStage === 'before' && 'Baseline: risk ≤ ε, guarantee holds.'}
                {driftStage === 'during' && '⚠ Drift detected: risk exceeds ε. Alarm triggered.'}
                {driftStage === 'after' && `✓ ${strategy} recalibration complete. Guarantee restored.`}
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
