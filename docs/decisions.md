# Architectural and Feature Removals

This document tracks decisions around dropping specific components, signals, or atom types from the primary pipeline based on validation and calibration data.

## Removal Rules

**1. Component / Signal Removal:**
An optional signal or component will be removed from the primary pipeline only if:
- The upper bound of the 95% marginal-effect interval (via bootstrap/grouped-CV) on AUROC is **< 0.01** AND on coverage-at-fixed-risk is **< 0.01**.
- It introduces a latency or failure bottleneck (failure rate **> 5%** over at least 200 calls, or median added latency **> 2s** with no AUROC gain).
- **Exception**: V1, V2, and Mondrian conformal risk control are core and are NOT pruned by this rule.

**2. Atom Type Removal:**
An atom type will be excluded from the main claim only if:
- Its validation AUROC interval upper bound is **< 0.60**.
- It has at least 20 supported and 20 unsupported atoms (otherwise it is marked "insufficient data").
- AND merging the atom type into a parent stratum (M7 fallback) fails to fix it.
- Note: Any atom type dropped from the main claim must still be reported in an appendix. Small strata are handled by M7 merging, not by dropping.

## Ablation Results & Outcomes

### PROVISIONAL (mock data, not binding)
*(Below values were computed on synthetic validation data and are not binding for removals.)*

#### Component Ablation (Mock Data)
| Component | Baseline AUROC | Ablation AUROC | AUROC Drop |
|-----------|----------------|----------------|------------|
| s_ret (Retrieval) | 0.5000 | 0.5000 | 0.0000 |
| s_ver (V1 Deterministic) | 0.5000 | 0.5000 | 0.0000 |
| s_nli (V2 Entailment) | 0.5000 | 0.5000 | 0.0000 |
| s_ent (V3 Entropy) | 0.5000 | 0.5000 | 0.0000 |
| s_div (V3 Divergence) | 0.5000 | 0.5000 | 0.0000 |

#### Atom Type Analysis (Mock Data)
| Atom Type | Total Atoms | Calib Atoms | Accuracy |
|-----------|-------------|-------------|----------|
| SECTION | 273 | 171 | 0.5092 |
| ENTITY | 283 | 181 | 0.4947 |
| APPLICABILITY | 261 | 173 | 0.4483 |
| RATE | 253 | 151 | 0.5296 |
| DATE | 255 | 153 | 0.5412 |
| THRESHOLD | 264 | 158 | 0.4811 |

#### Proposal (Pending Real Data)
1. **Components**: Retain all components (V1, V2, and Mondrian are core).
2. **Atom Types**: Retain all atom types pending real data evaluation with 95% CI bounds and parent stratum fallback checks.
