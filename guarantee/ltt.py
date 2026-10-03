"""
guarantee/ltt.py
Learn-then-Test (LTT) with fixed-sequence binomial-tail testing.
Supports atom-level and cluster-level (Hoeffding-Bentkus) certification.

Public API expected by tests:
  binom_pvalue, hb_pvalue
  ltt_threshold, ltt_thresholds_pair
  simulate_violation_rate
  cluster_z_scores, cluster_hb_pvalue
  build_lambda_grid
  DEFAULT_N_START, SAMPLE_SIZE_SANITY
  StrataStatus, LTTResult
  calibrate_ltt, build_mondrian_thresholds, print_stratum_status, get_strata_keys
"""
from __future__ import annotations

import math
import logging
from typing import Optional, Sequence

import numpy as np
from scipy.stats import binom

logger = logging.getLogger("guarantee.ltt")

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_N_START: int = 50

# Sanity table: minimum atoms to have power at (eps, delta).
# Derived from: smallest n where P(Bin(n, eps) <= 0) <= delta.
SAMPLE_SIZE_SANITY: dict[tuple[float, float], int] = {
    (0.05, 0.10): 45,
    (0.05, 0.05): 59,
    (0.02, 0.10): 114,
    (0.02, 0.05): 149,
    (0.10, 0.10): 22,
    (0.10, 0.05): 29,
}


# ── Clopper-Pearson interval ──────────────────────────────────────────────────

def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """
    Exact 95% Clopper-Pearson interval for a binomial proportion k/n.
    Returns (lo, hi).
    """
    from scipy.stats import beta as beta_dist
    if n == 0:
        return (0.0, 1.0)
    lo = beta_dist.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta_dist.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return (float(lo), float(hi))


# ── Certificate dataclass ─────────────────────────────────────────────────────

class Certificate:
    """
    Result of certify_lambda: a richer object than LTTResult, used by the
    publishable pipeline (test_phase_pipeline.py).
    """
    def __init__(
        self,
        status: str,
        threshold: Optional[float],
        coverage: float,
        selective_risk: float,
        n_calib: int,
        n_accepted: int,
        eps: float,
        delta: float,
    ):
        self.status = status
        self.threshold = threshold
        self.coverage = coverage
        self.selective_risk = selective_risk
        self.n_calib = n_calib
        self.n_accepted = n_accepted
        self.eps = eps
        self.delta = delta

    @property
    def valid(self) -> bool:
        return self.status == "certified" and self.threshold is not None and self.coverage > 0



# ── Min-atoms helper ──────────────────────────────────────────────────────────

def min_accepted(eps: float, delta: float) -> int:
    """
    ceil(ln(delta) / ln(1 - eps)) — minimum atoms needed for LTT to have power.
    """
    return int(math.ceil(math.log(delta) / math.log(1.0 - eps)))


# ── P-value functions ─────────────────────────────────────────────────────────

def binom_pvalue(k: int, n: int, eps: float) -> float:
    """Binomial tail: P(Bin(n, eps) <= k). Returns 1.0 when n=0."""
    if n == 0:
        return 1.0
    return float(binom.cdf(k, n, eps))


def hb_pvalue(k: int, n: int, eps: float) -> float:
    """
    Hoeffding-Bentkus (HB) p-value: min(1, e * binom_pvalue(k, n, eps)).
    Tighter than pure binomial for large n.
    """
    if n == 0:
        return 1.0
    return min(1.0, math.e * binom_pvalue(k, n, eps))


# ── Grid construction ─────────────────────────────────────────────────────────

def build_lambda_grid(
    agg_val_risk: np.ndarray,
    n_grid: int = 200,
) -> np.ndarray:
    """
    Build a threshold grid from the quantiles of agg_val risk scores.
    Falls back to uniform [0, 1] when agg_val_risk is empty.
    Returns a sorted ascending array.
    """
    if len(agg_val_risk) == 0:
        return np.linspace(0.0, 1.0, n_grid + 1)
    quantiles = np.linspace(0.0, 1.0, n_grid + 1)
    grid = np.unique(np.quantile(agg_val_risk, quantiles))
    return np.sort(grid)


# ── Cluster-level test (Section 9) ───────────────────────────────────────────

def cluster_z_scores(
    risk: np.ndarray,
    wrong: np.ndarray,
    question_ids: np.ndarray,
    lam: float,
    eps: float,
) -> list[float]:
    """
    Compute per-question Z_q = sum_{i in q, r_i <= lam}(wrong_i - eps).
    Returns a list of Z_q values (one per question with at least one accepted atom).
    """
    accepted = risk <= lam
    z_scores = []
    for qid in np.unique(question_ids):
        mask = accepted & (question_ids == qid)
        n_q = int(np.sum(mask))
        if n_q == 0:
            continue
        k_q = float(np.sum(wrong[mask]))
        z_scores.append(k_q - eps * n_q)
    return z_scores


def cluster_hb_pvalue(
    risk: np.ndarray,
    wrong: np.ndarray,
    question_ids: np.ndarray,
    lam: float,
    eps: float,
) -> float:
    """
    Hoeffding-Bentkus p-value for cluster-level H0: selective_risk > eps.
    p = min(1, e * exp(-2 * m * min(z_bar, 0)^2))
    When z_bar >= 0 (violations), H0 is supported -> p = 1.0 (cannot reject).
    When z_bar < 0 (compliance), H0 is rejected -> small p.
    """
    zs = cluster_z_scores(risk, wrong, question_ids, lam, eps)
    m = len(zs)
    if m == 0:
        return 1.0
    z_bar = sum(zs) / m
    if z_bar >= 0:
        # z_bar >= 0 means expected violation; p-value is large (do NOT certify)
        return 1.0
    # z_bar < 0: compliance direction; Hoeffding gives a small p
    hoeff = math.exp(-2.0 * m * z_bar ** 2)
    return min(1.0, math.e * hoeff)


# ── Core LTT threshold search ─────────────────────────────────────────────────

def ltt_threshold(
    risk: np.ndarray,
    wrong: np.ndarray,
    eps: float,
    delta: float,
    n_start: int = DEFAULT_N_START,
    grid: Optional[np.ndarray] = None,
    n_grid: int = 200,
    agg_val_risk: Optional[np.ndarray] = None,
    question_ids: Optional[np.ndarray] = None,
    use_hb: bool = True,
) -> Optional[float]:
    """
    Fixed-sequence LTT: ascend the lambda grid (low → high), test each lambda
    where n_accepted >= n_start.  Return the FIRST lambda that passes at level
    delta.  Return None (fail-closed) if no lambda passes.

    If question_ids is provided, uses cluster_hb_pvalue instead of atom-level test.
    use_hb: if True (default), use Hoeffding-Bentkus bound; else plain binom_pvalue.
    """
    if grid is None:
        if agg_val_risk is not None and len(agg_val_risk) > 0:
            grid = build_lambda_grid(agg_val_risk, n_grid=n_grid)
        else:
            grid = np.linspace(0.0, 1.0, n_grid + 1)

    risk = np.asarray(risk, dtype=float)
    wrong = np.asarray(wrong, dtype=float)

    # Ascending fixed-sequence: test from low lambda to high.
    # We break (fail-closed) on the FIRST lambda where n >= n_start AND p > delta.
    # We return the first lambda where p <= delta.
    started = False  # whether we've reached n_start yet

    for lam in np.sort(grid):
        accepted = risk <= lam
        n_acc = int(np.sum(accepted))

        if n_acc < n_start:
            continue  # skip until n_start reached

        started = True

        if question_ids is not None:
            p = cluster_hb_pvalue(risk, wrong, question_ids, lam, eps)
        else:
            k = int(np.sum(wrong[accepted]))
            p = hb_pvalue(k, n_acc, eps) if use_hb else binom_pvalue(k, n_acc, eps)

        if p <= delta:
            return float(lam)
        else:
            # Fixed-sequence: break on first failure after n_start
            break

    return None


def ltt_thresholds_pair(
    risk: np.ndarray,
    wrong: np.ndarray,
    eps_accept: float,
    delta: float,
    n_start: int = DEFAULT_N_START,
    agg_val_risk: Optional[np.ndarray] = None,
    grid: Optional[np.ndarray] = None,
    n_grid: int = 200,
    question_ids: Optional[np.ndarray] = None,
    use_hb: bool = True,
) -> tuple[Optional[float], float]:
    """
    Return (accept_below, abstain_above).

    accept_below: lowest certified lambda (or None if no certification).
    abstain_above: conservative judge-budget threshold. Always set.
      = accept_below * 1.5 if certified; else = eps_accept (safety default).

    Invariant: if both are not None, accept_below < abstain_above.
    """
    accept_below = ltt_threshold(
        risk, wrong, eps_accept, delta,
        n_start=n_start, agg_val_risk=agg_val_risk,
        grid=grid, n_grid=n_grid, question_ids=question_ids,
        use_hb=use_hb,
    )

    if accept_below is not None:
        abstain_above = min(1.0, accept_below * 1.5 + 0.05)
        if abstain_above <= accept_below:
            abstain_above = min(1.0, accept_below + 0.05)
    else:
        abstain_above = eps_accept  # conservative fallback for judge budget

    return accept_below, float(abstain_above)


# ── Violation-rate simulation ─────────────────────────────────────────────────

def simulate_violation_rate(
    risk: np.ndarray,
    wrong: np.ndarray,
    eps: float,
    delta: float,
    n_splits: int = 200,
    calib_frac: float = 0.6,
    n_start: int = DEFAULT_N_START,
    n_grid: int = 200,
    use_hb: bool = True,
    seed: int = 0,
    question_ids: Optional[np.ndarray] = None,
) -> dict:
    """
    Repeatedly split (risk, wrong) into calibrate/validate, run LTT on
    calibrate, measure selective risk on validate. Report violation rate.
    """
    rng = np.random.default_rng(seed)
    n = len(risk)
    n_calib = max(1, int(n * calib_frac))

    violations = 0
    n_valid = 0
    risks_on_valid = []
    coverages = []
    lambdas_certified = []

    for _ in range(n_splits):
        idx = rng.permutation(n)
        c_idx, v_idx = idx[:n_calib], idx[n_calib:]
        if len(v_idx) == 0:
            continue

        lam = ltt_threshold(
            risk[c_idx], wrong[c_idx], eps, delta,
            n_start=n_start,
            n_grid=n_grid,
            use_hb=use_hb,
            question_ids=question_ids[c_idx] if question_ids is not None else None,
        )
        n_valid += 1

        if lam is None:
            # Fail-closed: no atoms accepted -> coverage 0, no violation
            coverages.append(0.0)
            continue

        lambdas_certified.append(lam)
        accepted_v = risk[v_idx] <= lam
        n_acc_v = int(np.sum(accepted_v))
        coverage = n_acc_v / len(v_idx)
        coverages.append(coverage)

        if n_acc_v > 0:
            sel_risk = float(np.sum(wrong[v_idx][accepted_v])) / n_acc_v
            risks_on_valid.append(sel_risk)
            if sel_risk > eps:
                violations += 1

    violation_rate = violations / n_valid if n_valid > 0 else 0.0
    mean_cov = float(np.mean(coverages)) if coverages else 0.0
    mean_lam = float(np.mean(lambdas_certified)) if lambdas_certified else float("nan")
    # valid = we ran at least some splits AND at least some had coverage > 0
    n_certified = sum(1 for c in coverages if c > 0)
    valid = n_valid > 0 and n_certified > 0

    return {
        "violation_rate": violation_rate if valid else float("nan"),
        "target_delta": delta,
        "mean_lambda": mean_lam,
        "n_splits": n_splits,
        "n_valid_splits": n_valid if valid else 0,
        "n_violations": violations,
        "mean_selective_risk": float(np.mean(risks_on_valid)) if risks_on_valid else 0.0,
        "mean_coverage": mean_cov,
        "valid": valid,
        "n_start": n_start,
    }


# ── Mondrian stratum types ────────────────────────────────────────────────────

class StrataStatus:
    CERTIFIED = "certified"
    MERGED = "merged"
    NOT_CERTIFIABLE = "not_certifiable"


class LTTResult:
    def __init__(
        self,
        accept_below: Optional[float],
        status: str,
        n_calib: int,
        n_accepted: int,
        eps: float,
        delta: float,
        violation_rate: float = 0.0,
        abstain_above: Optional[float] = None,
    ):
        self.accept_below = accept_below
        self.abstain_above = abstain_above
        self.status = status
        self.n_calib = n_calib
        self.n_accepted = n_accepted
        self.eps = eps
        self.delta = delta
        self.violation_rate = violation_rate


def calibrate_ltt(
    risks: np.ndarray,
    labels: np.ndarray,
    eps: float,
    delta: float,
    grid_size: int = 200,
    n_start: int = DEFAULT_N_START,
) -> LTTResult:
    """
    Calibrate LTT on (risks, labels) arrays.
    labels: 1 = unsupported (wrong), 0 = supported (correct).
    """
    n_calib = len(risks)
    min_acc = int(math.ceil(math.log(delta) / math.log(1.0 - eps)))

    if n_calib < min_acc:
        return LTTResult(None, StrataStatus.MERGED, n_calib, 0, eps, delta, 0.0)

    grid = np.linspace(0.0, 1.0, grid_size)
    lam = ltt_threshold(risks, labels, eps, delta, n_start=n_start, grid=grid)

    if lam is not None:
        accepted_mask = risks <= lam
        n_acc = int(np.sum(accepted_mask))
        k = int(np.sum(labels[accepted_mask]))
        viol_rate = k / n_acc if n_acc > 0 else 0.0
        abstain = min(1.0, lam * 1.5 + 0.05)
        return LTTResult(lam, StrataStatus.CERTIFIED, n_calib, n_acc, eps, delta, viol_rate, abstain)
    else:
        return LTTResult(None, StrataStatus.NOT_CERTIFIABLE, n_calib, 0, eps, delta, 0.0)


def get_strata_keys(
    row: dict,
    use_regulator: bool = False,
) -> tuple[str, str, Optional[str]]:
    atom_type = row.get("atom_type", "UNKNOWN")
    if use_regulator:
        reg = row.get("regulator", "UNKNOWN")
        return f"{reg}|{atom_type}", f"*|{atom_type}", "*|*"
    return atom_type, "*|*", None


def build_mondrian_thresholds(
    rows: list[dict],
    eps: float,
    delta: float,
    use_regulator: bool = False,
) -> dict[str, LTTResult]:
    strata_data: dict[str, dict] = {}
    for row in rows:
        keys = get_strata_keys(row, use_regulator)
        for k in keys:
            if k is None:
                continue
            if k not in strata_data:
                strata_data[k] = {"risks": [], "labels": []}
            strata_data[k]["risks"].append(row.get("risk", 0.0))
            lbl = row.get("label")
            label_val = 1 if lbl in ("unsupported", "outdated", 1, "1", True) else 0
            strata_data[k]["labels"].append(label_val)

    results: dict[str, LTTResult] = {}
    for k, data in strata_data.items():
        risks = np.array(data["risks"])
        labels = np.array(data["labels"])
        results[k] = calibrate_ltt(risks, labels, eps, delta)
    return results


def print_stratum_status(
    calibration_rows: list[dict],
    eps: float = 0.05,
    delta: float = 0.10,
    use_regulator: bool = False,
) -> None:
    results = build_mondrian_thresholds(calibration_rows, eps, delta, use_regulator)
    header = f"{'Stratum':<20} | {'n_calib':<8} | {'n_accepted':<10} | {'coverage':<10} | {'violation':<10} | {'threshold':<10} | {'status':<15}"
    print(header)
    print("-" * len(header))
    for k, res in sorted(results.items()):
        cov = f"{res.n_accepted/res.n_calib:.1%}" if res.n_calib > 0 else "0%"
        viol = f"{res.violation_rate:.1%}"
        th = f"{res.accept_below:.4f}" if res.accept_below is not None else "None"
        print(f"{k:<20} | {res.n_calib:<8} | {res.n_accepted:<10} | {cov:<10} | {viol:<10} | {th:<10} | {res.status:<15}")


# ── High-level publishable API ────────────────────────────────────────────────

def certify_lambda(
    risk: np.ndarray,
    wrong: np.ndarray,
    eps: float,
    delta: float,
    n_start: int = DEFAULT_N_START,
    n_grid: int = 200,
    use_hb: bool = True,
    agg_val_risk: Optional[np.ndarray] = None,
    question_ids: Optional[np.ndarray] = None,
) -> "Certificate":
    """
    High-level certification entry point returning a Certificate object.
    use_hb=False uses binom_pvalue instead of hb_pvalue.
    """
    risk = np.asarray(risk, dtype=float)
    wrong = np.asarray(wrong, dtype=float)
    n_calib = len(risk)

    grid = build_lambda_grid(agg_val_risk, n_grid=n_grid) if agg_val_risk is not None and len(agg_val_risk) > 0 \
        else np.linspace(0.0, 1.0, n_grid + 1)

    # Choose p-value function
    def _pvalue(k: int, n: int) -> float:
        return hb_pvalue(k, n, eps) if use_hb else binom_pvalue(k, n, eps)

    lam: Optional[float] = None
    for candidate in np.sort(grid):
        accepted = risk <= candidate
        n_acc = int(np.sum(accepted))
        if n_acc < n_start:
            continue
        k = int(np.sum(wrong[accepted]))
        p = _pvalue(k, n_acc)
        if p <= delta:
            lam = float(candidate)
            break
        else:
            break  # fixed-sequence: stop on first failure after n_start

    if lam is not None:
        accepted_mask = risk <= lam
        n_acc = int(np.sum(accepted_mask))
        k = int(np.sum(wrong[accepted_mask]))
        coverage = n_acc / n_calib if n_calib > 0 else 0.0
        sel_risk = k / n_acc if n_acc > 0 else 0.0
        return Certificate("certified", lam, coverage, sel_risk, n_calib, n_acc, eps, delta)
    else:
        return Certificate("not_certifiable", None, 0.0, 0.0, n_calib, 0, eps, delta)


def evaluate_untouched_test(
    calib_risk: np.ndarray,
    calib_wrong: np.ndarray,
    test_risk: np.ndarray,
    test_wrong: np.ndarray,
    eps: float,
    delta: float,
    n_start: int = DEFAULT_N_START,
    n_grid: int = 200,
    use_hb: bool = True,
) -> dict:
    """
    Certify on (calib_risk, calib_wrong), then evaluate on the untouched test split.
    Returns a dict with 'certificate', 'test_coverage', 'test_selective_risk',
    'test_n_accepted', 'test_n_total', 'cp_interval'.
    """
    cert = certify_lambda(calib_risk, calib_wrong, eps, delta,
                          n_start=n_start, n_grid=n_grid, use_hb=use_hb)
    result: dict = {"certificate": {"status": cert.status, "threshold": cert.threshold,
                                    "coverage": cert.coverage, "valid": cert.valid}}

    if cert.threshold is not None:
        accepted = test_risk <= cert.threshold
        n_acc = int(np.sum(accepted))
        n_test = len(test_risk)
        k = int(np.sum(test_wrong[accepted]))
        sel_risk = k / n_acc if n_acc > 0 else 0.0
        coverage = n_acc / n_test if n_test > 0 else 0.0
        cp_lo, cp_hi = clopper_pearson(k, n_acc) if n_acc > 0 else (0.0, 1.0)
        result.update({
            "test_coverage": coverage,
            "test_selective_risk": sel_risk,
            "test_n_accepted": n_acc,
            "test_n_total": n_test,
            "cp_interval": [cp_lo, cp_hi],
        })
    else:
        result.update({
            "test_coverage": 0.0,
            "test_selective_risk": float("nan"),
            "test_n_accepted": 0,
            "test_n_total": len(test_risk),
            "cp_interval": [0.0, 1.0],
        })
    return result


# ── Backward-compat alias ─────────────────────────────────────────────────────
p_value = binom_pvalue
