import numpy as np
from guarantee.ltt import calibrate_ltt, p_value, min_accepted, StrataStatus, build_mondrian_thresholds, print_stratum_status

def run_tests():
    eps = 0.05
    delta = 0.10
    min_acc = min_accepted(eps, delta)
    print(f"min_accepted(eps={eps}, delta={delta}) = {min_acc}")

    # (a) Well-separated scores: 0-0.4 are correct, 0.6-1.0 are wrong.
    # We provide enough atoms (e.g. 50 correct).
    risks_a = np.concatenate([np.random.uniform(0.0, 0.4, 50), np.random.uniform(0.6, 1.0, 50)])
    labels_a = np.concatenate([np.zeros(50), np.ones(50)])
    res_a = calibrate_ltt(risks_a, labels_a, eps, delta)
    assert res_a.status == StrataStatus.CERTIFIED, f"Test A failed: {res_a.status}"
    assert res_a.accept_below >= 0.3 and res_a.accept_below < 0.6, f"Got {res_a.accept_below}"
    print(f"Test (a) passed: Certified threshold = {res_a.accept_below:.4f}")

    # (b) Stratum too small that must be merged.
    # Provide only 30 atoms total (< min_acc = 45).
    risks_b = np.random.uniform(0.0, 0.5, 30)
    labels_b = np.zeros(30)
    res_b = calibrate_ltt(risks_b, labels_b, eps, delta)
    assert res_b.status == StrataStatus.MERGED, f"Test B failed: {res_b.status}"
    print(f"Test (b) passed: Status = {res_b.status}")

    # (c) No threshold is safe -> not_certifiable
    # Even low risks have high error rates.
    risks_c = np.random.uniform(0.0, 1.0, 100)
    labels_c = np.ones(100) # all wrong
    res_c = calibrate_ltt(risks_c, labels_c, eps, delta)
    assert res_c.status == StrataStatus.NOT_CERTIFIABLE, f"Test C failed: {res_c.status}"
    print(f"Test (c) passed: Status = {res_c.status}")

    # (d) Guarantee must FAIL if the threshold is set too high.
    # E.g., if we manually try a threshold that accepts 50 atoms but 10 are wrong
    # (error rate 10/50 = 20% > 5%), p-value should be > delta, rejecting the threshold.
    n_acc = 50
    k_wrong = 10
    pval = p_value(k_wrong, n_acc, eps)
    assert pval > delta, f"Test D failed: pval {pval} <= delta {delta}"
    print(f"Test (d) passed: p-value {pval:.4f} > {delta} when error is high")
    
    # Test mondrian table
    rows = []
    # 50 correct DATE
    for r in np.random.uniform(0.0, 0.4, 50):
        rows.append({"risk": r, "label": "supported", "atom_type": "DATE"})
    # 10 correct RATE (too small, will merge to *|* or parent)
    for r in np.random.uniform(0.0, 0.4, 10):
        rows.append({"risk": r, "label": "supported", "atom_type": "RATE"})
    # 50 wrong DATE
    for r in np.random.uniform(0.6, 1.0, 50):
        rows.append({"risk": r, "label": "unsupported", "atom_type": "DATE"})
        
    print("\nStratum Status Table:")
    print_stratum_status(rows, eps, delta)

if __name__ == "__main__":
    run_tests()
