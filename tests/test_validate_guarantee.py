import numpy as np
from guarantee.validate import validate_guarantee
from guarantee.ltt import build_mondrian_thresholds, StrataStatus
from eval.metrics import clopper_pearson

def run_tests():
    eps = 0.05
    delta = 0.10
    
    # 1. Clopper-Pearson check: 0 errors in 16 should be (0, 0.17~0.21)
    # The actual exact value for alpha=0.05 is beta.ppf(0.975, 1, 16) ~ 0.2059
    # or one-sided? We use two-sided alpha=0.05, so 0.0 to ~0.206.
    ci = clopper_pearson(0, 16, 0.05)
    assert ci[0] == 0.0
    assert 0.17 < ci[1] < 0.21, f"Expected upper bound around 0.2, got {ci[1]}"
    print(f"Test CP passed: {ci}")
    
    # 2. validate_guarantee checks
    # Generate 1000 atoms across 100 documents
    calib_pool = []
    for i in range(100):
        doc_id = f"doc_{i}"
        for j in range(10): # 10 atoms per doc
            # Safe data: risk < 0.3 is always correct. risk > 0.6 is always wrong.
            if i < 50:
                # low risk
                risk = np.random.uniform(0.0, 0.2)
                label = "supported"
            else:
                # high risk
                risk = np.random.uniform(0.7, 1.0)
                label = "unsupported"
            calib_pool.append({
                "doc_id": doc_id,
                "atom_type": "DATE",
                "risk": risk,
                "label": label
            })
            
    # Test a calibrated threshold
    vr_res = validate_guarantee(calib_pool, eps, delta, n_splits=50)
    assert vr_res["violation_rate"] <= delta, f"Expected violation_rate <= {delta}, got {vr_res['violation_rate']}"
    assert vr_res["valid"] is True
    print(f"Test calibrated threshold passed: violation_rate={vr_res['violation_rate']} valid={vr_res['valid']}")
    
    # Test deliberately too-lax threshold (simulate by tricking the function, or providing bad data where low risk is wrong)
    # Make all low-risk items actually wrong. The threshold will pick them (because LTT overfits to calib set in small samples? 
    # Actually, LTT should abstain if it's too risky. 
    # To force a violation, we need LTT to accept, but then validation fails. 
    # If the calibrate split happens to have 0 errors, it will accept them, but the validate split has errors.
    # We can create a dataset where doc_0..doc_49 are perfectly correct at low risk, but doc_50..doc_99 have wrong at low risk.
    # LTT will calibrate on docs, if it gets mostly good docs it accepts, then on bad docs it fails.
    bad_pool = []
    for i in range(100):
        doc_id = f"doc_{i}"
        for j in range(20):
            risk = np.random.uniform(0.0, 0.2)
            # Make half the docs have 100% error rate, half have 0% error rate
            label = "unsupported" if i % 2 == 0 else "supported"
            bad_pool.append({
                "doc_id": doc_id,
                "atom_type": "DATE",
                "risk": risk,
                "label": label
            })
            
    vr_bad = validate_guarantee(bad_pool, eps, delta, n_splits=50)
    # The violation rate should be high because half the time we train on a safe subset and validate on a bad subset.
    # Wait, if we shuffle, calib will always have ~50% bad docs, so LTT will abstain.
    # So LTT will always abstain! Thus coverage = 0.
    
    print(f"Test bad data (should abstain): coverage={vr_bad['mean_coverage']} valid={vr_bad['valid']}")
    assert vr_bad["valid"] is False, "Expected valid=False when coverage is 0"
    
    import unittest.mock as mock
    
    with mock.patch("guarantee.validate.build_mondrian_thresholds") as mock_build:
        from common.schemas import Thresholds
        from guarantee.ltt import LTTResult
        
        # Mock a lax threshold that accepts everything (risk <= 1.0)
        lax_res = LTTResult(1.0, StrataStatus.CERTIFIED, 1000, 1000, eps, delta)
        mock_build.return_value = {"DATE": lax_res}
        
        vr_lax = validate_guarantee(calib_pool, eps, delta, n_splits=50)
        # Should have high violation rate since it accepts the wrong items
        assert vr_lax["violation_rate"] > delta, f"Expected violation rate > {delta}, got {vr_lax['violation_rate']}"
        print(f"Test deliberately too-lax passed: violation_rate={vr_lax['violation_rate']}")

if __name__ == "__main__":
    run_tests()
