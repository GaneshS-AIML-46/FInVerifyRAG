import numpy as np
from eval.systems import system_row, risk_vector
from eval.baselines import plain_rag, selfcheck, nli_only, conformal_no_mondrian

def test_systems():
    eps = 0.05
    
    # 1. Create synthetic data
    cal_records = []
    test_records = []
    
    def make_record(i, is_cal=False):
        # Base error rate ~ 25%
        label = "supported" if i % 4 != 0 else "unsupported"
        
        # Perfect signal: risk = 0 for correct, 1 for wrong
        s_perfect = 0.0 if label == "supported" else 1.0
        
        # Random signal: uninformative
        s_random = np.random.uniform(0.0, 1.0)
        
        # Bad signal: opposite of perfect
        s_bad = 1.0 if label == "supported" else 0.0
        
        return {
            "doc_id": f"doc_{i}",
            "atom_type": "DATE",
            "label": label,
            "s_ver": s_perfect, # use as perfect
            "s_nli": s_random,  # nli_only will use this -> random performance
            "s_ent": s_bad,     # selfcheck will use this -> bad performance
            "s_ret": 0.5,
            "s_div": 0.5,
            "risk": s_perfect,  # RegGuard
        }

    for i in range(200):
        cal_records.append(make_record(i, is_cal=True))
    for i in range(200, 400):
        test_records.append(make_record(i, is_cal=False))
        
    # Test plain_rag coverage
    rag_row = system_row("plain_rag", cal_records, test_records, eps)
    assert rag_row["coverage"] == 1.0, f"Expected coverage 1.0 for plain_rag, got {rag_row['coverage']}"
    print("plain_rag coverage is 1.0")
    
    # Check that systems produce different AURCs
    regguard_row = system_row("RegGuard (ours)", cal_records, test_records, eps)
    selfcheck_row = system_row("selfcheck", cal_records, test_records, eps)
    nli_row = system_row("nli_only", cal_records, test_records, eps)
    
    # RegGuard should have excellent AURC (perfect signal)
    # selfcheck should have terrible AURC (bad signal)
    print(f"RegGuard AURC: {regguard_row['aurc']}")
    print(f"Selfcheck AURC: {selfcheck_row['aurc']}")
    assert regguard_row["aurc"] < selfcheck_row["aurc"], "Expected RegGuard to have better (lower) AURC than Selfcheck"
    print("Different AURCs observed based on signal quality")
    
    # NLI only uses random risk, so AURC should be near base error rate (0.25)
    base_error = sum(1 for r in test_records if r["label"] == "unsupported") / len(test_records)
    print(f"NLI_only (random) AURC: {nli_row['aurc']}, Base Error: {base_error}")
    assert abs(nli_row["aurc"] - base_error) < 0.1, f"Expected AURC near {base_error}, got {nli_row['aurc']}"
    print("Random risk has AURC near base error rate")

if __name__ == "__main__":
    test_systems()
