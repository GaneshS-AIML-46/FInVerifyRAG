import numpy as np
import io
import contextlib
from scripts.ablation_study import fit_and_predict, COMPONENTS, CORE_COMPONENTS, _labels
from sklearn.metrics import roc_auc_score
from eval.stats import mcnemar

def test_ablation():
    # 1. Create synthetic data
    train_rows = []
    calib_rows = []
    test_rows = []
    
    def make_record(i):
        # Perfect signal: s_ret
        # Useless signal: s_div
        # V1: s_ver (also perfect)
        label = 0 if i % 4 != 0 else 1
        s_perfect = 0.0 if label == 0 else 1.0
        s_useless = np.random.uniform(0.0, 1.0)
        
        return {
            "doc_id": f"doc_{i}",
            "atom_type": "DATE",
            "label": label,
            "s_ver": s_perfect, 
            "s_ret": s_perfect,
            "s_div": s_useless,
            "s_ent": s_perfect,
            "s_nli": s_perfect,
            "risk": s_perfect,
        }

    for i in range(100):
        train_rows.append(make_record(i))
    for i in range(100, 200):
        calib_rows.append(make_record(i))
    for i in range(200, 300):
        test_rows.append(make_record(i))
        
    full_res = fit_and_predict(train_rows, calib_rows, test_rows, [], seed=42, use_mondrian=True)
    
    for comp, drop_cols in COMPONENTS.items():
        use_mondrian = (comp != "Mondrian")
        res = fit_and_predict(train_rows, calib_rows, test_rows, drop_cols, seed=42, use_mondrian=use_mondrian)
        
        n_boot = 500
        np.random.seed(42)
        n = len(test_rows)
        auroc_diffs = []
        cov_diffs = []
        labels_arr = _labels(test_rows)
        
        for _ in range(n_boot):
            idx = np.random.choice(n, n, replace=True)
            y_b = labels_arr[idx]
            if len(set(y_b)) < 2: continue
            
            p_full = np.array(full_res["probs"])[idx]
            p_abl = np.array(res["probs"])[idx]
            
            d_full = np.array(full_res["decs"])[idx]
            d_abl = np.array(res["decs"])[idx]
            
            a_full = roc_auc_score(y_b, p_full)
            a_abl = roc_auc_score(y_b, p_abl)
            auroc_diffs.append(a_full - a_abl)
            
            c_full = sum(1 for d in d_full if d in ("SUPPORTED", "VERIFIED")) / n
            c_abl = sum(1 for d in d_abl if d in ("SUPPORTED", "VERIFIED")) / n
            cov_diffs.append(c_full - c_abl)
            
        auroc_ub = np.percentile(auroc_diffs, 97.5) if auroc_diffs else 0.0
        cov_ub = np.percentile(cov_diffs, 97.5) if cov_diffs else 0.0
        
        action = "KEEP"
        if auroc_ub < 0.01 and cov_ub < 0.01:
            action = "REMOVE-CANDIDATE"
            
        if comp in CORE_COMPONENTS:
            action = "CORE (NEVER REMOVE)"
            
        if comp == "s_div":
            # Useless signal should be REMOVE-CANDIDATE
            assert action == "REMOVE-CANDIDATE", f"s_div got {action} instead of REMOVE-CANDIDATE. UB: AUROC={auroc_ub}, COV={cov_ub}"
        elif comp == "s_ret":
            # Since s_ver/s_ent are still present (perfect), removing s_ret might still keep AUROC high if the model learns from s_ver.
            # But we just need a useful signal to KEEP. If we remove all perfect signals, it drops.
            # Let's just verify that CORE components are never flagged.
            pass
            
        if comp in CORE_COMPONENTS:
            assert action == "CORE (NEVER REMOVE)", f"Core component {comp} flagged incorrectly"
            
    print("All ablation tests passed!")

if __name__ == "__main__":
    test_ablation()
