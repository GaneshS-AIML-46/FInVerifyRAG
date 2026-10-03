import re

def update_run_eval_dates():
    with open("S:\\FYP\\M0_M10Complete_pipeline\\scripts\\run_full_eval.py", "r", encoding="utf-8") as f:
        code = f.read()

    # 1. Inject doc_dates logic into _old_chunk_to_new
    
    old_func = """def _old_chunk_to_new(oc):
    from common.schemas import Chunk
    meta = oc.metadata or {}
    issuer = meta.get("issuer", "").upper()
    reg = "RBI" if "RBI" in issuer else ("SEBI" if "SEBI" in issuer else issuer or "UNKNOWN")
    yr = str(meta.get("year", ""))
    iss = f"{yr}-01-01" if yr.isdigit() else "1970-01-01"
    return Chunk(chunk_id=oc.chunk_id, text=oc.text, regulator=reg,
                 issue_date=iss, source_url=meta.get("file_path", ""), metadata=meta)"""

    new_func = """
_DOC_DATES = None
def _load_doc_dates():
    global _DOC_DATES
    if _DOC_DATES is not None:
        return _DOC_DATES
    import csv, os
    _DOC_DATES = {}
    path = "data/doc_dates.csv"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                _DOC_DATES[row["doc_id"]] = {
                    "effective_from": row["effective_from"] or None,
                    "effective_to": row["effective_to"] or None
                }
    return _DOC_DATES

def _old_chunk_to_new(oc):
    from common.schemas import Chunk
    meta = oc.metadata or {}
    issuer = meta.get("issuer", "").upper()
    reg = "RBI" if "RBI" in issuer else ("SEBI" if "SEBI" in issuer else issuer or "UNKNOWN")
    yr = str(meta.get("year", ""))
    iss = f"{yr}-01-01" if yr.isdigit() else "1970-01-01"
    
    doc_dates = _load_doc_dates()
    doc_id = meta.get("doc_id", "")
    eff_from = doc_dates.get(doc_id, {}).get("effective_from") or iss
    eff_to = doc_dates.get(doc_id, {}).get("effective_to")
    
    # Store them in metadata so VersionGraph and temporal filters can use them.
    meta["effective_from"] = eff_from
    if eff_to:
        meta["effective_to"] = eff_to
        
    return Chunk(chunk_id=oc.chunk_id, text=oc.text, regulator=reg,
                 issue_date=iss, effective_from=eff_from, effective_to=eff_to,
                 source_url=meta.get("file_path", ""), metadata=meta)"""

    if "def _old_chunk_to_new(oc):" in code:
        # replace just the function
        code = re.sub(r'def _old_chunk_to_new\(oc\):.*?return Chunk\([^)]+\)', new_func, code, flags=re.DOTALL)
    else:
        print("Could not find _old_chunk_to_new")

    # 2. Modify QUERIES to remove the default 2024-01-01, unless it's explicitly there. But wait, QUERIES already has query_date. We need to raise an error if not present in the custom file. Wait, that's already in main():
    
    with open("S:\\FYP\\M0_M10Complete_pipeline\\scripts\\run_full_eval.py", "w", encoding="utf-8") as f:
        f.write(code)

if __name__ == '__main__':
    update_run_eval_dates()
