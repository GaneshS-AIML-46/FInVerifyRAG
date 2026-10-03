import json
import random
import os

types = ["RATE", "THRESHOLD", "SECTION", "DATE", "ENTITY", "APPLICABILITY"]

# generate 250 docs
docs = [f"doc_{i}" for i in range(250)]
atoms = []
aid = 1

for doc in docs:
    # 5 to 15 atoms per doc
    num = random.randint(5, 15)
    for _ in range(num):
        atoms.append({
            "id": f"a{aid}",
            "question_id": f"q{aid//3}",
            "doc_id": doc,
            "atom_type": random.choice(types),
            "atom_text": f"Mock atom text {aid}",
            "cited_chunk_ids": [f"{doc}_chunk1"]
        })
        aid += 1

os.makedirs("scratch", exist_ok=True)
with open("scratch/mock_raw_atoms.jsonl", "w") as f:
    for a in atoms:
        f.write(json.dumps(a) + "\n")
        
print(f"Generated {len(atoms)} mock atoms.")
