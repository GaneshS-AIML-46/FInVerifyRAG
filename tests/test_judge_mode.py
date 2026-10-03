import pytest
from unittest.mock import MagicMock
from common.schemas import ScoredAtom, Atom, VerifiedAtom, Thresholds, GeneratedAnswer
from decision.combine import decide, AnswerDecision
from judge.schemas import JudgeResult, VERIFIED, NOT_VERIFIED
from decision.router import SUPPORTED, UNCERTAIN, ABSTAINED

def test_selective_vs_judge_all():
    # Setup mock judge
    judge = MagicMock()
    # Judge will always verify atoms
    judge.judge_batch.side_effect = lambda uncertain, rr: {
        sa.verified.atom.atom_id: JudgeResult(
            verdict=VERIFIED, rationale="test", evidence_quote="test", atom_id=sa.verified.atom.atom_id, risk=sa.risk, parse_error=False
        )
        for sa in uncertain
    }
    
    # Create 3 atoms with risks: 0.2 (SUPPORTED), 0.5 (UNCERTAIN), 0.9 (ABSTAINED)
    a1 = ScoredAtom(risk=0.2, verified=VerifiedAtom(atom=Atom(atom_id="1", type="RATE", text="1", claim="1", span=(0,1)), v1_status="VERIFIED"))
    a2 = ScoredAtom(risk=0.5, verified=VerifiedAtom(atom=Atom(atom_id="2", type="RATE", text="2", claim="2", span=(0,1)), v1_status="VERIFIED"))
    a3 = ScoredAtom(risk=0.9, verified=VerifiedAtom(atom=Atom(atom_id="3", type="RATE", text="3", claim="3", span=(0,1)), v1_status="VERIFIED"))
    
    scored_atoms = [a1, a2, a3]
    
    thresholds = {
        "*|*": Thresholds(stratum_name="*|*", accept_below=0.35, abstain_above=0.7)
    }
    
    answer = GeneratedAnswer(query="test", answer_text="test", citations=[], token_logprobs=[], model_id="")
    
    # 1. Test selective mode (default)
    ad_selective = decide(scored_atoms, thresholds, judge, rr=None, answer=answer, mode="selective")
    
    # Only a2 (risk 0.5) should be sent to the judge
    assert ad_selective.judge_calls == 1
    # a1 is SUPPORTED, a2 is VERIFIED, a3 is ABSTAINED
    assert ad_selective.n_supported == 1
    assert ad_selective.n_verified == 1
    assert ad_selective.n_abstained == 1
    
    # 2. Test llm_judge_all mode
    judge.reset_mock()
    ad_all = decide(scored_atoms, thresholds, judge, rr=None, answer=answer, mode="llm_judge_all")
    
    # All 3 atoms should be sent to the judge
    assert ad_all.judge_calls == 3
    # All 3 atoms should be VERIFIED because the mock judge returns VERIFIED for all
    assert ad_all.n_supported == 0
    assert ad_all.n_verified == 3
    assert ad_all.n_abstained == 0
