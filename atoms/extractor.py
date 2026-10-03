from typing import List, Optional
from common.schemas import Atom, GeneratedAnswer
from common.llm_client import LLMClient
from atoms.splitter import AtomSplitter

_splitter: Optional[AtomSplitter] = None

def get_atom_splitter(config: Optional[dict] = None) -> AtomSplitter:
    global _splitter
    if _splitter is None:
        _splitter = AtomSplitter(llm=LLMClient(config=config))
    return _splitter

def extract(answer: GeneratedAnswer, config: Optional[dict] = None) -> List[Atom]:
    """Extract discrete atomic facts from a generated regulatory answer."""
    splitter = get_atom_splitter(config=config)
    return splitter.split(answer)
