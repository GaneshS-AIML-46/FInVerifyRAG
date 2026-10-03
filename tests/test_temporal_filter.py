import pytest
from common.schemas import Chunk
from ingest.version_graph.build import VersionGraph, VersionNode
from retrieval.temporal import temporal_filter

def test_temporal_filter_superseded():
    c1 = Chunk(chunk_id="doc1", text="Old rate is 5%", regulator="RBI", issue_date="2020-01-01",
               effective_from="2020-01-01", effective_to="2022-01-01", source_url="http://rbi/doc1", metadata={"section_id": "RBI_Sec1", "doc_id": "doc1"})
    c2 = Chunk(chunk_id="doc2", text="New rate is 6%", regulator="RBI", issue_date="2022-01-01",
               effective_from="2022-01-01", effective_to=None, source_url="http://rbi/doc2", metadata={"section_id": "RBI_Sec1", "doc_id": "doc2"})
    
    graph = VersionGraph()
    graph.add_node(VersionNode(doc_id="doc1", section_id="RBI_Sec1", effective_from="2020-01-01", effective_to="2022-01-01", supersedes=[]))
    graph.add_node(VersionNode(doc_id="doc2", section_id="RBI_Sec1", effective_from="2022-01-01", effective_to=None, supersedes=["doc1"]))
    
    chunks = [c1, c2]
    
    # Query in 2021: doc1 is active, doc2 is not yet effective
    valid_2021 = temporal_filter(chunks, graph, "2021-06-01")
    assert len(valid_2021) == 1
    assert valid_2021[0].chunk_id == "doc1"
    
    # Query in 2023: doc1 is expired/superseded, doc2 is active
    valid_2023 = temporal_filter(chunks, graph, "2023-06-01")
    assert len(valid_2023) == 1
    assert valid_2023[0].chunk_id == "doc2"
