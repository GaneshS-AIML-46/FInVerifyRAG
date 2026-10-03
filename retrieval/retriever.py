"""
retrieval/retriever.py
-----------------------
M2: Master retrieval pipeline: BM25 + Dense → RRF → Temporal filter →
    Phase 6 entropy soft-ranking → Cross-encoder reranking.

Phase 6 changes
---------------
After the hard temporal filter, chunks are soft-ranked by
``temporal_entropy_score`` (lower = more temporally relevant) before
being passed to the cross-encoder reranker.  This biases reranking
toward recent, unambiguous, still-active regulatory text without
discarding any temporally valid evidence.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import List, Optional

from common.schemas import Chunk, RetrievalResult
from retrieval.bm25 import BM25Index
from retrieval.dense import DenseIndex
from retrieval.fusion import rrf
from retrieval.rerank import Reranker
from retrieval.temporal import temporal_filter, temporal_entropy_score
from ingest.version_graph.build import VersionGraph

logger = logging.getLogger("retrieval.retriever")


class Retriever:
    """
    Master retrieval pipeline combining dense, sparse, RRF, temporal filters,
    entropy soft-ranking, and cross-encoder reranking.
    """

    def __init__(
        self,
        bm25_index:    BM25Index,
        dense_index:   DenseIndex,
        reranker:      Reranker,
        version_graph: VersionGraph,
    ):
        self.bm25          = bm25_index
        self.dense         = dense_index
        self.reranker      = reranker
        self.version_graph = version_graph

        # Load chunks from BM25 index as canonical store
        self.chunk_store = self.bm25.chunks

    def retrieve(
        self,
        query:         str,
        query_date:    Optional[str] = None,
        top_k_dense:   int = 50,
        top_k_bm25:    int = 50,
        rerank_top_in: int = 30,
        final_k:       int = 6,
    ) -> RetrievalResult:
        """
        Run the full retrieval pipeline.

        Pipeline stages
        ---------------
        1. BM25 sparse search.
        2. Dense embedding search.
        3. Reciprocal Rank Fusion (k=60).
        4. Hard temporal filter — drop superseded / out-of-window chunks.
        5. Phase 6: Soft temporal entropy sort — prefer recent, unambiguous
           chunks before presenting to the cross-encoder.
        6. Cross-encoder reranking → top final_k.
        """
        if not query_date:
            raise ValueError("query_date must be provided.")

        # Stage 1: Sparse search
        bm25_ids = self.bm25.search(query, top_k=top_k_bm25)

        # Stage 2: Dense search
        dense_ids = self.dense.search(query, top_k=top_k_dense)

        # Stage 3: RRF fusion
        fused_ids = rrf([bm25_ids, dense_ids])
        candidates = [self.chunk_store[cid] for cid in fused_ids if cid in self.chunk_store]

        # Stage 4: Hard temporal filter
        time_filtered = temporal_filter(candidates, self.version_graph, query_date)

        # Stage 5 (Phase 6): Soft temporal entropy sort
        # Sort ascending by entropy score — lower = more temporally relevant
        entropy_sorted = sorted(
            time_filtered,
            key=lambda c: temporal_entropy_score(c, query_date),
        )

        # Take top `rerank_top_in` for cross-encoder
        rerank_candidates = entropy_sorted[:rerank_top_in]

        # Stage 6: Cross-encoder reranking
        final_chunks = self.reranker.rerank(query, rerank_candidates, top_k=final_k)

        logger.info(
            "Retrieve: candidates=%d  after_temporal=%d  after_entropy_sort=%d  final=%d",
            len(candidates), len(time_filtered), len(rerank_candidates), len(final_chunks),
        )

        return RetrievalResult(
            query=query,
            query_date=query_date,
            chunks=final_chunks,
            metadata={
                "total_retrieved_before_temporal": len(candidates),
                "after_temporal":                  len(time_filtered),
                "after_entropy_sort":              len(rerank_candidates),
            },
        )
