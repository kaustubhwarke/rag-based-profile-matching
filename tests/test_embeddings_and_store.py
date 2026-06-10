from __future__ import annotations

import numpy as np

from profile_matching.embeddings.providers import HashingEmbeddings
from profile_matching.vectorstore.base import VectorRecord
from profile_matching.vectorstore.memory_store import InMemoryVectorStore


def test_hashing_embeddings_are_normalised():
    emb = HashingEmbeddings(dimension=128)
    vecs = emb.embed_documents(["python aws kubernetes", "java spring kafka"])
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    assert vecs.shape == (2, 128)


def test_hashing_similarity_is_semantically_ordered():
    emb = HashingEmbeddings(dimension=512)
    q = emb.embed_query("python machine learning aws")
    docs = emb.embed_documents(
        ["python machine learning aws docker", "graphic design photoshop illustrator"]
    )
    sims = docs @ q
    assert sims[0] > sims[1]


def test_memory_store_query_and_filter():
    store = InMemoryVectorStore()
    emb = HashingEmbeddings(dimension=256)
    vecs = emb.embed_documents(["python aws", "java kafka", "python django"])
    store.upsert([
        VectorRecord(id="a", embedding=vecs[0], document="python aws", metadata={"role": "ml"}),
        VectorRecord(id="b", embedding=vecs[1], document="java kafka", metadata={"role": "be"}),
        VectorRecord(id="c", embedding=vecs[2], document="python django", metadata={"role": "ml"}),
    ])
    assert store.count() == 3

    hits = store.query(emb.embed_query("python aws"), top_k=3)
    assert hits[0].id == "a"

    filtered = store.query(emb.embed_query("python"), top_k=3, where={"role": "be"})
    assert all(h.metadata["role"] == "be" for h in filtered)


def test_memory_store_upsert_is_idempotent():
    store = InMemoryVectorStore()
    emb = HashingEmbeddings(dimension=64)
    v = emb.embed_documents(["x"])[0]
    store.upsert([VectorRecord(id="a", embedding=v, document="x", metadata={})])
    store.upsert([VectorRecord(id="a", embedding=v, document="x2", metadata={})])
    assert store.count() == 1
