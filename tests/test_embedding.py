"""Tests for scene_db.embedding."""

import math
import sqlite3

import pytest

from scene_db.embedding import (
    EMBEDDING_DIM,
    _cosine_similarity,
    _decode_embedding,
    _encode_embedding,
    ensure_embedding_table,
)


class TestEncodeDecodeEmbedding:
    def test_roundtrip(self):
        vec = [1.0, 2.0, 3.0, -0.5, 0.0]
        encoded = _encode_embedding(vec)
        decoded = _decode_embedding(encoded)
        assert len(decoded) == len(vec)
        for a, b in zip(vec, decoded):
            assert a == pytest.approx(b)

    def test_empty_vector(self):
        vec = []
        encoded = _encode_embedding(vec)
        decoded = _decode_embedding(encoded)
        assert decoded == []

    def test_single_element(self):
        vec = [42.0]
        decoded = _decode_embedding(_encode_embedding(vec))
        assert decoded == pytest.approx([42.0])

    def test_large_vector(self):
        vec = list(range(EMBEDDING_DIM))
        vec = [float(x) for x in vec]
        decoded = _decode_embedding(_encode_embedding(vec))
        assert len(decoded) == EMBEDDING_DIM
        assert decoded == pytest.approx(vec)

    def test_encoded_is_bytes(self):
        vec = [1.0, 2.0, 3.0]
        encoded = _encode_embedding(vec)
        assert isinstance(encoded, bytes)
        # 3 floats * 4 bytes each
        assert len(encoded) == 12


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = [1.0, 2.0]
        b = [-1.0, -2.0]
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        assert _cosine_similarity(a, b) == 0.0
        assert _cosine_similarity(b, a) == 0.0

    def test_both_zero_vectors(self):
        assert _cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0

    def test_known_angle(self):
        # 45 degrees -> cos(45) = sqrt(2)/2
        a = [1.0, 0.0]
        b = [1.0, 1.0]
        expected = 1.0 / math.sqrt(2)
        assert _cosine_similarity(a, b) == pytest.approx(expected)


class TestEnsureEmbeddingTable:
    def test_creates_table(self):
        conn = sqlite3.connect(":memory:")
        # Need scene_chunks table first for FK reference
        conn.execute("CREATE TABLE scene_chunks (id TEXT PRIMARY KEY)")
        ensure_embedding_table(conn)
        # Verify table exists by inserting
        conn.execute(
            "INSERT INTO scene_embeddings (scene_id, embedding) VALUES (?, ?)",
            ("test_id", b"\x00"),
        )
        row = conn.execute("SELECT * FROM scene_embeddings").fetchone()
        assert row[0] == "test_id"
        conn.close()

    def test_idempotent(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE scene_chunks (id TEXT PRIMARY KEY)")
        ensure_embedding_table(conn)
        ensure_embedding_table(conn)  # should not raise
        conn.close()
