"""
Unit tests for VectorDB
"""
import numpy as np
import tempfile
import shutil
from pathlib import Path
from pocketvectordb import VectorDB, cosine_similarity


def test_initialization():
    """Test database initialization"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = VectorDB(storage_path=tmpdir, dimension=128)
        assert db.count() == 0
        assert db.dimension == 128
        print("✓ Initialization test passed")


def test_add_document():
    """Test adding a single document"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = VectorDB(storage_path=tmpdir)
        
        embedding = np.random.randn(128)
        doc_id = db.add(
            embedding=embedding,
            text="Test document",
            metadata={"category": "test"}
        )
        
        assert db.count() == 1
        assert doc_id in db.documents
        print("✓ Add document test passed")


def test_add_batch():
    """Test batch addition"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = VectorDB(storage_path=tmpdir, dimension=128)
        
        embeddings = [np.random.randn(128) for _ in range(10)]
        texts = [f"Document {i}" for i in range(10)]
        metadatas = [{"index": i} for i in range(10)]
        
        doc_ids = db.add_batch(embeddings, metadatas, texts)
        
        assert len(doc_ids) == 10
        assert db.count() == 10
        print("✓ Batch add test passed")


def test_query():
    """Test similarity search"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = VectorDB(storage_path=tmpdir, dimension=128)
        
        # Add similar documents
        base_embedding = np.random.randn(128)
        
        # Add documents with small variations
        for i in range(5):
            embedding = base_embedding + np.random.randn(128) * 0.1
            db.add(embedding, text=f"Document {i}", metadata={"id": i})
        
        # Query with similar embedding
        query_embedding = base_embedding + np.random.randn(128) * 0.05
        results = db.query(query_embedding, n_results=3)
        
        assert len(results['ids']) == 3
        assert len(results['documents']) == 3
        assert len(results['distances']) == 3
        print("✓ Query test passed")


def test_metadata_filtering():
    """Test querying with metadata filter"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = VectorDB(storage_path=tmpdir, dimension=128)
        
        # Add documents with different categories
        for i in range(10):
            embedding = np.random.randn(128)
            category = "A" if i < 5 else "B"
            db.add(embedding, text=f"Doc {i}", metadata={"category": category})
        
        # Query with filter
        query_embedding = np.random.randn(128)
        results = db.query(query_embedding, n_results=10, where={"category": "A"})
        
        assert len(results['ids']) == 5
        assert all(m['category'] == "A" for m in results['metadatas'])
        print("✓ Metadata filtering test passed")


def test_get_operations():
    """Test get operations"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = VectorDB(storage_path=tmpdir, dimension=128)
        
        # Add documents
        doc_ids = []
        for i in range(5):
            doc_id = db.add(
                np.random.randn(128),
                text=f"Doc {i}",
                metadata={"index": i}
            )
            doc_ids.append(doc_id)
        
        # Get by IDs
        results = db.get(doc_ids=[doc_ids[0], doc_ids[1]])
        assert len(results['ids']) == 2
        
        # Get by metadata
        results = db.get(where={"index": 2})
        assert len(results['ids']) == 1
        
        # Get all
        results = db.get()
        assert len(results['ids']) == 5
        
        print("✓ Get operations test passed")


def test_update():
    """Test update operations"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = VectorDB(storage_path=tmpdir, dimension=128)
        
        # Add document
        doc_id = db.add(
            np.random.randn(128),
            text="Original",
            metadata={"status": "old"}
        )
        
        # Update
        db.update(
            doc_id,
            text="Updated",
            metadata={"status": "new", "updated": True}
        )
        
        # Verify
        results = db.get(doc_ids=[doc_id])
        assert results['documents'][0] == "Updated"
        assert results['metadatas'][0]['status'] == "new"
        assert results['metadatas'][0]['updated'] == True
        
        print("✓ Update test passed")


def test_delete():
    """Test delete operations"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = VectorDB(storage_path=tmpdir, dimension=128)
        
        # Add documents
        doc_ids = []
        for i in range(10):
            category = "delete" if i < 5 else "keep"
            doc_id = db.add(
                np.random.randn(128),
                metadata={"category": category}
            )
            doc_ids.append(doc_id)
        
        # Delete by metadata
        deleted = db.delete(where={"category": "delete"})
        assert deleted == 5
        assert db.count() == 5
        
        # Delete by IDs
        deleted = db.delete(doc_ids=[doc_ids[5]])
        assert deleted == 1
        assert db.count() == 4
        
        print("✓ Delete test passed")


def test_persistence():
    """Test save and load"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and populate database
        db1 = VectorDB(storage_path=tmpdir, dimension=128)
        
        embeddings = [np.random.randn(128) for _ in range(5)]
        texts = [f"Doc {i}" for i in range(5)]
        db1.add_batch(embeddings, texts=texts)
        db1.save()
        
        original_count = db1.count()
        
        # Load in new instance
        db2 = VectorDB(storage_path=tmpdir)
        
        assert db2.count() == original_count
        assert db2.dimension == 128
        
        # Verify data integrity
        results = db2.get()
        assert len(results['ids']) == 5
        
        print("✓ Persistence test passed")


def test_cosine_similarity():
    """Test cosine similarity function"""
    a = np.array([1, 0, 0])
    b = np.array([1, 0, 0])
    assert abs(cosine_similarity(a, b) - 1.0) < 1e-6
    
    a = np.array([1, 0, 0])
    b = np.array([0, 1, 0])
    assert abs(cosine_similarity(a, b) - 0.0) < 1e-6
    
    print("✓ Cosine similarity test passed")


def test_empty_query():
    """Test querying empty database"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = VectorDB(storage_path=tmpdir, dimension=128)
        
        query_embedding = np.random.randn(128)
        results = db.query(query_embedding, n_results=5)
        
        assert len(results['ids']) == 0
        assert len(results['documents']) == 0
        
        print("✓ Empty query test passed")


def test_dimension_validation():
    """Test dimension validation"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = VectorDB(storage_path=tmpdir, dimension=128)
        
        # Add first document
        db.add(np.random.randn(128))
        
        # Try to add document with wrong dimension
        try:
            db.add(np.random.randn(256))
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "dimension" in str(e).lower()
        
        print("✓ Dimension validation test passed")


def run_all_tests():
    """Run all tests"""
    print("Running VectorDB Tests...")
    print("=" * 60)
    
    test_initialization()
    test_add_document()
    test_add_batch()
    test_query()
    test_metadata_filtering()
    test_get_operations()
    test_update()
    test_delete()
    test_persistence()
    test_cosine_similarity()
    test_empty_query()
    test_dimension_validation()
    
    print("=" * 60)
    print("All tests passed! ✓")


if __name__ == "__main__":
    run_all_tests()
