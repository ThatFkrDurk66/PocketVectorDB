"""
Example usage of the Pocket Vector Database
"""
import numpy as np
from pocketvectordb import VectorDB
import time


def generate_sample_embedding(dimension=384, seed=None):
    """Generate a random embedding for testing"""
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(dimension).astype(np.float32)


def example_basic_usage():
    """Basic CRUD operations"""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Initialize database
    db = VectorDB(storage_path="./my_vectordb", dimension=384)
    print(f"Initialized: {db}\n")
    
    # Add documents
    doc_id_1 = db.add(
        embedding=generate_sample_embedding(384, seed=1),
        text="The quick brown fox jumps over the lazy dog",
        metadata={"category": "animals", "type": "sentence"}
    )
    print(f"Added document 1: {doc_id_1}")
    
    doc_id_2 = db.add(
        embedding=generate_sample_embedding(384, seed=2),
        text="Machine learning is a subset of artificial intelligence",
        metadata={"category": "tech", "type": "sentence"}
    )
    print(f"Added document 2: {doc_id_2}")
    
    doc_id_3 = db.add(
        embedding=generate_sample_embedding(384, seed=3),
        text="Deep learning models require large amounts of data",
        metadata={"category": "tech", "type": "sentence"}
    )
    print(f"Added document 3: {doc_id_3}")
    
    print(f"\nTotal documents: {db.count()}")
    
    # Save to disk
    db.save()
    print("Database saved to disk\n")


def example_batch_operations():
    """Batch operations for efficiency"""
    print("=" * 60)
    print("Example 2: Batch Operations")
    print("=" * 60)
    
    db = VectorDB(storage_path="./my_vectordb")
    db.clear()
    
    # Batch add
    embeddings = [generate_sample_embedding(384, seed=i) for i in range(100)]
    texts = [f"Document {i} about various topics" for i in range(100)]
    metadatas = [{"doc_num": i, "category": "tech" if i % 2 == 0 else "science"} for i in range(100)]
    
    start = time.time()
    doc_ids = db.add_batch(embeddings, metadatas, texts)
    elapsed = time.time() - start
    
    print(f"Added {len(doc_ids)} documents in {elapsed:.4f} seconds")
    print(f"Rate: {len(doc_ids)/elapsed:.2f} docs/second")
    print(f"Total documents: {db.count()}\n")
    
    db.save()


def example_similarity_search():
    """Semantic/similarity search"""
    print("=" * 60)
    print("Example 3: Similarity Search")
    print("=" * 60)
    
    db = VectorDB(storage_path="./my_vectordb")
    
    # Query similar documents
    query_embedding = generate_sample_embedding(384, seed=5)
    
    start = time.time()
    results = db.query(query_embedding, n_results=5)
    elapsed = time.time() - start
    
    print(f"Search completed in {elapsed:.4f} seconds")
    print(f"\nTop 5 similar documents:")
    for i, (doc_id, text, distance) in enumerate(zip(results['ids'], results['documents'], results['distances'])):
        print(f"{i+1}. ID: {doc_id[:8]}... | Distance: {distance:.4f}")
        print(f"   Text: {text}")
    print()


def example_metadata_filtering():
    """Search with metadata filters"""
    print("=" * 60)
    print("Example 4: Metadata Filtering")
    print("=" * 60)
    
    db = VectorDB(storage_path="./my_vectordb")
    
    # Query with metadata filter
    query_embedding = generate_sample_embedding(384, seed=6)
    
    results = db.query(
        query_embedding, 
        n_results=3,
        where={"category": "tech"}
    )
    
    print(f"Found {len(results['ids'])} tech documents:")
    for i, (doc_id, text, metadata) in enumerate(zip(results['ids'], results['documents'], results['metadatas'])):
        print(f"{i+1}. {text}")
        print(f"   Metadata: {metadata}")
    print()


def example_get_and_update():
    """Get and update documents"""
    print("=" * 60)
    print("Example 5: Get and Update")
    print("=" * 60)
    
    db = VectorDB(storage_path="./my_vectordb")
    
    # Get documents by metadata
    results = db.get(where={"category": "science"})
    print(f"Found {len(results['ids'])} science documents")
    
    if results['ids']:
        # Update first document
        doc_id = results['ids'][0]
        print(f"\nUpdating document: {doc_id[:8]}...")
        
        db.update(
            doc_id,
            metadata={"updated": True, "timestamp": "2025-11-16"},
            text="Updated text content"
        )
        
        # Verify update
        updated = db.get(doc_ids=[doc_id])
        print(f"Updated metadata: {updated['metadatas'][0]}")
        print(f"Updated text: {updated['documents'][0]}")
    print()


def example_persistence():
    """Test persistence across sessions"""
    print("=" * 60)
    print("Example 6: Persistence Test")
    print("=" * 60)
    
    # Create new database instance (loads from disk)
    db_reload = VectorDB(storage_path="./my_vectordb")
    
    print(f"Reloaded database: {db_reload}")
    print(f"Documents in reloaded DB: {db_reload.count()}")
    
    # Verify data integrity
    results = db_reload.get()
    print(f"\nFirst 3 documents:")
    for i, (doc_id, text) in enumerate(zip(results['ids'][:3], results['documents'][:3])):
        print(f"{i+1}. {text}")
    print()


def example_performance_benchmark():
    """Performance benchmarking"""
    print("=" * 60)
    print("Example 7: Performance Benchmark")
    print("=" * 60)
    
    db = VectorDB(storage_path="./benchmark_db", dimension=384)
    db.clear()
    
    # Test different dataset sizes
    sizes = [100, 500, 1000]
    
    for size in sizes:
        db.clear()
        
        # Add documents
        embeddings = [generate_sample_embedding(384, seed=i) for i in range(size)]
        texts = [f"Document {i}" for i in range(size)]
        
        start = time.time()
        db.add_batch(embeddings, texts=texts)
        add_time = time.time() - start
        
        # Query
        query_embedding = generate_sample_embedding(384, seed=9999)
        
        start = time.time()
        results = db.query(query_embedding, n_results=10)
        query_time = time.time() - start
        
        print(f"Size: {size:4d} docs | Add: {add_time:.4f}s | Query: {query_time*1000:.2f}ms")
    
    print()


def example_delete_operations():
    """Delete operations"""
    print("=" * 60)
    print("Example 8: Delete Operations")
    print("=" * 60)
    
    db = VectorDB(storage_path="./my_vectordb")
    
    initial_count = db.count()
    print(f"Initial document count: {initial_count}")
    
    # Delete by metadata
    deleted = db.delete(where={"category": "science"})
    print(f"Deleted {deleted} science documents")
    print(f"Remaining documents: {db.count()}")
    
    db.save()
    print()


if __name__ == "__main__":
    # Run all examples
    example_basic_usage()
    example_batch_operations()
    example_similarity_search()
    example_metadata_filtering()
    example_get_and_update()
    example_persistence()
    example_performance_benchmark()
    example_delete_operations()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
