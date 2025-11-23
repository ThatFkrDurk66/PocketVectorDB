"""
Pocket Vector Database - Fast, persistent vector storage with semantic search
"""
import numpy as np
import pickle
import json
import os
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import hashlib
from pathlib import Path


@dataclass
class Document:
    """Document with vector embedding and metadata"""
    id: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    text: Optional[str] = None
    
    def to_dict(self):
        return {
            'id': self.id,
            'embedding': self.embedding.tolist(),
            'metadata': self.metadata,
            'text': self.text
        }


class VectorDB:
    """
    Fast personal vector database with persistent storage and semantic search.
    
    Features:
    - Persistent storage using efficient binary format
    - Fast similarity search using numpy operations
    - Support for metadata filtering
    - CRUD operations (Create, Read, Update, Delete)
    - Batch operations for efficiency
    """
    
    def __init__(self, storage_path: str = "./vectordb_storage", dimension: Optional[int] = None):
        """
        Initialize VectorDB
        
        Args:
            storage_path: Directory to store database files
            dimension: Vector dimension (auto-detected from first insert if None)
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.dimension = dimension
        self.documents: Dict[str, Document] = {}
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.doc_ids: List[str] = []
        
        # File paths
        self.embeddings_file = self.storage_path / "embeddings.npy"
        self.metadata_file = self.storage_path / "metadata.json"
        self.index_file = self.storage_path / "index.pkl"
        
        # Load existing data
        self.load()
    
    def _generate_id(self, text: str) -> str:
        """Generate unique ID from text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector for cosine similarity"""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
    
    def _rebuild_matrix(self):
        """Rebuild embeddings matrix from documents"""
        if not self.documents:
            self.embeddings_matrix = None
            self.doc_ids = []
            return
        
        self.doc_ids = list(self.documents.keys())
        embeddings_list = [self.documents[doc_id].embedding for doc_id in self.doc_ids]
        self.embeddings_matrix = np.vstack(embeddings_list)
    
    def add(self, 
            embedding: Union[np.ndarray, List[float]], 
            metadata: Optional[Dict[str, Any]] = None,
            text: Optional[str] = None,
            doc_id: Optional[str] = None) -> str:
        """
        Add a document with its embedding
        
        Args:
            embedding: Vector embedding (will be normalized)
            metadata: Optional metadata dictionary
            text: Optional original text
            doc_id: Optional document ID (auto-generated if None)
        
        Returns:
            Document ID
        """
        # Convert to numpy array
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
        else:
            embedding = embedding.astype(np.float32)
        
        # Set dimension if first document
        if self.dimension is None:
            self.dimension = len(embedding)
        elif len(embedding) != self.dimension:
            raise ValueError(f"Embedding dimension {len(embedding)} doesn't match database dimension {self.dimension}")
        
        # Generate ID if not provided
        if doc_id is None:
            if text:
                doc_id = self._generate_id(text)
            else:
                doc_id = self._generate_id(str(embedding[:10]))
        
        # Normalize embedding
        embedding = self._normalize_vector(embedding)
        
        # Create document
        doc = Document(
            id=doc_id,
            embedding=embedding,
            metadata=metadata or {},
            text=text
        )
        
        # Add to documents
        self.documents[doc_id] = doc
        
        # Rebuild matrix
        self._rebuild_matrix()
        
        return doc_id
    
    def add_batch(self, 
                  embeddings: List[Union[np.ndarray, List[float]]], 
                  metadatas: Optional[List[Dict[str, Any]]] = None,
                  texts: Optional[List[str]] = None,
                  doc_ids: Optional[List[str]] = None) -> List[str]:
        """
        Add multiple documents efficiently
        
        Args:
            embeddings: List of vector embeddings
            metadatas: Optional list of metadata dictionaries
            texts: Optional list of original texts
            doc_ids: Optional list of document IDs
        
        Returns:
            List of document IDs
        """
        if metadatas is None:
            metadatas = [{}] * len(embeddings)
        if texts is None:
            texts = [None] * len(embeddings)
        if doc_ids is None:
            doc_ids = [None] * len(embeddings)
        
        added_ids = []
        for emb, meta, text, doc_id in zip(embeddings, metadatas, texts, doc_ids):
            added_id = self.add(emb, meta, text, doc_id)
            added_ids.append(added_id)
        
        return added_ids
    
    def query(self, 
              query_embedding: Union[np.ndarray, List[float]], 
              n_results: int = 10,
              where: Optional[Dict[str, Any]] = None,
              include_distances: bool = True) -> Dict[str, Any]:
        """
        Query for similar vectors using cosine similarity
        
        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            where: Optional metadata filter (exact match)
            include_distances: Include similarity distances in results
        
        Returns:
            Dictionary with ids, documents, metadatas, and optionally distances
        """
        if self.embeddings_matrix is None or len(self.documents) == 0:
            return {
                'ids': [],
                'documents': [],
                'metadatas': [],
                'distances': [] if include_distances else None
            }
        
        # Convert to numpy array and normalize
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding, dtype=np.float32)
        query_embedding = self._normalize_vector(query_embedding)
        
        # Filter by metadata if specified
        if where:
            valid_indices = []
            for i, doc_id in enumerate(self.doc_ids):
                doc = self.documents[doc_id]
                match = all(doc.metadata.get(k) == v for k, v in where.items())
                if match:
                    valid_indices.append(i)
            
            if not valid_indices:
                return {
                    'ids': [],
                    'documents': [],
                    'metadatas': [],
                    'distances': [] if include_distances else None
                }
            
            filtered_embeddings = self.embeddings_matrix[valid_indices]
            filtered_doc_ids = [self.doc_ids[i] for i in valid_indices]
        else:
            filtered_embeddings = self.embeddings_matrix
            filtered_doc_ids = self.doc_ids
        
        # Compute cosine similarity (dot product since vectors are normalized)
        similarities = np.dot(filtered_embeddings, query_embedding)
        
        # Get top k results
        n_results = min(n_results, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:n_results]
        
        # Prepare results
        result_ids = [filtered_doc_ids[i] for i in top_indices]
        result_docs = [self.documents[doc_id] for doc_id in result_ids]
        
        results = {
            'ids': result_ids,
            'documents': [doc.text for doc in result_docs],
            'metadatas': [doc.metadata for doc in result_docs],
        }
        
        if include_distances:
            # Convert similarity to distance (1 - similarity for cosine distance)
            results['distances'] = [1 - similarities[i] for i in top_indices]
        
        return results
    
    def get(self, doc_ids: Optional[List[str]] = None, 
            where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get documents by IDs or metadata filter
        
        Args:
            doc_ids: List of document IDs to retrieve
            where: Optional metadata filter
        
        Returns:
            Dictionary with ids, documents, metadatas, embeddings
        """
        if doc_ids:
            docs = [self.documents.get(doc_id) for doc_id in doc_ids]
            docs = [d for d in docs if d is not None]
        elif where:
            docs = [doc for doc in self.documents.values() 
                   if all(doc.metadata.get(k) == v for k, v in where.items())]
        else:
            docs = list(self.documents.values())
        
        return {
            'ids': [doc.id for doc in docs],
            'documents': [doc.text for doc in docs],
            'metadatas': [doc.metadata for doc in docs],
            'embeddings': [doc.embedding.tolist() for doc in docs]
        }
    
    def delete(self, doc_ids: Optional[List[str]] = None, 
               where: Optional[Dict[str, Any]] = None) -> int:
        """
        Delete documents by IDs or metadata filter
        
        Args:
            doc_ids: List of document IDs to delete
            where: Optional metadata filter
        
        Returns:
            Number of documents deleted
        """
        if doc_ids:
            to_delete = set(doc_ids) & set(self.documents.keys())
        elif where:
            to_delete = {doc_id for doc_id, doc in self.documents.items()
                        if all(doc.metadata.get(k) == v for k, v in where.items())}
        else:
            return 0
        
        for doc_id in to_delete:
            del self.documents[doc_id]
        
        self._rebuild_matrix()
        return len(to_delete)
    
    def update(self, doc_id: str, 
               embedding: Optional[Union[np.ndarray, List[float]]] = None,
               metadata: Optional[Dict[str, Any]] = None,
               text: Optional[str] = None):
        """
        Update a document's embedding, metadata, or text
        
        Args:
            doc_id: Document ID to update
            embedding: New embedding (optional)
            metadata: New metadata (optional, will be merged)
            text: New text (optional)
        """
        if doc_id not in self.documents:
            raise KeyError(f"Document {doc_id} not found")
        
        doc = self.documents[doc_id]
        
        if embedding is not None:
            if isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)
            doc.embedding = self._normalize_vector(embedding)
            self._rebuild_matrix()
        
        if metadata is not None:
            doc.metadata.update(metadata)
        
        if text is not None:
            doc.text = text
    
    def count(self) -> int:
        """Get total number of documents"""
        return len(self.documents)
    
    def save(self):
        """Save database to disk"""
        if not self.documents:
            return
        
        # Save embeddings matrix
        if self.embeddings_matrix is not None:
            np.save(self.embeddings_file, self.embeddings_matrix)
        
        # Save metadata and texts
        metadata_data = {
            'dimension': self.dimension,
            'doc_ids': self.doc_ids,
            'documents': {
                doc_id: {
                    'metadata': doc.metadata,
                    'text': doc.text
                }
                for doc_id, doc in self.documents.items()
            }
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata_data, f)
    
    def load(self):
        """Load database from disk"""
        if not self.metadata_file.exists():
            return
        
        try:
            # Load metadata
            with open(self.metadata_file, 'r') as f:
                metadata_data = json.load(f)
            
            self.dimension = metadata_data.get('dimension')
            self.doc_ids = metadata_data.get('doc_ids', [])
            
            # Load embeddings
            if self.embeddings_file.exists():
                self.embeddings_matrix = np.load(self.embeddings_file)
            
            # Reconstruct documents
            documents_data = metadata_data.get('documents', {})
            for i, doc_id in enumerate(self.doc_ids):
                doc_data = documents_data[doc_id]
                self.documents[doc_id] = Document(
                    id=doc_id,
                    embedding=self.embeddings_matrix[i] if self.embeddings_matrix is not None else None,
                    metadata=doc_data['metadata'],
                    text=doc_data['text']
                )
        
        except Exception as e:
            print(f"Warning: Could not load database: {e}")
            self.documents = {}
            self.embeddings_matrix = None
            self.doc_ids = []
    
    def clear(self):
        """Clear all documents from database"""
        self.documents = {}
        self.embeddings_matrix = None
        self.doc_ids = []
    
    def __len__(self):
        return len(self.documents)
    
    def __repr__(self):
        return f"VectorDB(documents={len(self.documents)}, dimension={self.dimension}, storage='{self.storage_path}')"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
