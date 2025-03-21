#!/usr/bin/env python3
"""
Vectorize MCP servers from JSON into Chroma using LangChain.
Uses sentence-transformers/all-MiniLM-L6-v2 for embeddings and retrieval.
"""

import json
import os
from typing import Dict, List, Any

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Constants
MCP_SERVERS_FILE = "mcp-servers.json"
PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "mcp_servers"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_mcp_servers(file_path: str) -> List[Dict[str, Any]]:
    """Load MCP servers from JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return data["servers"]

def create_documents_from_servers(servers: List[Dict[str, Any]]) -> List[Document]:
    """Convert server data to Document objects for vectorization."""
    documents = []
    
    for i, server in enumerate(servers):
        # Create a rich text representation for each server
        content = f"""
        Name: {server.get('displayName', '')}
        Qualified Name: {server.get('qualifiedName', '')}
        Description: {server.get('description', '')}
        Homepage: {server.get('homepage', '')}
        Use Count: {server.get('useCount', 0)}
        Created At: {server.get('createdAt', '')}
        """
        
        # Create a Document with the content and metadata
        doc = Document(
            page_content=content,
            metadata={
                "id": i,
                "qualifiedName": server.get("qualifiedName", ""),
                "displayName": server.get("displayName", ""),
                "description": server.get("description", ""),
                "homepage": server.get("homepage", ""),
                "useCount": server.get("useCount", 0),
                "createdAt": server.get("createdAt", "")
            }
        )
        documents.append(doc)
    
    return documents

def vectorize_documents(documents: List[Document]) -> Chroma:
    """Vectorize documents and store in Chroma."""
    # Initialize the embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # Create or load the vector store
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME
    )
    
    # Persist the vector store
    vector_store.persist()
    
    return vector_store

def query_vector_store(vector_store: Chroma, query: str, k: int = 5) -> List[Document]:
    """Query the vector store for similar documents."""
    # Create a retriever from the vector store
    retriever = vector_store.as_retriever(
        search_kwargs={"k": k}
    )
    
    # Retrieve similar documents
    docs = retriever.get_relevant_documents(query)
    
    return docs

def main():
    """Main function to vectorize MCP servers and demonstrate retrieval."""
    print(f"Loading MCP servers from {MCP_SERVERS_FILE}...")
    servers = load_mcp_servers(MCP_SERVERS_FILE)
    print(f"Loaded {len(servers)} MCP servers.")
    
    print(f"Creating documents from server data...")
    documents = create_documents_from_servers(servers)
    
    print(f"Vectorizing documents using {MODEL_NAME}...")
    vector_store = vectorize_documents(documents)
    print(f"Vectorization complete. Data stored in {PERSIST_DIRECTORY}")
    
    # Example queries to demonstrate retrieval
    example_queries = [
        "GitHub integration",
        "Database management",
        "Web search capabilities",
        "Task management tools",
        "Browser automation"
    ]
    
    print("\nDemonstrating retrieval with example queries:")
    for query in example_queries:
        print(f"\nQuery: '{query}'")
        results = query_vector_store(vector_store, query, k=3)
        
        print(f"Top {len(results)} results:")
        for i, doc in enumerate(results):
            print(f"  {i+1}. {doc.metadata.get('displayName')} ({doc.metadata.get('qualifiedName')})")
            print(f"     Homepage: {doc.metadata.get('homepage')}")
            print(f"     Use Count: {doc.metadata.get('useCount')}")

if __name__ == "__main__":
    main()
