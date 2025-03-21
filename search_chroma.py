#!/usr/bin/env python3
"""
Search Chroma database of MCP servers.
Usage: python search_chroma.py "your search query" [number_of_results]
"""

import chromadb
import json
import sys

# Constants
PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "mcp_servers"

def search_collection(query_text, n_results=5):
    """Search the collection for documents matching the query."""
    # Show full path to database
    import os
    full_path = os.path.abspath(PERSIST_DIRECTORY)
    print(f"Using Chroma database at: {full_path}")
    
    # Connect to the Chroma client
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    
    # Get the collection
    collection = client.get_collection(COLLECTION_NAME)
    
    # Search
    print(f"Searching for: '{query_text}'")
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    
    print(f"\nFound {len(results['ids'][0])} results:")
    for i, (doc_id, metadata) in enumerate(zip(
        results['ids'][0], 
        results['metadatas'][0]
    )):
        print(f"\n--- Result {i+1} ---")
        print(f"Name: {metadata.get('displayName', 'Unknown')}")
        print(f"Qualified Name: {metadata.get('qualifiedName', 'Unknown')}")
        print(f"Description: {metadata.get('description', 'No description available')}")
        print(f"Homepage: {metadata.get('homepage', 'Unknown')}")
        print(f"Use Count: {metadata.get('useCount', 0)}")
        print(f"Distance: {results['distances'][0][i]}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python search_chroma.py \"your search query\" [number_of_results]")
        sys.exit(1)
    
    query = sys.argv[1]
    n_results = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    search_collection(query, n_results)
