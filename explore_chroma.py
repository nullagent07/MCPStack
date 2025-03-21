#!/usr/bin/env python3
"""
Explore Chroma database of MCP servers.
"""

import chromadb
import json
import sys

# Constants
PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "mcp_servers"

def list_collections(client):
    """List all collections in the database."""
    collections = client.list_collections()
    print(f"Found {len(collections)} collections:")
    for i, collection in enumerate(collections):
        print(f"  {i+1}. {collection.name} (count: {collection.count()})")
    return collections

def explore_collection(collection):
    """Explore a collection, showing metadata and sample documents."""
    count = collection.count()
    print(f"\nCollection '{collection.name}' contains {count} documents.")
    
    # Get collection info
    print("\nCollection info:")
    print(json.dumps(collection.get(), indent=2))
    
    # Show a few sample documents
    limit = min(5, count)
    print(f"\nShowing {limit} sample documents:")
    results = collection.get(limit=limit)
    
    for i in range(limit):
        print(f"\n--- Document {i+1} ---")
        print(f"ID: {results['ids'][i]}")
        print(f"Metadata: {json.dumps(results['metadatas'][i], indent=2)}")
        print(f"Document: {results['documents'][i][:200]}...")  # Show first 200 chars

def search_collection(collection, query_text, n_results=5):
    """Search the collection for documents matching the query."""
    print(f"\nSearching for: '{query_text}'")
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    
    print(f"\nFound {len(results['ids'][0])} results:")
    for i, (doc_id, doc, metadata) in enumerate(zip(
        results['ids'][0], 
        results['documents'][0], 
        results['metadatas'][0]
    )):
        print(f"\n--- Result {i+1} ---")
        print(f"ID: {doc_id}")
        print(f"Metadata: {json.dumps(metadata, indent=2)}")
        print(f"Document: {doc[:200]}...")  # Show first 200 chars
        print(f"Distance: {results['distances'][0][i]}")

def main():
    """Main function to explore Chroma database."""
    # Connect to the Chroma client
    print(f"Connecting to Chroma database at {PERSIST_DIRECTORY}...")
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    
    # List collections
    collections = list_collections(client)
    
    if not collections:
        print("No collections found in the database.")
        return
    
    # Get the collection
    collection = client.get_collection(COLLECTION_NAME)
    
    # Explore the collection
    explore_collection(collection)
    
    # Interactive search
    while True:
        print("\n--- Search Options ---")
        print("1. Search by query")
        print("2. Exit")
        
        choice = input("Enter your choice (1-2): ")
        
        if choice == "1":
            query = input("Enter your search query: ")
            n_results = int(input("Number of results to show (default 5): ") or "5")
            search_collection(collection, query, n_results)
        elif choice == "2":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
