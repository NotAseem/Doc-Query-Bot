# Doc-Querier

A powerful document querying system that enables semantic search and question answering over your documents.

## Features

- Document embedding and storage using ChromaDB
- Semantic search capabilities
- Question answering over stored documents
- Caching system for improved performance

## System Requirements

- Microsoft Visual C++ Build Tools
- Llama model (for text generation and embeddings)
- Python 3.8 or higher

## Setup

1. Install Microsoft Visual C++ Build Tools
2. Install Llama model and set up environment
3. Install dependencies: `pip install -r requirements.txt`
4. Run `populate_database.py` to add your documents
5. Use `query_data.py` for querying your documents

## Usage

```python
# Query your documents
python query_data.py "Your question here"
```

## Requirements

See `requirements.txt` for full list of dependencies.
