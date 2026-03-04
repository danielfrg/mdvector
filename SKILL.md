---
name: doc-search
description: |
  mdvector CLI tool for searching documentsin a vector database.

  Use for:
  - Searching for documents, notes, meetings transcripts, or other type of knowledge questions.

requirements:
  - command: mdvector
    install: "uv tool install mdvector"
---

# doc-search

Use the `mdvector search` command to search for notes in a vector database
and get similar notes to a query with a similarity score.

The full notes are stored in markdown files in: ~/Documents/notes

Based on the user requests creata a query string and search for it in the vector database. After that, if the question of the user is specific, read one or more notes (the output includes the path to the note) and answer the question.

```
mdvector search [-h] [--qdrant-url QDRANT_URL] [--collection COLLECTION]
                       [--model MODEL] [--limit LIMIT] [--verbose]
                       query [query ...]

positional arguments:
  query                 Search query text

options:
  -h, --help            show this help message and exit
  --qdrant-url QDRANT_URL
                        Qdrant server URL
  --collection COLLECTION
                        Qdrant collection name (default: notes)
  --model MODEL         Sentence-transformers model (default: onnx-community/embeddinggemma-300m-ONNX)
  --limit, -n LIMIT     Number of results to return (default: 10)
  --verbose, -v         Enable verbose logging
```

The `QDRANT_URL` and `QDRANT_COLLECTION` should be configured already based on environment variables.

The output looks like this:

```
{id}. {note path} | score={score}
    {short note content}
```
