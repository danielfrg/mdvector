# mdvector

`mdvector` is a tool for syncing markdown files to a vector databases.

Currently, it supports [Qdrant](https://qdrant.tech/).

## Usage

```bash
mdvector sync <directory>
mdvector search <query>
```

Configure with environment variables:

- `MDVECTOR_QDRANT_URL`
- `MDVECTOR_QDRANT_COLLECTION`
