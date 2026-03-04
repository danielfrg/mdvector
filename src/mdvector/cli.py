import logging
import os
from pathlib import Path

import click

DEFAULT_MODEL = "onnx-community/embeddinggemma-300m-ONNX"
DEFAULT_QDRANT_URL = os.environ.get("MDVECTOR_QDRANT_URL", "http://localhost:6333")
DEFAULT_COLLECTION = os.environ.get("MDVECTOR_QDRANT_COLLECTION", "mdvector")


def configure_logging(verbose: bool, default_level: int = logging.INFO) -> None:
    """Set up logging, suppressing noisy third-party loggers."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else default_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for name in ("httpx", "httpcore", "urllib3", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)


def normalize_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    return url


@click.group()
def main() -> None:
    """Sync markdown files to a Qdrant vector database."""


@main.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--qdrant-url",
    default=DEFAULT_QDRANT_URL,
    show_default=True,
    help="Qdrant server URL.",
)
@click.option(
    "--collection",
    default=DEFAULT_COLLECTION,
    show_default=True,
    help="Qdrant collection name.",
)
@click.option(
    "--model",
    default=DEFAULT_MODEL,
    show_default=True,
    help="Sentence-transformers model.",
)
@click.option(
    "--batch-size",
    type=int,
    default=32,
    show_default=True,
    help="Batch size for embedding.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Re-embed and upsert all files regardless of content hash.",
)
@click.option(
    "--dry-run", is_flag=True, help="Show what would change without modifying Qdrant."
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
def sync(
    directory: Path,
    qdrant_url: str,
    collection: str,
    model: str,
    batch_size: int,
    force: bool,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Sync markdown files from a directory into Qdrant."""
    configure_logging(verbose)

    from .sync import sync as do_sync

    url = normalize_url(qdrant_url)

    result = do_sync(
        directory=directory,
        qdrant_url=url,
        collection=collection,
        model_name=model,
        batch_size=batch_size,
        force=force,
        dry_run=dry_run,
    )

    prefix = "[dry-run] " if dry_run else ""
    click.echo(
        f"{prefix}Done: "
        f"{result.upserted} upserted, "
        f"{result.deleted} deleted, "
        f"{result.unchanged} unchanged "
        f"({result.total_on_disk} files on disk)"
    )


@main.command()
@click.argument("query", nargs=-1, required=True)
@click.option(
    "--qdrant-url",
    default=DEFAULT_QDRANT_URL,
    show_default=True,
    help="Qdrant server URL.",
)
@click.option(
    "--collection",
    default=DEFAULT_COLLECTION,
    show_default=True,
    help="Qdrant collection name.",
)
@click.option(
    "--model",
    default=DEFAULT_MODEL,
    show_default=True,
    help="Sentence-transformers model.",
)
@click.option(
    "--limit",
    "-n",
    type=int,
    default=10,
    show_default=True,
    help="Number of results to return.",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
def search(
    query: tuple[str, ...],
    qdrant_url: str,
    collection: str,
    model: str,
    limit: int,
    verbose: bool,
) -> None:
    """Search the VectorDB collection with a text query."""
    configure_logging(verbose, default_level=logging.WARNING)

    from .embed import embed_texts, load_model
    from .store import QdrantStore

    url = normalize_url(qdrant_url)
    query_text = " ".join(query)

    loaded_model = load_model(model, show_progress=False)
    vectors = embed_texts(loaded_model, [query_text])
    query_vector = vectors[0]

    store = QdrantStore(url=url)
    try:
        results = store.search(
            collection=collection,
            vector=query_vector,
            limit=limit,
        )
    finally:
        store.close()

    if not results:
        click.echo("No results found.")
        return

    for i, r in enumerate(results):
        preview = r.content[:200].replace("\n", " ")
        if len(r.content) > 200:
            preview += "..."
        click.echo(f"{i+1}. {r.path} - score={r.score:.4f}\n    {preview}\n")


@main.command("download-model")
@click.option(
    "--model", default=DEFAULT_MODEL, show_default=True, help="Model to download."
)
def download_model(model: str) -> None:
    """Download the embedding model from HuggingFace."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from .embed import get_dimension, load_model

    loaded_model = load_model(model)
    dim = get_dimension(loaded_model)
    click.echo(f"Model ready: {model} (dimension={dim})")


if __name__ == "__main__":
    main()
