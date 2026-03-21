import argparse
import logging
from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from meno_core.config.settings import settings  # noqa: E402
from meno_core.core.rag.ingestion.corpus_builder import DEFAULT_INPUT_PATHS, build_chunk_rag_corpus  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a unified Chunk-RAG corpus JSONL from all resources/data sources."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=list(DEFAULT_INPUT_PATHS),
        help="Input files to merge into the output corpus.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=settings.chunk_rag_corpus_path,
        help="Output JSONL path for the generated chunk corpus.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum token count for each emitted chunk, including the metadata prefix.",
    )

    empty_group = parser.add_mutually_exclusive_group()
    empty_group.add_argument("--drop-empty", dest="drop_empty", action="store_true")
    empty_group.add_argument("--keep-empty", dest="drop_empty", action="store_false")
    parser.set_defaults(drop_empty=True)

    not_found_group = parser.add_mutually_exclusive_group()
    not_found_group.add_argument("--drop-404", dest="drop_404", action="store_true")
    not_found_group.add_argument("--keep-404", dest="drop_404", action="store_false")
    parser.set_defaults(drop_404=True)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    stats = build_chunk_rag_corpus(
        args.inputs,
        output_jsonl=args.output_jsonl,
        max_tokens=args.max_tokens,
        drop_empty=args.drop_empty,
        drop_404=args.drop_404,
    )
    logging.info(
        "Done: %s input documents, %s deduplicated documents, %s chunks -> %s",
        stats.input_documents,
        stats.deduplicated_documents,
        stats.emitted_chunks,
        args.output_jsonl,
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
