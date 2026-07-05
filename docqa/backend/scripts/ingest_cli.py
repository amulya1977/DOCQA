"""Command-line ingestion: `python -m scripts.ingest_cli path/to/file.pdf`

Handy for bulk-loading documents without going through the UI, and for
demonstrating the pipeline in isolation.
"""
import sys
import os

# Make the `app` package importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import get_conn
from app.extract import extract_text
from app.ingest import ingest_text


def main():
    if len(sys.argv) < 2:
        print("usage: python -m scripts.ingest_cli <file> [<file> ...]")
        sys.exit(1)

    conn = get_conn()
    try:
        for path in sys.argv[1:]:
            with open(path, "rb") as f:
                raw = f.read()
            text = extract_text(os.path.basename(path), raw)
            result = ingest_text(conn, os.path.basename(path), text)
            print(f"{path}: {result}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
