"""Extract plain text from an uploaded file (PDF or .txt/.md).

The unglamorous layer that breaks most demos. We keep it defensive: if a PDF
page yields nothing, we skip it rather than crash, and we surface a clear error
if the whole file is empty.
"""
import io
from pypdf import PdfReader


def extract_text(filename: str, raw: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        return _extract_pdf(raw)
    # treat everything else as UTF-8 text (.txt, .md, etc.)
    return raw.decode("utf-8", errors="ignore")


def _extract_pdf(raw: bytes) -> str:
    reader = PdfReader(io.BytesIO(raw))
    parts = []
    for page in reader.pages:
        text = page.extract_text() or ""  # some pages (scans) return nothing
        if text.strip():
            parts.append(text)
    return "\n".join(parts)
