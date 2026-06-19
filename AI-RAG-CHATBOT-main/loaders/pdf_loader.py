import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pypdf import PdfReader
from loguru import logger
from typing import List, Dict, Any
import re

def load_pdf(file_path: str) -> List[Dict[str, Any]]:
    path = Path(file_path)
    if not path.exists():
        logger.error("PDF not found: {}", file_path)
        return []
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text(extraction_mode="layout") or ""
        text = _clean_text(text)
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    logger.info("Loaded {} pages from {}", len(pages), path.name)
    return pages

def _clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

if __name__ == "__main__":
    import sys
    pages = load_pdf(sys.argv[1])
    for p in pages:
        print(p["text"])