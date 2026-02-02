from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from app.services.llm import extract_text_from_images


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text:
            parts.append(text)
    if parts:
        return "\n\n".join(parts)

    image_bytes: list[bytes] = []
    for page in reader.pages[:3]:
        page_images = getattr(page, "images", [])
        for image in page_images:
            data = getattr(image, "data", None)
            if data:
                image_bytes.append(data)
        if len(image_bytes) >= 3:
            break
    if not image_bytes:
        return ""
    return extract_text_from_images(image_bytes)
