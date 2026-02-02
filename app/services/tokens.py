from __future__ import annotations

from typing import Optional

import tiktoken


def _encoding_for_model(model: Optional[str]):
    if model:
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            pass
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: Optional[str] = None) -> int:
    if not text:
        return 0
    encoding = _encoding_for_model(model)
    return len(encoding.encode(text))
