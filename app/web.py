from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Optional

import markdown
from bs4 import BeautifulSoup
from fastapi.templating import Jinja2Templates
from markupsafe import Markup


templates = Jinja2Templates(directory="app/templates")


def render_markdown(text: Optional[str]) -> Markup:
    if not text:
        return Markup("")
    html = markdown.markdown(text, extensions=["extra", "nl2br", "sane_lists"])
    return Markup(html)


def render_markdown_text(text: Optional[str]) -> str:
    if not text:
        return ""
    html = markdown.markdown(text, extensions=["extra", "nl2br", "sane_lists"])
    soup = BeautifulSoup(html, "html.parser")
    return re.sub(r"\s+", " ", soup.get_text(" ")).strip()


def format_dt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        candidate = value.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(candidate)
        except ValueError:
            return value
    else:
        return str(value)
    return dt.strftime("%d:%m:%Y %H:%M:%S")


templates.env.filters["markdown"] = render_markdown
templates.env.filters["markdown_text"] = render_markdown_text
templates.env.filters["format_dt"] = format_dt
