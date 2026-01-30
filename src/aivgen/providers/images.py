from __future__ import annotations

import base64


def parse_data_url(url: str) -> tuple[str, bytes]:
    if not url.startswith("data:"):
        raise ValueError(f"Invalid data URL: {url[:50]}...")

    content = url[5:]
    comma_idx = content.find(",")
    if comma_idx == -1:
        raise ValueError("Invalid data URL: missing comma separator")

    metadata = content[:comma_idx]
    b64_data = content[comma_idx + 1 :]

    parts = metadata.split(";")
    mime_type = parts[0] if parts[0] else "application/octet-stream"
    is_base64 = "base64" in parts

    if is_base64:
        return mime_type, base64.b64decode(b64_data)

    from urllib.parse import unquote

    return mime_type, unquote(b64_data).encode("utf-8")
